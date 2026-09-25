"""The split decoder: colour and edges restored apart, then recombined.

These pin the rules the design rests on: the split is exact, the edge branch
can write luminance but never colour, the edge loss cannot train the colour
branch, the decoder starts at the input, and the ResNet layer names that p7
checkpoints were saved under did not move.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.evaluation.metrics import image_metrics
from qjepa.models import RestorationSystem
from qjepa.models.color_edge import chroma, color_base, compose, luminance, split_targets
from qjepa.models.decoders import PixelResNetDecoder, SplitColorEdgeDecoder
from qjepa.training.checkpoints import state_dict_hash
from qjepa.training.losses import color_edge_split_loss
from qjepa.training.phase2 import Phase2Trainer

SPLIT = dict(image_decoder="split_color_edge", split_color_width=8, split_color_blocks=2,
             split_edge_width=16, split_edge_blocks=2, split_color_scale=2,
             split_illumination_scale=8, split_color_weight=1.0, split_edge_weight=1.0,
             split_gradient_weight=0.0)


def _images(seed=0, size=64):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, size, size, generator=generator)


def test_the_split_is_exact():
    clean = _images()
    base, light, detail = split_targets(clean, 2, 8)
    rebuilt = compose(base, light, detail)
    assert torch.allclose(luminance(rebuilt), luminance(clean), atol=1e-6)
    assert torch.allclose(chroma(rebuilt), chroma(base), atol=1e-6)


def test_adding_the_same_value_to_rgb_changes_luminance_only():
    image = _images()
    shifted = image + 0.3 * torch.randn(2, 1, 64, 64)
    assert torch.allclose(chroma(shifted), chroma(image), atol=1e-6)
    assert not torch.allclose(luminance(shifted), luminance(image))


def _decoder():
    torch.manual_seed(0)
    return SplitColorEdgeDecoder(32, color_width=8, color_blocks=2, edge_width=16, edge_blocks=2)


def test_starts_at_the_input_luminance_and_colour():
    image, latent = _images(), torch.randn(2, 32, 4, 4)
    out, _ = _decoder()(latent, image)
    assert torch.allclose(luminance(out), luminance(image), atol=1e-6)
    assert torch.allclose(chroma(out), chroma(color_base(image, 2)), atol=1e-6)


def test_the_edge_branch_cannot_change_colour():
    decoder = _decoder()
    image, latent = _images(), torch.randn(2, 32, 4, 4)
    before, _ = decoder(latent, image)
    with torch.no_grad():
        for parameter in decoder.edge.tail[-1].parameters():
            parameter.normal_(0.0, 0.5)
    after, _ = decoder(latent, image)
    assert not torch.allclose(luminance(after), luminance(before))
    assert torch.allclose(chroma(after), chroma(before), atol=1e-6)


def test_the_edge_loss_does_not_train_the_colour_branch():
    decoder = _decoder()
    with torch.no_grad():
        for parameter in list(decoder.color.tail.parameters()) + list(decoder.edge.tail[-1].parameters()):
            parameter.normal_(0.0, 0.1)
    image, clean, latent = _images(0), _images(1), torch.randn(2, 32, 4, 4)
    out, parts = decoder(latent, image)
    _, loss_parts = color_edge_split_loss(parts["image_color_base"], parts["image_illumination"],
                                          parts["image_detail"], out, clean, color_scale=2,
                                          illumination_scale=8, color_weight=1.0, edge_weight=1.0,
                                          gradient_weight=0.0)
    loss_parts["image_edge_detail_l1"].backward()
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in decoder.color.parameters())
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in decoder.edge.parameters())


def test_resnet_layer_names_are_unchanged_for_p7_checkpoints():
    names = set(PixelResNetDecoder(128, 64, 8).state_dict())
    expected = {f"{layer}.{kind}" for layer in ("head.0", "down.0", "latent", "fuse", "up.0", "tail.0", "tail.2")
                for kind in ("weight", "bias")}
    expected |= {f"trunk.{i}.{conv}.{kind}" for i in range(8) for conv in ("first", "second")
                 for kind in ("weight", "bias")}
    assert names == expected


def test_colour_error_metric_sees_a_colour_cast_but_not_a_brightness_change():
    clean = _images(size=64).clamp(0.1, 0.9)
    cast = clean.clone(); cast[:, 2] += 0.1
    brighter = clean + 0.05
    assert image_metrics(clean, clean)["image_color_error"] == pytest.approx(0.0, abs=1e-7)
    assert image_metrics(cast, clean)["image_color_error"] > 0.01
    assert image_metrics(brighter, clean)["image_color_error"] == pytest.approx(0.0, abs=1e-6)


def test_missing_split_settings_are_rejected():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SPLIT)
    validate_config(config)
    config["phase2"].pop("split_edge_blocks")
    with pytest.raises(ValueError, match="split_edge_blocks"):
        validate_config(config)


@pytest.mark.parametrize("gradient_weight", [0.0, 1.0])
def test_phase2_trains_both_branches_and_leaves_the_backbone_alone(gradient_weight):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SPLIT, split_gradient_weight=gradient_weight,
                            image_detail_source="restored_image", reconstruction_detail_weight=2.0)
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    assert isinstance(system.decoders.image, SplitColorEdgeDecoder)
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    backbone = state_dict_hash(system.backbone)
    colour_before = [p.detach().clone() for p in system.decoders.image.color.parameters()]
    edge_before = [p.detach().clone() for p in system.decoders.image.edge.parameters()]
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean = torch.rand(1, 3, 32, 32)
    imu = torch.randn(1, 6, 32)
    batch = {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1),
             "imu_clean_phys": imu, "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu),
             "image_time": times.mean(dim=1), "imu_times": times, "sample_id": ["s"]}
    metrics = trainer.step([batch] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    assert {"image_color_l1", "image_edge_detail_l1"} <= set(metrics)
    assert ("image_edge_gradient_l1" in metrics) is (gradient_weight > 0)
    assert any(not torch.equal(a, b) for a, b in zip(colour_before, system.decoders.image.color.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(edge_before, system.decoders.image.edge.parameters()))
    assert state_dict_hash(system.backbone) == backbone
    payload = trainer.checkpoint_payload(config)
    assert payload["metadata"]["image_decoder"] == "split_color_edge"
    seed_everything(0)
    rebuilt = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    rebuilt.load_state_dict(payload["system"], strict=True)
    assert state_dict_hash(rebuilt.decoders) == state_dict_hash(system.decoders)
