"""The pixel-domain ResNet image decoder (phase2.image_decoder: resnet_pixel).

It restores the image from the blurry input pixels plus the JEPA latent. A local
A/B with the same phase 1 and loss had it reproduce more of the clean edge
content in place than the QWT-coefficient decoder (0.359 vs 0.309). These
tests pin what it must keep from the old decoder:
an identity start, a frozen backbone, an unchanged phase 1, and checkpoints that
rebuild into the network they were trained as.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.models.decoders import LatentCoefficientDecoder, PixelResNetDecoder
from qjepa.training.checkpoints import configuration_hash, state_dict_hash
from qjepa.training.phase2 import Phase2Trainer


def _config(**phase2):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(image_decoder="resnet_pixel", image_resnet_width=16,
                            image_resnet_blocks=2, **phase2)
    validate_config(config)
    return config


def _batch(batch_size=2):
    times = torch.arange(32).float().mul(0.01).repeat(batch_size, 1)
    clean = torch.rand(batch_size, 3, 32, 32)
    imu = torch.randn(batch_size, 6, 32)
    return {
        "image_clean": clean,
        "image_noisy": (clean * 0.35 + 0.03 * torch.randn_like(clean)).clamp(0, 1),
        "imu_clean_phys": imu,
        "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu),
        "image_time": times.mean(dim=1),
        "imu_times": times,
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
    }


def _system(config):
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    return phase1, RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))


def _restore(system, batch):
    return system(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])


def test_starts_as_the_identity_on_the_blurry_input():
    """Update 0 returns the input exactly, so training can only move up from it."""
    _, system = _system(_config())
    assert isinstance(system.decoders.image, PixelResNetDecoder)
    batch = _batch()
    restored = _restore(system.eval(), batch)
    assert torch.allclose(restored.image, batch["image_noisy"], atol=1e-5)


def test_image_coefficients_are_those_of_the_restored_image():
    """No redundant QWT channels to hide energy in: they are analysis(image)."""
    _, system = _system(_config())
    restored = _restore(system.eval(), _batch())
    again, _ = system.backbone.image_transform.analysis(restored.image)
    assert torch.allclose(restored.image_coefficients, again, atol=1e-6)


def test_the_latent_reaches_the_output():
    _, system = _system(_config())
    with torch.no_grad():
        for parameter in system.decoders.image.tail[-1].parameters():
            parameter.normal_(0.0, 0.1)
    system.eval()
    batch = _batch()
    latent = system.encode(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
    with torch.no_grad():
        with_latent = system.decode(latent).image
        zeroed = system.decode(type(latent)(**{**latent.__dict__, "ZI": torch.zeros_like(latent.ZI)})).image
    assert float((with_latent - zeroed).abs().max()) > 1e-4


def test_phase1_anchor_stays_the_coefficient_decoder():
    """phase2.image_decoder must not change the phase-1 model or its hash."""
    config = _config()
    config["phase1"].update(decoder_enabled=True, coefficient_reconstruction_loss_weight=0.3)
    validate_config(config)
    phase1 = build_phase1_model(config, ImuNormalizer())
    assert isinstance(phase1.decoders.image, LatentCoefficientDecoder)
    plain = copy.deepcopy(config)
    for key in ("image_decoder", "image_resnet_width", "image_resnet_blocks"):
        plain["phase2"].pop(key)
    assert configuration_hash(config, "phase1") == configuration_hash(plain, "phase1")
    assert configuration_hash(config, "phase2") != configuration_hash(plain, "phase2")


def test_configs_written_before_the_key_rebuild_the_coefficient_decoder():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].pop("image_decoder", None)
    assert isinstance(build_decoders(config).image, LatentCoefficientDecoder)


def test_invalid_settings_are_rejected():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"]["image_decoder"] = "unet"
    with pytest.raises(ValueError, match="image_decoder"):
        validate_config(config)
    config["phase2"]["image_decoder"] = "resnet_pixel"
    config["phase2"].pop("image_resnet_width")
    with pytest.raises(ValueError, match="image_resnet_width"):
        validate_config(config)


def test_phase2_trains_the_resnet_and_leaves_the_backbone_alone() -> None:
    config = _config(image_detail_source="restored_image", reconstruction_detail_weight=2.0,
                     detail_energy_weight=1.0)
    _, system = _system(config)
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    backbone = state_dict_hash(system.backbone)
    before = [p.detach().clone() for p in system.decoders.image.parameters()]
    metrics = trainer.step([_batch(1)] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    assert metrics["image_detail_invisible_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert any(not torch.equal(a, b) for a, b in zip(before, system.decoders.image.parameters()))
    assert state_dict_hash(system.backbone) == backbone
    payload = trainer.checkpoint_payload(config)
    assert payload["metadata"]["image_decoder"] == "resnet_pixel"
    # The checkpoint rebuilds into the same network, strictly.
    _, rebuilt = _system(config)
    rebuilt.load_state_dict(payload["system"], strict=True)
    assert state_dict_hash(rebuilt.decoders) == state_dict_hash(system.decoders)
