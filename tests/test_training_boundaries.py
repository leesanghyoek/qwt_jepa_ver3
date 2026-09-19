import copy

import torch

import pytest

from qjepa.config import (
    build_decoders,
    build_phase1_model,
    load_config,
    seed_everything,
    validate_config,
)
from qjepa.models.blocks import Upsample, resize
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.training.checkpoints import require_phase1_checkpoint, state_dict_hash
from qjepa.training.phase1 import Phase1Trainer
from qjepa.training.phase2 import Phase2Trainer


def _batch(batch_size=2):
    times = torch.arange(32).float().mul(0.01).repeat(batch_size, 1)
    clean_image = torch.rand(batch_size, 3, 32, 32)
    clean_imu = torch.randn(batch_size, 6, 32)
    return {
        "image_clean": clean_image,
        "image_noisy": (clean_image * 0.35 + 0.03 * torch.randn_like(clean_image)).clamp(0, 1),
        "imu_clean_phys": clean_imu,
        "imu_noisy_phys": clean_imu + 0.05 * torch.randn_like(clean_imu),
        "image_time": times.mean(dim=1),
        "imu_times": times,
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
    }


def test_phase1_optimizer_excludes_teacher_and_checkpoint_provenance():
    config = load_config("configs/smoke.yaml")
    seed_everything(2)
    model = build_phase1_model(config, ImuNormalizer())
    trainer = Phase1Trainer(model, config, torch.device("cpu"), "manifest")
    optimized = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
    teachers = {id(parameter) for parameter in model.teachers.parameters()}
    assert optimized.isdisjoint(teachers)
    metrics = trainer.step(_batch())
    assert not metrics["skipped"]
    assert "image_l1" not in metrics and "reconstruction" not in metrics
    payload = trainer.checkpoint_payload(config)
    require_phase1_checkpoint(payload)


def test_phase2_changes_decoder_but_not_backbone():
    config = load_config("configs/smoke.yaml")
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    backbone_before = state_dict_hash(system.backbone)
    decoder_before = [parameter.detach().clone() for parameter in system.decoders.parameters()]
    one = _batch(1)
    metrics = trainer.step([one])
    assert not metrics["skipped"]
    assert state_dict_hash(system.backbone) == backbone_before
    assert any(
        not torch.equal(before, after)
        for before, after in zip(decoder_before, system.decoders.parameters())
    )
    trainer.assert_backbone_frozen()



def test_learned_upsampling_can_vary_inside_a_cell_but_bilinear_cannot():
    constant = torch.ones(1, 8, 4, 4)
    # Noi suy bilinear cua anh hang so van la hang so: no khong sinh duoc chi tiet.
    assert float(resize(constant, (8, 8), dim=2).std()) == pytest.approx(0.0, abs=1e-6)
    block = Upsample(8, dim=2)
    torch.nn.init.normal_(block.conv.weight, std=0.5)
    with torch.no_grad():
        learned = block(constant)
    assert tuple(learned.shape) == (1, 8, 8, 8)
    # Bon vi tri con trong cung mot o latent phai khac nhau duoc.
    assert float(learned[0, 0, :2, :2].std()) > 1e-3


def test_learned_upsampling_matches_sub_pixel_ordering_in_one_dimension():
    block = Upsample(3, dim=1, factor=2)
    packed = torch.randn(2, 3, 5)
    scales = torch.arange(1.0, 7.0)
    with torch.no_grad():
        block.conv.weight.zero_()
        block.conv.bias.zero_()
        # Kenh ra thu o chi lay kenh vao o // 2, nhan mot he so rieng biet.
        for out_channel in range(6):
            block.conv.weight[out_channel, out_channel // 2, 1] = scales[out_channel]
    output = block(packed)
    # Sub-pixel: out[b, c, l*r + j] phai lay tu kenh conv c*r + j.
    for channel in range(3):
        for position in range(5):
            for offset in range(2):
                expected = packed[:, channel, position] * scales[channel * 2 + offset]
                assert torch.allclose(output[:, channel, position * 2 + offset], expected, atol=1e-6)


def test_phase1_reconstruction_steers_the_latent_and_marks_the_checkpoint():
    config = load_config("configs/smoke.yaml")
    config["phase1"]["decoder_enabled"] = True
    config["phase1"]["coefficient_reconstruction_loss_weight"] = 0.3
    config["phase1"]["reconstruction_detail_weight"] = 0.5
    validate_config(config)
    seed_everything(5)
    model = build_phase1_model(config, ImuNormalizer())
    assert model.reconstructs
    trainer = Phase1Trainer(model, config, torch.device("cpu"), "manifest")
    optimized = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
    assert optimized.issuperset({id(parameter) for parameter in model.decoders.parameters()})
    before = [parameter.detach().clone() for parameter in model.decoders.parameters()]
    metrics = trainer.step(_batch())
    assert not metrics["skipped"]
    assert metrics["reconstruction"] > 0
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.decoders.parameters())
    )
    payload = trainer.checkpoint_payload(config)
    assert payload["metadata"]["trained_with_reconstruction"] is True
    assert payload["metadata"]["phase1_decoder_forward_calls"] == 1
    require_phase1_checkpoint(payload)


def test_phase1_rejects_reconstruction_settings_that_do_nothing():
    config = load_config("configs/smoke.yaml")
    config["phase1"]["decoder_enabled"] = True
    with pytest.raises(ValueError, match="coefficient_reconstruction_loss_weight"):
        validate_config(config)
    config["phase1"]["decoder_enabled"] = False
    config["phase1"]["coefficient_reconstruction_loss_weight"] = 0.3
    with pytest.raises(ValueError, match="decoder_enabled"):
        validate_config(config)


def _adversarial_config(start_after=0):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(
        adversarial_weight=0.5,
        adversarial_start_after_updates=start_after,
        adversarial_ramp_updates=1,
        discriminator_channels=[8, 16],
        discriminator_learning_rate=0.0002,
        discriminator_initialization_seed=7,
    )
    validate_config(config)
    return config


def _adversarial_trainer(config):
    seed_everything(5)
    phase1 = build_phase1_model(config, ImuNormalizer())
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    return system, Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")


def test_discriminator_trains_itself_and_stays_out_of_the_decoder_optimizer():
    system, trainer = _adversarial_trainer(_adversarial_config())
    decoder_ids = {id(p) for p in trainer.parameters}
    critic = list(trainer.discriminator.parameters())
    # Neu discriminator lot vao optimizer cua decoder thi no se duoc toi uu de
    # THUA chinh no, va so hang doi khang mat het y nghia.
    assert decoder_ids.isdisjoint({id(p) for p in critic})

    backbone_before = state_dict_hash(system.backbone)
    critic_before = [p.detach().clone() for p in critic]
    metrics = trainer.step([_batch(2)])

    assert not metrics["skipped"]
    assert metrics["adversarial_weight"] > 0
    assert "adversarial_generator" in metrics and "adversarial_critic" in metrics
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, critic))
    assert state_dict_hash(system.backbone) == backbone_before
    trainer.assert_backbone_frozen()


def test_adversarial_term_is_silent_before_its_start_update():
    _, trainer = _adversarial_trainer(_adversarial_config(start_after=1000))
    metrics = trainer.step([_batch(2)])
    assert metrics["adversarial_weight"] == 0.0
    assert "adversarial_generator" not in metrics
    assert trainer.discriminator is not None      # da dung, chi chua dung toi


def test_no_discriminator_at_all_when_the_weight_is_zero():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"]["adversarial_weight"] = 0.0
    validate_config(config)
    _, trainer = _adversarial_trainer(config)
    assert trainer.discriminator is None
    metrics = trainer.step([_batch(2)])
    assert not metrics["skipped"]
    assert metrics["adversarial_weight"] == 0.0
    assert "discriminator" not in trainer.checkpoint_payload(config)
