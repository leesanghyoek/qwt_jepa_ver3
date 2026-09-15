import copy

import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything
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

