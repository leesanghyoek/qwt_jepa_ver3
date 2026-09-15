import torch

from qjepa.cli import _system_from_phase2
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.training.checkpoints import atomic_torch_save, load_checkpoint, require_phase1_checkpoint
from qjepa.training.phase1 import Phase1Trainer
from qjepa.training.phase2 import Phase2Trainer


def _batch(batch_size):
    times = torch.arange(32).float().mul(0.01).repeat(batch_size, 1)
    clean_image = torch.rand(batch_size, 3, 32, 32)
    clean_imu = torch.randn(batch_size, 6, 32)
    return {
        "image_clean": clean_image,
        "image_noisy": (0.4 * clean_image).clamp(0, 1),
        "imu_clean_phys": clean_imu,
        "imu_noisy_phys": clean_imu + 0.02 * torch.randn_like(clean_imu),
        "image_time": times.mean(1),
        "imu_times": times,
        "sample_id": [f"checkpoint-{index}" for index in range(batch_size)],
    }


def test_phase_checkpoints_roundtrip_strictly(tmp_path):
    config = load_config("configs/smoke.yaml")
    device = torch.device("cpu")
    seed_everything(12)
    phase1_model = build_phase1_model(config, ImuNormalizer())
    phase1_trainer = Phase1Trainer(phase1_model, config, device, "synthetic")
    assert not phase1_trainer.step(_batch(2))["skipped"]
    phase1_path = tmp_path / "phase1.pt"
    atomic_torch_save(phase1_trainer.checkpoint_payload(config), phase1_path)
    phase1_payload = load_checkpoint(phase1_path)
    require_phase1_checkpoint(phase1_payload)

    seed_everything(13)
    system = RestorationSystem(phase1_model.backbone, phase1_model.normalizer, build_decoders(config))
    phase2_trainer = Phase2Trainer(system, config, device, str(phase1_path.resolve()), "synthetic")
    assert not phase2_trainer.step([_batch(1)])["skipped"]
    phase2_path = tmp_path / "phase2.pt"
    atomic_torch_save(phase2_trainer.checkpoint_payload(config), phase2_path)

    loaded, loaded_config = _system_from_phase2(str(phase2_path), device)
    batch = _batch(1)
    output = loaded(
        batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"]
    )
    assert loaded_config["pipeline_version"] == 3
    assert output.image.shape == (1, 3, 32, 32)
    assert output.imu_physical.shape == (1, 6, 32)

