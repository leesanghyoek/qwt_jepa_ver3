"""The blur experiment must be repeatable and score only actual blur."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from qjepa.cli import _phase2_full_blur_guard, _validate_active_blur
from qjepa.config import load_config
from qjepa.data.dataset import PairedCameraImuDataset
from qjepa.training.checkpoints import configuration_hash


def test_blur_recipe_reuses_phase1_but_changes_phase2_contract():
    original = load_config("configs/kaggle_tartanair_v2.yaml")
    mixed = load_config("configs/kaggle_phase2_blur_mix.yaml")
    assert configuration_hash(original, "phase1") == configuration_hash(mixed, "phase1")
    assert configuration_hash(original, "phase2") != configuration_hash(mixed, "phase2")
    guarded = load_config("configs/kaggle_phase2_guarded_finetune.yaml")
    assert configuration_hash(original, "phase1") == configuration_hash(guarded, "phase1")


def test_full_blur_guard_rejects_quality_tradeoff():
    reference = {
        "full": {"image_psnr_db": 20, "image_ssim": 0.7, "accel_rmse": 0.8, "gyro_rmse": 0.08},
        "blur": {"image_mae_restored": 0.05, "strong_edge_gradient_mae_restored": 0.2},
    }
    limits = load_config("configs/kaggle_phase2_guarded_finetune.yaml")["phase2"]["full_guard"]
    good_full = {"image_psnr_db": 19.9, "image_ssim": 0.695, "accel_rmse": 0.81, "gyro_rmse": 0.081}
    good_blur = {"image_mae_restored": 0.03, "strong_edge_gradient_mae_restored": 0.18}
    passed, reasons = _phase2_full_blur_guard(good_full, good_blur, reference, limits)
    assert passed and not reasons
    bad_full = dict(good_full, image_psnr_db=17.3, accel_rmse=0.96)
    passed, reasons = _phase2_full_blur_guard(bad_full, good_blur, reference, limits)
    assert not passed and {"image_psnr_db", "accel_rmse"}.issubset(reasons)


def test_scenario_mix_is_deterministic_per_sample_and_realization():
    scenarios = load_config("configs/kaggle_phase2_blur_mix.yaml")["phase2"]["train_scenarios"]
    samples = [SimpleNamespace(sample_id=f"sample-{index}") for index in range(1000)]
    dataset = PairedCameraImuDataset(samples, scenarios=scenarios, scenario_seed=73128)
    modes = [dataset._scenario_modes(sample) for sample in samples]
    assert modes == [dataset._scenario_modes(sample) for sample in reversed(samples)][::-1]
    blur_count = sum(image_mode == "blur_only" for image_mode, _ in modes)
    assert 340 < blur_count < 460
    dataset.set_realization(1)
    assert any(dataset._scenario_modes(sample) != old for sample, old in zip(samples, modes))


def test_blur_validation_ignores_frames_without_optical_blur():
    clean = torch.zeros(2, 3, 32, 32)
    clean[..., 16:] = 1
    noisy = F.avg_pool2d(clean, 5, stride=1, padding=2)
    noisy[1] = 0
    raw = {
        "image_clean": clean,
        "image_noisy": noisy,
        "imu_noisy_phys": torch.zeros(2, 6, 8),
        "image_time": torch.zeros(2, dtype=torch.float64),
        "imu_times": torch.zeros(2, 8, dtype=torch.float64),
        "corruption": [
            {"image": {"defocus": True, "motion": False, "downsample": False}},
            {"image": {"defocus": False, "motion": False, "downsample": False}},
        ],
    }
    metrics = _validate_active_blur(
        torch.nn.Identity(), [raw], torch.device("cpu"),
        forward_model=lambda *_: {"image": clean},
    )
    assert metrics["active_frames"] == 1
    assert metrics["image_mae_input"] > 0
    assert metrics["strong_edge_gradient_mae_input"] > 0
    assert metrics["image_mae_restored"] == 0
    assert metrics["strong_edge_gradient_mae_restored"] == 0
