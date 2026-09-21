"""The blur experiment must be repeatable and score only actual blur."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from qjepa.cli import _validate_active_blur
from qjepa.config import load_config
from qjepa.data.dataset import PairedCameraImuDataset
from qjepa.training.checkpoints import configuration_hash


def test_blur_recipe_reuses_phase1_but_changes_phase2_contract():
    original = load_config("configs/kaggle_tartanair_v2.yaml")
    mixed = load_config("configs/kaggle_phase2_blur_mix.yaml")
    assert configuration_hash(original, "phase1") == configuration_hash(mixed, "phase1")
    assert configuration_hash(original, "phase2") != configuration_hash(mixed, "phase2")


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
