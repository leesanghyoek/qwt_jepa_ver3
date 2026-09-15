from types import SimpleNamespace

import torch

from qjepa.cli import _evaluate_with_overlap
from qjepa.data import ImuNormalizer


class _IdentitySystem:
    def __init__(self):
        self.normalizer = ImuNormalizer()

    def eval(self):
        return self

    def __call__(self, image, imu, image_time, imu_times):
        return SimpleNamespace(image=image, imu_physical=imu)


def test_final_evaluation_merges_imu_windows_before_counting_rows():
    dataset = SimpleNamespace(
        samples=[
            SimpleNamespace(trajectory_key="T", imu_end=4),
            SimpleNamespace(trajectory_key="T", imu_end=6),
        ]
    )
    image = torch.rand(2, 3, 8, 8)
    imu = torch.randn(2, 6, 4)
    batch = {
        "image_clean": image,
        "image_noisy": image,
        "imu_clean_phys": imu,
        "imu_noisy_phys": imu,
        "image_time": torch.tensor([0.015, 0.035]),
        "imu_times": torch.tensor(
            [[0.00, 0.01, 0.02, 0.03], [0.02, 0.03, 0.04, 0.05]]
        ),
        "imu_start": torch.tensor([0, 2]),
        "trajectory_key": ["T", "T"],
    }
    metrics = _evaluate_with_overlap(
        _IdentitySystem(), [batch], dataset, torch.device("cpu"), maximum_batches=1
    )
    assert metrics["imu_covered_unique_rows"] == 6
    assert metrics["accel_rmse"] == 0
    assert metrics["gyro_rmse"] == 0
    assert metrics["image_mae"] == 0

