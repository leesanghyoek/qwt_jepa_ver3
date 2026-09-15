import torch
from types import SimpleNamespace

from qjepa.cli import _training_batch_stream
from qjepa.config import load_config


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, length=10):
        self.length = length
        self.realization = 0
        self.samples = [SimpleNamespace(trajectory_key="dummy") for _ in range(length)]

    def set_realization(self, realization):
        self.realization = realization

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "image_clean": torch.zeros(3, 2, 2),
            "image_noisy": torch.zeros(3, 2, 2),
            "imu_clean_phys": torch.zeros(2, 6),
            "imu_noisy_phys": torch.zeros(2, 6),
            "image_time": torch.tensor(0.005, dtype=torch.float64),
            "imu_times": torch.tensor([0.0, 0.01], dtype=torch.float64),
            "imu_start": torch.tensor(0),
            "sample_id": f"{self.realization}:{index}",
            "trajectory_key": "dummy",
            "corruption": {},
        }


def test_sampler_resume_reconstructs_epoch_and_batch_offset():
    config = load_config("configs/smoke.yaml")
    dataset = _Dataset()
    full = _training_batch_stream(
        config, dataset, 2, start_microbatch=0, namespace="phase1"
    )
    full_ids = [next(full)["sample_id"] for _ in range(7)]

    resumed_dataset = _Dataset()
    resumed = _training_batch_stream(
        config, resumed_dataset, 2, start_microbatch=3, namespace="phase1"
    )
    resumed_ids = [next(resumed)["sample_id"] for _ in range(4)]
    assert resumed_ids == full_ids[3:7]
