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



def test_resume_does_not_load_the_batches_it_skips():
    """Bo qua phai xay ra o muc SAMPLER, truoc khi du lieu duoc nap.

    Truoc day vong lap bo qua o muc DataLoader: moi batch bi bo van duoc giai nen
    PNG va chay het corruption roi moi bi vut di. Resume o update 2500 vi the phai
    nap 20.000 anh truoc khi in duoc dong log dau tien — nhin nhu treo.
    """

    class _Counting(_Dataset):
        def __init__(self, length=10):
            super().__init__(length)
            self.loaded = []

        def __getitem__(self, index):
            self.loaded.append(index)
            return super().__getitem__(index)

    config = load_config("configs/smoke.yaml")
    dataset = _Counting()
    stream = _training_batch_stream(config, dataset, 2, start_microbatch=3, namespace="phase1")
    first = next(stream)

    # Bo 3 batch x 2 mau = 6 mau. Chi batch duoc yield moi duoc nap.
    assert len(dataset.loaded) == 2, f"da nap {len(dataset.loaded)} mau thay vi 2"
    assert sorted(dataset.loaded) == sorted(int(s.split(":")[1]) for s in first["sample_id"])
