"""Paired clean/corrupted RGB and IMU dataset."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..corruptions import LowLightImageCorruptor, TrajectoryImuCorruptor
from ..corruptions.rng import derive_seed
from .manifest import PairedSample
from .tartanair import Trajectory


def load_rgb(path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        target_height, target_width = size
        if image.size != (target_width, target_height):
            scale = max(target_width / image.width, target_height / image.height)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
            )
            left = (image.width - target_width) // 2
            top = (image.height - target_height) // 2
            image = image.crop((left, top, left + target_width, top + target_height))
        return np.asarray(image, dtype=np.float32) / 255.0


class _TrajectoryCache:
    def __init__(self, size: int = 4):
        self.size = size
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, sample: PairedSample) -> tuple[np.ndarray, np.ndarray]:
        key = sample.trajectory_key
        if key not in self.values:
            trajectory = Trajectory(
                path=Path(sample.trajectory_path),
                environment=sample.environment,
                difficulty=sample.difficulty,
                trajectory_id=sample.trajectory_id,
            )
            self.values[key] = trajectory.load_imu()
            if len(self.values) > self.size:
                self.values.popitem(last=False)
        else:
            self.values.move_to_end(key)
        return self.values[key]


class PairedCameraImuDataset(Dataset):
    def __init__(
        self,
        samples: list[PairedSample],
        *,
        image_corruptor: LowLightImageCorruptor | None = None,
        imu_corruptor: TrajectoryImuCorruptor | None = None,
        image_size: tuple[int, int] = (256, 256),
        realization: int = 0,
        image_mode: str = "full",
        imu_mode: str = "full",
        scenarios: list[dict[str, object]] | None = None,
        scenario_seed: int = 0,
        cache_size: int = 4,
    ) -> None:
        if not samples:
            raise ValueError("Dataset cannot be empty")
        self.samples = samples
        self.image_corruptor = image_corruptor or LowLightImageCorruptor()
        self.imu_corruptor = imu_corruptor or TrajectoryImuCorruptor(cache_size=cache_size)
        self.image_size = image_size
        self.realization = realization
        self.image_mode = image_mode
        self.imu_mode = imu_mode
        self.scenarios = scenarios
        self.scenario_seed = scenario_seed
        self.cache = _TrajectoryCache(cache_size)

    def set_realization(self, realization: int) -> None:
        self.realization = int(realization)

    def __len__(self) -> int:
        return len(self.samples)

    def _scenario_modes(self, sample: PairedSample) -> tuple[str, str]:
        image_mode, imu_mode = self.image_mode, self.imu_mode
        if self.scenarios:
            rng = np.random.default_rng(derive_seed(
                self.scenario_seed, "phase2_scenario", sample.sample_id, self.realization,
            ))
            weights = np.asarray([float(item["weight"]) for item in self.scenarios], dtype=np.float64)
            choice = self.scenarios[int(rng.choice(len(weights), p=weights / weights.sum()))]
            image_mode, imu_mode = str(choice["image_mode"]), str(choice["imu_mode"])
        return image_mode, imu_mode

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_mode, imu_mode = self._scenario_modes(sample)
        imu_all, imu_times_all = self.cache.get(sample)
        clean_image = load_rgb(sample.image_path, self.image_size)
        clean_imu = np.asarray(imu_all[sample.imu_start : sample.imu_end], dtype=np.float32)
        imu_times = np.asarray(imu_times_all[sample.imu_start : sample.imu_end], dtype=np.float64)
        noisy_image, image_parameters = self.image_corruptor(
            clean_image,
            split=sample.split,
            realization=self.realization,
            trajectory=sample.trajectory_key,
            timestamp=sample.image_time,
            frame_index=sample.image_index,
            mode=image_mode,
        )
        noisy_imu, imu_parameters = self.imu_corruptor.window(
            imu_all,
            imu_times_all,
            sample.imu_start,
            sample.imu_end,
            split=sample.split,
            realization=self.realization,
            trajectory=sample.trajectory_key,
            mode=imu_mode,
        )
        chw = lambda value: torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float()
        return {
            "image_clean": chw(clean_image),
            "image_noisy": chw(noisy_image),
            "imu_clean_phys": torch.from_numpy(clean_imu).float(),
            "imu_noisy_phys": torch.from_numpy(noisy_imu).float(),
            "image_time": torch.tensor(sample.image_time, dtype=torch.float64),
            "imu_times": torch.from_numpy(imu_times),
            "imu_start": torch.tensor(sample.imu_start),
            "sample_id": sample.sample_id,
            "trajectory_key": sample.trajectory_key,
            "corruption": {"image": image_parameters, "imu": imu_parameters},
        }


def collate_paired(batch: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in ("image_clean", "image_noisy", "image_time", "imu_times", "imu_start"):
        output[key] = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
    for key in ("imu_clean_phys", "imu_noisy_phys"):
        stacked = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
        output[key] = stacked.transpose(1, 2).contiguous()
    # Keep absolute timestamps in float64; float32 loses 100 Hz spacing at Unix epochs.
    output["sample_id"] = [item["sample_id"] for item in batch]
    output["trajectory_key"] = [item["trajectory_key"] for item in batch]
    output["corruption"] = [item["corruption"] for item in batch]
    return output
