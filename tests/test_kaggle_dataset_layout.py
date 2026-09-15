"""Regression tests for the real Kaggle TartanAir V2 layout (see kaggle_dataset.md).

The dataset ships 166 trajectories as <env>/<Data_easy|Data_hard>/<Pxxx>/ with
.npy IMU at 100 Hz, a 10 Hz camera and 256x256 RGB frames. These tests pin the
geometry that the audit measured, so a loader change cannot silently stop
matching the dataset. Images are small here because framing does not depend on
pixel size; the timing is the part under test.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qjepa.data.manifest import assign_splits, build_manifest
from qjepa.data.tartanair import discover_trajectories

IMU_HZ, CAMERA_HZ, WINDOW = 100.0, 10.0, 128
# A 128-sample window at 100 Hz spans 1.27 s and must bracket the frame, so the
# first and last ~6.35 frames of every trajectory can never be centred.
EXPECTED_REJECTED_CENTRE = 13


def _trajectory(root: Path, environment: str, difficulty: str, trajectory_id: str, frames: int) -> None:
    path = root / environment / difficulty / trajectory_id
    (path / "image_lcam_front").mkdir(parents=True)
    (path / "imu").mkdir()
    rows = int(frames * IMU_HZ / CAMERA_HZ)
    imu_time = np.arange(rows, dtype=np.float64) / IMU_HZ
    cam_time = np.arange(frames, dtype=np.float64) / CAMERA_HZ
    accel = np.stack([np.sin(imu_time * (axis + 1)) for axis in range(3)], axis=-1)
    accel[:, 2] -= 9.81  # TartanAir acc.npy keeps gravity; acc_nograv.npy does not.
    gyro = np.stack([np.cos(imu_time * (axis + 1)) for axis in range(3)], axis=-1) * 0.3
    np.save(path / "imu" / "acc.npy", accel)
    np.save(path / "imu" / "gyro.npy", gyro)
    np.save(path / "imu" / "imu_time.npy", imu_time)
    np.save(path / "imu" / "cam_time.npy", cam_time)
    # Files TartanAir V2 also ships; the loader must ignore them.
    np.save(path / "imu" / "acc_nograv.npy", accel + np.array([0.0, 0.0, 9.81]))
    np.save(path / "imu" / "vel_body.npy", gyro)
    for index in range(frames):
        Image.new("RGB", (32, 32), (index % 256, 30, 40)).save(
            path / "image_lcam_front" / f"{index:06d}_lcam_front.png"
        )


def _dataset(root: Path, frames: int = 40) -> Path:
    for environment in ("AmericanDiner", "Office"):
        for difficulty in ("Data_easy", "Data_hard"):
            for index in range(2):
                _trajectory(root, environment, difficulty, f"P{index:03d}", frames)
    return root


def test_discovery_parses_environment_difficulty_and_ignores_extra_imu_files(tmp_path):
    trajectories = discover_trajectories(_dataset(tmp_path / "tartanair-v2"))
    assert len(trajectories) == 8
    assert {item.difficulty for item in trajectories} == {"Data_easy", "Data_hard"}
    assert {item.environment for item in trajectories} == {"AmericanDiner", "Office"}
    # Data_easy/Data_hard of one motion share a motion_key, so they cannot split apart.
    assert len({item.motion_key for item in trajectories}) == 4
    assert all(item.split_hint is None for item in trajectories)
    imu, times = trajectories[0].load_imu()
    assert imu.shape == (400, 6) and times.shape == (400,)
    assert imu[:, 2].mean() < -9.0, "acc.npy must keep gravity"


def test_window_framing_matches_the_audited_100hz_10hz_geometry(tmp_path):
    manifest = build_manifest(_dataset(tmp_path / "tartanair-v2"), window=WINDOW)
    audit = manifest["meta"]["pairing_audit"]
    assert len(audit) == 8
    for entry in audit:
        assert entry["rejected_centre"] == EXPECTED_REJECTED_CENTRE
        assert entry["rejected_nonuniform"] == 0 and entry["rejected_outside"] == 0
        assert entry["accepted"] == 40 - EXPECTED_REJECTED_CENTRE
    # Losing a fixed 13 frames per trajectory is a shrinking cost on longer ones.
    longer = build_manifest(_dataset(tmp_path / "longer", frames=80), window=WINDOW)
    assert all(entry["rejected_centre"] == EXPECTED_REJECTED_CENTRE
               for entry in longer["meta"]["pairing_audit"])


def test_easy_and_hard_of_one_motion_stay_in_the_same_split(tmp_path):
    meta = build_manifest(_dataset(tmp_path / "tartanair-v2"), window=WINDOW)["meta"]
    placement: dict[str, set[str]] = {}
    for split, keys in meta["trajectories_per_split"].items():
        for key in keys:
            environment, _, trajectory_id = key.split("/")
            placement.setdefault(f"{environment}/{trajectory_id}", set()).add(split)
    assert placement, "no trajectories were assigned"
    assert all(len(splits) == 1 for splits in placement.values())
    assert meta["normalization"]["mean"][2] < -9.0
    assert sum(meta["samples_per_split"].values()) == 8 * (40 - EXPECTED_REJECTED_CENTRE)


def test_root_one_level_too_high_names_the_offending_directories(tmp_path):
    """The Kaggle mount holds tartanair-v2 beside tartanair-v2-jepa/train/."""
    _dataset(tmp_path / "tartanair-v2")
    _trajectory(tmp_path / "tartanair-v2-jepa" / "train", "Office", "Data_easy", "P000", 40)
    with pytest.raises(ValueError) as error:
        assign_splits(discover_trajectories(tmp_path))
    message = str(error.value)
    assert "Mixed explicit" in message and "tartanair-v2-jepa" in message
    assert "points one level too high" in message
    # Pointing at the right root works.
    assert len(discover_trajectories(tmp_path / "tartanair-v2")) == 8
