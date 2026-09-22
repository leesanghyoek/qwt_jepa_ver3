"""Verify the gyro-to-camera mapping that IMU-driven motion blur rests on.

``qjepa/corruptions/motion.py`` assumes the ``lcam_front`` frame coincides with
the IMU body frame, so that body gyro axes map to camera axes by the identity.
That assumption decides which way every blurred frame is smeared. It is cheap to
check and expensive to get wrong, so it is checked here rather than asserted.

Method: the ground-truth poses give the camera orientation at each frame, so the
relative rotation between consecutive frames, expressed in the camera frame, is
the rotation the camera actually underwent. Integrating the body gyro over the
same interval gives the rotation the IMU reports. If the two frames coincide the
correlation is near 1 on each matching axis and near 0 off-diagonal.

A slope near 1 confirms there is no scale factor. The gyro is integrated on a
dense grid resampled with ``np.interp``, exactly as ``motion.py`` integrates an
exposure. Selecting raw samples by a boolean mask instead loses a sample whenever
the float32 camera timestamp rounds just past an IMU tick, which costs a tenth of
each interval and shows up as a spurious slope near 1.07.

No model, no checkpoint, no manifest: only the raw trajectories.

    python3 tools/imu_blur_axis_check.py --data-root <root> --trajectories 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - scipy is a hard dependency of the repo
    print("scipy is required for this check", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.corruptions.motion import trapezoid

POSE_FILE = "pose_lcam_front.txt"
AXIS_NAMES = ("roll / optical axis", "pitch / right axis", "yaw / down axis")


def _load(directory: Path, name: str) -> np.ndarray:
    npy, txt = directory / f"{name}.npy", directory / f"{name}.txt"
    if npy.is_file():
        return np.load(npy)
    if txt.is_file():
        return np.loadtxt(txt, dtype=np.float64)
    raise FileNotFoundError(f"Missing {name}.npy/.txt in {directory}")


def trajectory_paths(root: Path, limit: int) -> list[Path]:
    found: list[Path] = []
    for pose in sorted(root.glob(f"*/*/*/{POSE_FILE}")):
        if (pose.parent / "imu").is_dir():
            found.append(pose.parent)
        if len(found) >= limit:
            break
    return found


def compare(path: Path, samples: int = 41) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (correlation matrix, per-axis slope, peak |gyro|) for one trajectory."""
    pose = np.loadtxt(path / POSE_FILE, dtype=np.float64)
    imu = path / "imu"
    gyro = np.asarray(_load(imu, "gyro"), dtype=np.float64)
    imu_time = np.asarray(_load(imu, "imu_time"), dtype=np.float64).reshape(-1)
    cam_time = np.asarray(_load(imu, "cam_time"), dtype=np.float64).reshape(-1)
    if pose.shape[1] != 7:
        raise ValueError(f"{path}: expected pose rows of x,y,z,qx,qy,qz,qw")

    world_from_camera = Rotation.from_quat(pose[:, 3:7])
    count = min(len(pose), len(cam_time)) - 1
    measured = np.empty((count, 3))
    reported = np.empty((count, 3))
    for index in range(count):
        relative = world_from_camera[index].inv() * world_from_camera[index + 1]
        measured[index] = relative.as_rotvec()
        grid = np.linspace(cam_time[index], cam_time[index + 1], samples)
        resampled = np.stack(
            [np.interp(grid, imu_time, gyro[:, axis]) for axis in range(3)], axis=1
        )
        reported[index] = trapezoid(resampled, float(grid[1] - grid[0]))

    correlation = np.empty((3, 3))
    for camera_axis in range(3):
        for body_axis in range(3):
            correlation[camera_axis, body_axis] = np.corrcoef(
                measured[:, camera_axis], reported[:, body_axis]
            )[0, 1]
    slope = np.array(
        [np.polyfit(reported[:, axis], measured[:, axis], 1)[0] for axis in range(3)]
    )
    return correlation, slope, float(np.abs(gyro).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--trajectories", type=int, default=12)
    parser.add_argument("--minimum-correlation", type=float, default=0.95)
    parser.add_argument(
        "--samples", type=int, default=41,
        help="resampling density per camera interval before integrating",
    )
    args = parser.parse_args()

    paths = trajectory_paths(args.data_root, args.trajectories)
    if not paths:
        print(f"No trajectory with {POSE_FILE} under {args.data_root}", file=sys.stderr)
        return 2

    diagonals: list[np.ndarray] = []
    off_diagonals: list[float] = []
    slopes: list[np.ndarray] = []
    print(f"{'trajectory':<46} {'corr (diag)':>22} {'slope':>22} {'max|w|':>7}")
    for path in paths:
        correlation, slope, peak = compare(path, args.samples)
        diagonal = np.diag(correlation)
        mask = ~np.eye(3, dtype=bool)
        diagonals.append(diagonal)
        off_diagonals.append(float(np.abs(correlation[mask]).max()))
        slopes.append(slope)
        name = "/".join(path.parts[-3:])
        print(
            f"{name:<46} {np.array2string(diagonal, precision=3):>22} "
            f"{np.array2string(slope, precision=3):>22} {peak:>7.2f}"
        )

    diagonal = np.concatenate(diagonals)
    worst = float(diagonal.min())
    strongest_off = float(max(off_diagonals))
    mean_slope = float(np.mean(slopes))
    print()
    for axis, name in enumerate(AXIS_NAMES):
        column = np.array([value[axis] for value in diagonals])
        print(f"  gyro[{axis}] {name:<22} min corr {column.min():.4f}")
    print(f"\n  weakest matching-axis correlation : {worst:.4f}")
    print(f"  strongest off-axis correlation    : {strongest_off:.4f}")
    print(f"  mean slope (expect ~1.0)          : {mean_slope:.3f}")

    ok = worst >= args.minimum_correlation and strongest_off < worst
    print(f"\n  {'PASS' if ok else 'FAIL'} - identity gyro-to-camera mapping in motion.py "
          f"{'holds' if ok else 'is NOT supported by these trajectories'}")
    if not ok:
        print("  Do not train with motion_from_imu until this passes: the blur would "
              "point the wrong way relative to the IMU the model reads.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
