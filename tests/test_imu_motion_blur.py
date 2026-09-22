"""Motion blur must follow the IMU, not an independent random draw.

The point of these tests is the coupling itself: if the blur can be predicted
from the gyro window the model reads, the IMU branch has something to contribute
to the image branch. If it cannot, the IMU is decoration. Each test below pins
one link in that chain -- direction, magnitude, determinism, and the refusal to
silently fall back to a random draw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor
from qjepa.corruptions.motion import exposure_path, imu_blur_kernel, path_to_kernel

RATE = 100.0
FOCAL = 128.0


def constant_gyro(roll: float, pitch: float, yaw: float, samples: int = 128):
    times = np.arange(samples, dtype=np.float64) / RATE
    gyro = np.tile(np.array([roll, pitch, yaw], dtype=np.float64), (samples, 1))
    return gyro, times


def centre_time(times: np.ndarray) -> float:
    return float(0.5 * (times[0] + times[-1]))


def test_still_camera_produces_no_blur() -> None:
    gyro, times = constant_gyro(0.0, 0.0, 0.0)
    kernel, report = imu_blur_kernel(
        gyro, times, centre_time(times), 0.02, focal_length_px=FOCAL
    )
    assert kernel.shape == (1, 1)
    assert report["path_span_px"] == pytest.approx(0.0, abs=1e-9)


def test_kernel_preserves_brightness() -> None:
    gyro, times = constant_gyro(0.0, 0.5, 1.2)
    kernel, _ = imu_blur_kernel(gyro, times, centre_time(times), 0.03, focal_length_px=FOCAL)
    assert kernel.sum() == pytest.approx(1.0, abs=1e-12)
    assert (kernel >= 0).all()


def test_yaw_smears_horizontally_and_pitch_vertically() -> None:
    """The axis mapping measured in tools/imu_blur_axis_check.py, pinned here."""
    times = constant_gyro(0, 0, 0)[1]
    yaw_kernel, _ = imu_blur_kernel(
        *constant_gyro(0.0, 0.0, 2.0), centre_time(times), 0.03, focal_length_px=FOCAL
    )
    pitch_kernel, _ = imu_blur_kernel(
        *constant_gyro(0.0, 2.0, 0.0), centre_time(times), 0.03, focal_length_px=FOCAL
    )
    # Mass spread along columns for yaw, along rows for pitch.
    assert yaw_kernel.sum(axis=0).astype(bool).sum() > yaw_kernel.sum(axis=1).astype(bool).sum()
    assert pitch_kernel.sum(axis=1).astype(bool).sum() > pitch_kernel.sum(axis=0).astype(bool).sum()


def test_roll_is_reported_but_never_folded_into_the_kernel() -> None:
    """Roll is a spatially varying rotation; one kernel cannot represent it."""
    times = constant_gyro(0, 0, 0)[1]
    kernel, report = imu_blur_kernel(
        *constant_gyro(3.0, 0.0, 0.0), centre_time(times), 0.03, focal_length_px=FOCAL
    )
    assert kernel.shape == (1, 1)
    assert abs(report["roll_radians"]) > 0.05


@pytest.mark.parametrize("exposure", [0.005, 0.01, 0.02, 0.04])
def test_blur_grows_with_exposure(exposure: float) -> None:
    gyro, times = constant_gyro(0.0, 0.0, 1.0)
    _, report = imu_blur_kernel(
        gyro, times, centre_time(times), exposure, focal_length_px=FOCAL
    )
    # Constant rotation: span = f * omega * exposure, exactly.
    assert report["path_span_px"] == pytest.approx(FOCAL * 1.0 * exposure, rel=1e-6)


def test_blur_grows_with_rotation_rate() -> None:
    times = constant_gyro(0, 0, 0)[1]
    spans = [
        imu_blur_kernel(
            *constant_gyro(0.0, 0.0, rate), centre_time(times), 0.02, focal_length_px=FOCAL
        )[1]["path_span_px"]
        for rate in (0.25, 0.5, 1.0, 2.0)
    ]
    assert spans == sorted(spans)
    assert spans[-1] > 4 * spans[0] - 1e-6


def test_long_paths_are_clipped_and_the_clipping_is_reported() -> None:
    gyro, times = constant_gyro(0.0, 0.0, 20.0)
    kernel, report = imu_blur_kernel(
        gyro, times, centre_time(times), 0.05, focal_length_px=FOCAL, max_radius_px=8.0
    )
    assert report["path_clip_scale"] < 1.0
    assert (kernel.shape[0] - 1) // 2 <= 8


def test_path_is_centred_on_the_capture_instant() -> None:
    gyro, times = constant_gyro(0.0, 0.0, 1.0)
    u, v, _ = exposure_path(
        gyro, times, centre_time(times), 0.02, focal_length_px=FOCAL, samples=25
    )
    assert u[len(u) // 2] == pytest.approx(0.0, abs=1e-12)
    assert v[len(v) // 2] == pytest.approx(0.0, abs=1e-12)
    assert u[0] == pytest.approx(-u[-1], rel=1e-9)


def test_varying_gyro_bends_the_path() -> None:
    """A path that only ever comes from a straight line would ignore the trace shape."""
    times = np.arange(128, dtype=np.float64) / RATE
    gyro = np.zeros((128, 3))
    gyro[:, 2] = np.sin(2 * math.pi * 6.0 * times)  # yaw oscillating within the exposure
    gyro[:, 1] = 1.0
    u, v, _ = exposure_path(gyro, times, centre_time(times), 0.04, focal_length_px=FOCAL)
    straight = np.polyfit(v, u, 1)
    residual = np.abs(u - np.polyval(straight, v)).max()
    assert residual > 1e-3, "path collapsed onto a straight line"


def test_corruptor_refuses_to_run_without_the_gyro() -> None:
    """Falling back to a random kernel here would silently remove the coupling."""
    corruptor = LowLightImageCorruptor(LowLightImageCorruptionConfig(motion_from_imu=True))
    image = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="motion_from_imu"):
        corruptor(image, split="train", realization=0, trajectory="t", timestamp=0.4,
                  frame_index=4, mode="blur_only")


def test_corruption_is_deterministic_for_the_same_sample() -> None:
    corruptor = LowLightImageCorruptor(LowLightImageCorruptionConfig())
    rng = np.random.default_rng(3)
    image = rng.random((32, 32, 3)).astype(np.float32)
    gyro, times = constant_gyro(0.1, 0.4, 0.9)
    kwargs = dict(split="train", realization=0, trajectory="env/Data_easy/P000",
                  timestamp=centre_time(times), frame_index=7, mode="full",
                  gyro=gyro, imu_times=times)
    first, first_params = corruptor(image, **kwargs)
    second, second_params = corruptor(image, **kwargs)
    assert np.array_equal(first, second)
    assert first_params["path_span_px"] == second_params["path_span_px"]


def test_blurred_image_tracks_rotation_rate() -> None:
    """End to end: the same frame, two rotation rates, measurably different sharpness."""
    corruptor = LowLightImageCorruptor(
        LowLightImageCorruptionConfig(defocus_probability=0.0, downsample_probability=0.0)
    )
    rng = np.random.default_rng(11)
    image = rng.random((64, 64, 3)).astype(np.float32)
    times = constant_gyro(0, 0, 0)[1]

    def detail(rate: float) -> float:
        gyro, _ = constant_gyro(0.0, 0.0, rate)
        out, _ = corruptor(image, split="train", realization=0, trajectory="t",
                           timestamp=centre_time(times), frame_index=0, mode="blur_only",
                           gyro=gyro, imu_times=times)
        return float(np.abs(np.diff(out.mean(axis=-1), axis=1)).mean())

    assert detail(3.0) < detail(0.5), "faster rotation did not produce a softer frame"


def test_exposure_lengthens_as_the_frame_darkens() -> None:
    """Dark frames come from a longer shutter, so low light and blur arrive together."""
    config = LowLightImageCorruptionConfig(exposure_tracks_darkness=True)
    corruptor = LowLightImageCorruptor(config)
    gains, exposures = [], []
    for frame in range(60):
        params = corruptor._parameters("train", 0, "env/Data_easy/P000", frame * 0.1, "full")
        gains.append(params["exposure_gain"])
        exposures.append(params["exposure_seconds"])
    assert np.corrcoef(gains, exposures)[0, 1] < -0.99


def test_path_to_kernel_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError):
        path_to_kernel(np.array([0.0, np.nan]), np.array([0.0, 1.0]))
