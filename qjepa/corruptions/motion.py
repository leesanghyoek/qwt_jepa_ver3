"""Motion blur derived from the same IMU trace the model is given.

Blur in a real capture is not an independent random draw: it is the camera's own
motion integrated over the exposure. Drawing it independently -- as the
``motion_length_px``/``motion_angle`` pair did -- severs the only physical link
between the two branches of this model, because the IMU window then carries no
information whatsoever about how the image was degraded. With the blur generated
here, the IMU is the one signal that can tell the decoder which way the image was
smeared, and by how much.

Geometry, measured rather than assumed (``tools/imu_blur_axis_check.py``): in
TartanAir V2 the ``lcam_front`` frame coincides with the IMU body frame. The
camera-frame relative rotation between consecutive poses matches the body gyro on
all three axes with correlation >= 0.996 across twelve trajectories, easy and
hard, with a slope of 1.0 once the integration window covers the full interval.
The body convention is x forward, y right, z down, so in the usual vision
convention (x right, y down, z forward) the angular velocity is

    omega_vision = (gyro_y, gyro_z, gyro_x)

For a pure rotation the optical flow at the principal point is

    u = -f * omega_vision_y = -f * gyro_z      (yaw   -> horizontal shift)
    v = +f * omega_vision_x = +f * gyro_y      (pitch -> vertical shift)

``gyro_x`` is roll about the optical axis. Roll produces a spatially varying
in-plane rotation, not one displacement, so a single convolution kernel cannot
represent it. It is reported in the parameters as ``roll_radians`` and excluded
from the kernel; treating it as a shift would be wrong, and pretending it is
absent would be worse.

The blur is driven by the **clean** gyro on purpose. Physically the true motion
smears the image while the IMU measures that motion with error, so the model sees
a noisy observation of the very quantity that explains the blur. That gap is the
task, not a defect.
"""

from __future__ import annotations

import math

import numpy as np

# Camera frame equals IMU body frame for lcam_front; see the module docstring.
# Index into the [ax, ay, az, gx, gy, gz] row layout used everywhere else.
GYRO_ROLL_AXIS = 0   # about the optical axis (x forward)
GYRO_PITCH_AXIS = 1  # about the right axis   (y right) -> vertical image motion
GYRO_YAW_AXIS = 2    # about the down axis    (z down)  -> horizontal image motion


def trapezoid(values: np.ndarray, step: float, axis: int = 0) -> np.ndarray:
    """Trapezoidal integral over a uniform grid.

    Spelled out rather than calling ``np.trapz``, which NumPy 2.0 removed while
    ``numpy>=1.24`` is still what this package declares.
    """
    moved = np.moveaxis(values, axis, 0)
    total = (0.5 * (moved[1:] + moved[:-1]) * step).sum(axis=0)
    return total


def exposure_path(
    gyro: np.ndarray,
    times: np.ndarray,
    centre_time: float,
    exposure_seconds: float,
    *,
    focal_length_px: float,
    angular_gain: float = 1.0,
    samples: int = 25,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pixel displacement traced by the principal point during one exposure.

    Returns ``(u, v, roll_radians)`` where ``u``/``v`` are horizontal/vertical
    offsets in pixels relative to mid-exposure, and ``roll_radians`` is the total
    in-plane rotation over the exposure (reported, never folded into ``u``/``v``).

    ``samples`` is forced odd so that mid-exposure is an exact sample and the path
    can be centred without interpolating the integral back onto itself.
    """
    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError(f"Expected gyro [N,3], got {gyro.shape}")
    if times.shape != (gyro.shape[0],):
        raise ValueError("Gyro and timestamps must have matching length")
    if gyro.shape[0] < 2:
        raise ValueError("Need at least two gyro samples to integrate an exposure")
    if not math.isfinite(exposure_seconds) or exposure_seconds <= 0:
        raise ValueError("exposure_seconds must be positive and finite")
    if not np.all(np.diff(times) > 0):
        raise ValueError("IMU timestamps must be strictly increasing")

    count = int(samples) if int(samples) % 2 else int(samples) + 1
    count = max(count, 3)
    half = 0.5 * float(exposure_seconds)
    grid = np.linspace(centre_time - half, centre_time + half, count)

    # np.interp clamps outside the window. The 1.27 s IMU window always surrounds
    # the capture instant by a wide margin, so clamping is an unreachable guard
    # rather than a silent approximation.
    pitch = np.interp(grid, times, gyro[:, GYRO_PITCH_AXIS])
    yaw = np.interp(grid, times, gyro[:, GYRO_YAW_AXIS])
    roll = np.interp(grid, times, gyro[:, GYRO_ROLL_AXIS])

    step = grid[1] - grid[0]
    # Cumulative trapezoid, then re-zero at mid-exposure so the path is centred.
    def integrate(values: np.ndarray) -> np.ndarray:
        increments = 0.5 * (values[1:] + values[:-1]) * step
        angle = np.concatenate(([0.0], np.cumsum(increments)))
        return angle - angle[count // 2]


    pitch_angle = integrate(pitch)
    yaw_angle = integrate(yaw)
    scale = float(focal_length_px) * float(angular_gain)
    u = -scale * yaw_angle
    v = scale * pitch_angle
    roll_radians = float(trapezoid(roll, step) * float(angular_gain))
    return u, v, roll_radians


def path_to_kernel(
    u: np.ndarray, v: np.ndarray, *, max_radius_px: float = 16.0
) -> tuple[np.ndarray, float]:
    """Rasterise a displacement path into a normalised blur kernel.

    Every path sample carries equal weight, which is the constant-illumination
    assumption: the shutter is open for the whole exposure and the scene does not
    change brightness during it.

    Returns ``(kernel, applied_scale)``. ``applied_scale`` is below 1.0 only when
    the path had to be shortened to respect ``max_radius_px``; it is recorded so a
    clipped kernel never passes as a faithful one.
    """
    if u.shape != v.shape or u.ndim != 1:
        raise ValueError("Path components must be 1-D arrays of equal length")
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("Displacement path must be finite")

    extent = float(max(np.abs(u).max(), np.abs(v).max()))
    applied_scale = 1.0
    if extent > max_radius_px > 0:
        applied_scale = float(max_radius_px / extent)
        u, v, extent = u * applied_scale, v * applied_scale, float(max_radius_px)

    radius = int(math.ceil(extent - 1e-9))
    if radius <= 0:
        # A stationary camera produces a sharp frame. That is the correct answer,
        # not a degenerate case to paper over.
        return np.array([[1.0]], dtype=np.float64), applied_scale

    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    weight = 1.0 / len(u)
    for offset_x, offset_y in zip(u, v):
        row, column = radius + float(offset_y), radius + float(offset_x)
        row0, column0 = int(math.floor(row)), int(math.floor(column))
        row_frac, column_frac = row - row0, column - column0
        for rr, wr in ((row0, 1.0 - row_frac), (row0 + 1, row_frac)):
            for cc, wc in ((column0, 1.0 - column_frac), (column0 + 1, column_frac)):
                if 0 <= rr < size and 0 <= cc < size:
                    kernel[rr, cc] += weight * wr * wc
    total = kernel.sum()
    if total <= 0:
        return np.array([[1.0]], dtype=np.float64), applied_scale
    return kernel / total, applied_scale


def imu_blur_kernel(
    gyro: np.ndarray,
    times: np.ndarray,
    centre_time: float,
    exposure_seconds: float,
    *,
    focal_length_px: float,
    angular_gain: float = 1.0,
    samples: int = 25,
    max_radius_px: float = 16.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Blur kernel for one frame plus the measurements that produced it."""
    u, v, roll_radians = exposure_path(
        gyro, times, centre_time, exposure_seconds,
        focal_length_px=focal_length_px, angular_gain=angular_gain, samples=samples,
    )
    kernel, applied_scale = path_to_kernel(u, v, max_radius_px=max_radius_px)
    span_x = float(u.max() - u.min())
    span_y = float(v.max() - v.min())
    report = {
        "exposure_seconds": float(exposure_seconds),
        "path_span_px": float(math.hypot(span_x, span_y)),
        "path_angle_radians": float(math.atan2(v[-1] - v[0], u[-1] - u[0])),
        "roll_radians": roll_radians,
        "kernel_radius_px": float((kernel.shape[0] - 1) // 2),
        "path_clip_scale": float(applied_scale),
    }
    return kernel, report
