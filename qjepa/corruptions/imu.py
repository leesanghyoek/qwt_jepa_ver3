"""Trajectory-consistent IMU degradation in physical SI units."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .rng import generator

IMU_MODES = ("full", "clean", "white_noise_only", "bias_only", "bandwidth_only")


@dataclass(frozen=True)
class ImuCorruptionConfig:
    clean_probability: float = 0.05
    accel_white_noise_std: tuple[float, float] = (0.02, 0.45)
    gyro_white_noise_std: tuple[float, float] = (0.001, 0.025)
    accel_bias_bound: float = 0.12
    gyro_bias_bound: float = 0.008
    accel_bias_random_walk: float = 0.008
    gyro_bias_random_walk: float = 0.0004
    scale_error_std: float = 0.006
    cross_axis_std: float = 0.003
    lowpass_sigma_samples: tuple[float, float] = (0.0, 2.5)
    accel_wander_std: tuple[float, float] = (0.15, 1.60)
    gyro_wander_std: tuple[float, float] = (0.010, 0.110)
    wander_seconds: tuple[float, float] = (0.3, 2.0)
    noise_gain_drift: tuple[float, float] = (0.25, 4.0)
    noise_drift_seconds: float = 1.5
    quantization_step_accel: tuple[float, float] = (0.0, 0.008)
    quantization_step_gyro: tuple[float, float] = (0.0, 0.0004)
    vibration_tones: tuple[int, int] = (0, 3)
    vibration_frequency_hz: tuple[float, float] = (8.0, 45.0)
    vibration_modulation_hz: tuple[float, float] = (0.2, 1.5)
    accel_vibration_amplitude: tuple[float, float] = (0.0, 0.35)
    gyro_vibration_amplitude: tuple[float, float] = (0.0, 0.02)
    spike_rate_hz: float = 0.6
    accel_spike_amplitude: tuple[float, float] = (0.25, 2.5)
    gyro_spike_amplitude: tuple[float, float] = (0.01, 0.15)
    dropout_rate_hz: float = 0.10
    dropout_length_samples: tuple[int, int] = (2, 8)

    def validate(self) -> None:
        if not 0 <= self.clean_probability <= 1:
            raise ValueError("clean_probability must be in [0,1]")
        if self.accel_white_noise_std[0] < 0 or self.gyro_white_noise_std[0] < 0:
            raise ValueError("Noise standard deviations cannot be negative")


class TrajectoryImuCorruptor:
    """Builds a full corrupted trajectory before slicing overlapping windows."""

    def __init__(
        self,
        config: ImuCorruptionConfig | None = None,
        master_seed: int = 73128,
        cache_size: int = 8,
    ) -> None:
        self.config = config or ImuCorruptionConfig()
        self.config.validate()
        self.master_seed = master_seed
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[object, ...], tuple[np.ndarray, dict[str, object]]] = OrderedDict()

    def _build(
        self,
        imu_clean: np.ndarray,
        timestamps: np.ndarray,
        *,
        split: str,
        realization: int,
        trajectory: str,
        mode: str,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if mode not in IMU_MODES:
            raise ValueError(f"Unsupported IMU corruption mode {mode!r}")
        if imu_clean.ndim != 2 or imu_clean.shape[1] != 6:
            raise ValueError(f"Expected IMU [N,6], got {imu_clean.shape}")
        if timestamps.shape != (len(imu_clean),) or (np.diff(timestamps) <= 0).any():
            raise ValueError("IMU timestamps must be a matching, strictly increasing vector")
        if not np.isfinite(imu_clean).all() or not np.isfinite(timestamps).all():
            raise ValueError("IMU data contains NaN/Inf")

        cfg = self.config
        rng = generator(self.master_seed, "imu_trajectory", split, realization, trajectory)
        clean = mode == "clean" or (mode == "full" and rng.random() < cfg.clean_probability)
        sigma = np.r_[
            rng.uniform(*cfg.accel_white_noise_std, 3),
            rng.uniform(*cfg.gyro_white_noise_std, 3),
        ]
        lowpass_sigma = float(rng.uniform(*cfg.lowpass_sigma_samples))
        scale = rng.normal(1.0, cfg.scale_error_std, 6)
        mixing = np.eye(6)
        for offset in (0, 3):
            block = rng.normal(0.0, cfg.cross_axis_std, (3, 3))
            np.fill_diagonal(block, 0.0)
            mixing[offset : offset + 3, offset : offset + 3] += block
        parameters: dict[str, object] = {
            "mode": mode,
            "clean": bool(clean),
            "white_noise_std": sigma.tolist(),
            "scale": scale.tolist(),
            "mixing": mixing.tolist(),
            "lowpass_sigma_samples": lowpass_sigma,
            "dropout_events": 0,
            "spike_events": 0,
        }
        if clean:
            return imu_clean.astype(np.float64, copy=True), parameters

        white_noise = mode in ("full", "white_noise_only")
        bias_noise = mode in ("full", "bias_only")
        bandwidth = mode in ("full", "bandwidth_only")
        out = imu_clean.astype(np.float64, copy=True)

        if bandwidth and lowpass_sigma > 1e-6:
            out = ndimage.gaussian_filter1d(out, lowpass_sigma, axis=0, mode="nearest")
        if mode == "full":
            out = (out * scale[None, :]) @ mixing.T

        if bias_noise:
            bounds = np.r_[[cfg.accel_bias_bound] * 3, [cfg.gyro_bias_bound] * 3]
            walk = np.r_[[cfg.accel_bias_random_walk] * 3, [cfg.gyro_bias_random_walk] * 3]
            bias0 = rng.uniform(-bounds, bounds)
            dt = np.diff(timestamps)
            increments = rng.normal(size=(len(out) - 1, 6)) * walk[None, :] * np.sqrt(dt)[:, None]
            bias = bias0 + np.vstack((np.zeros((1, 6)), np.cumsum(increments, axis=0)))
            out += bias
            parameters["bias_initial"] = bias0.tolist()

            # Bias instability: dao dong ngau nhien bang hep quanh gia tri that.
            # Khac random walk o cho co gioi han, nen khong troi vo han theo thoi gian.
            step = float(np.median(dt))
            correlation = float(rng.uniform(*cfg.wander_seconds))
            amplitude = np.r_[
                rng.uniform(*cfg.accel_wander_std, 3), rng.uniform(*cfg.gyro_wander_std, 3)
            ]
            rough = ndimage.gaussian_filter1d(
                rng.normal(size=out.shape), max(1.0, correlation / step), axis=0, mode="wrap"
            )
            rough /= rough.std(axis=0, keepdims=True) + 1e-12
            out += rough * amplitude[None, :]
            parameters["wander_std"] = amplitude.tolist()
            parameters["wander_seconds"] = correlation

        if white_noise:
            # Nen nhieu khong dung: troi cham doc trajectory nen moi window thay
            # mot muc khac nhau, trong khi cac window chong nhau van khop tuyet doi.
            step = float(np.median(np.diff(timestamps)))
            drift = ndimage.gaussian_filter1d(
                rng.normal(size=len(out)), max(1.0, cfg.noise_drift_seconds / step), mode="wrap"
            )
            drift = (drift - drift.mean()) / (drift.std() + 1e-12)
            low, high = cfg.noise_gain_drift
            envelope = np.exp(
                np.log(low) + (np.log(high) - np.log(low)) * 0.5 * (1.0 + np.tanh(drift))
            )
            out += rng.normal(size=out.shape) * sigma[None, :] * envelope[:, None]
            parameters["noise_gain_range"] = [float(envelope.min()), float(envelope.max())]

        if mode == "full":
            median_dt = float(np.median(np.diff(timestamps)))
            duration = float(timestamps[-1] - timestamps[0])

            # Rung co hoc: vai tone bang hep, bien do bien thien cham -> khong dung.
            elapsed = timestamps - timestamps[0]
            ceiling = 0.45 / median_dt  # giu duoi Nyquist cho moi tan so lay mau
            tones: list[dict[str, object]] = []
            for _ in range(int(rng.integers(cfg.vibration_tones[0], cfg.vibration_tones[1] + 1))):
                frequency = min(float(rng.uniform(*cfg.vibration_frequency_hz)), ceiling)
                amplitude = np.r_[
                    [rng.uniform(*cfg.accel_vibration_amplitude)] * 3,
                    [rng.uniform(*cfg.gyro_vibration_amplitude)] * 3,
                ] * rng.uniform(0.4, 1.0, 6)
                phase = rng.uniform(0.0, 2.0 * np.pi, 6)
                envelope = 0.6 + 0.4 * np.sin(
                    2.0 * np.pi * float(rng.uniform(*cfg.vibration_modulation_hz)) * elapsed
                    + float(rng.uniform(0.0, 2.0 * np.pi))
                )
                carrier = np.sin(2.0 * np.pi * frequency * elapsed[:, None] + phase[None, :])
                out += carrier * envelope[:, None] * amplitude[None, :]
                tones.append({"frequency_hz": frequency, "amplitude": amplitude.tolist()})
            parameters["vibration_tones"] = tones

            for group, amplitude in ((slice(0, 3), cfg.accel_spike_amplitude), (slice(3, 6), cfg.gyro_spike_amplitude)):
                count = int(rng.poisson(cfg.spike_rate_hz * duration))
                for _ in range(count):
                    row = int(rng.integers(0, len(out)))
                    axis = int(rng.integers(group.start, group.stop))
                    out[row, axis] += rng.choice((-1.0, 1.0)) * rng.uniform(*amplitude)
                parameters["spike_events"] = int(parameters["spike_events"]) + count

            dropout_count = int(rng.poisson(cfg.dropout_rate_hz * duration))
            for _ in range(dropout_count):
                start = int(rng.integers(1, max(2, len(out) - 1)))
                length = int(rng.integers(cfg.dropout_length_samples[0], cfg.dropout_length_samples[1] + 1))
                end = min(len(out), start + length)
                out[start:end] = out[start - 1]
            parameters["dropout_events"] = dropout_count
            parameters["median_dt"] = median_dt

            steps = np.r_[
                [rng.uniform(*cfg.quantization_step_accel)] * 3,
                [rng.uniform(*cfg.quantization_step_gyro)] * 3,
            ]
            nonzero = steps > 0
            out[:, nonzero] = np.round(out[:, nonzero] / steps[nonzero]) * steps[nonzero]
            parameters["quantization_step"] = steps.tolist()
        return out, parameters

    def trajectory(
        self,
        imu_clean: np.ndarray,
        timestamps: np.ndarray,
        *,
        split: str,
        realization: int,
        trajectory: str,
        mode: str = "full",
    ) -> tuple[np.ndarray, dict[str, object]]:
        key = (split, realization, trajectory, mode)
        if key not in self._cache:
            self._cache[key] = self._build(
                imu_clean,
                timestamps,
                split=split,
                realization=realization,
                trajectory=trajectory,
                mode=mode,
            )
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return self._cache[key]

    def window(
        self,
        imu_clean_trajectory: np.ndarray,
        timestamps: np.ndarray,
        start: int,
        end: int,
        **context,
    ) -> tuple[np.ndarray, dict[str, object]]:
        corrupted, parameters = self.trajectory(imu_clean_trajectory, timestamps, **context)
        if not 0 <= start < end <= len(corrupted):
            raise IndexError(f"Invalid IMU window [{start}:{end}] for length {len(corrupted)}")
        return corrupted[start:end].astype(np.float32, copy=True), parameters

    def metadata(self) -> dict[str, object]:
        return {"type": type(self).__name__, "master_seed": self.master_seed, "config": asdict(self.config)}

