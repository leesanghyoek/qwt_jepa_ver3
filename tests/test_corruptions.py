import numpy as np
import pytest

from qjepa.corruptions import (
    ImuCorruptionConfig,
    LowLightImageCorruptionConfig,
    LowLightImageCorruptor,
    TrajectoryImuCorruptor,
)


def test_low_light_corruption_is_deterministic_and_actually_darkens():
    config = LowLightImageCorruptionConfig(clean_probability=0.0)
    corruptor = LowLightImageCorruptor(config, master_seed=7)
    image = np.full((32, 32, 3), 0.8, dtype=np.float32)
    kwargs = dict(
        split="train", realization=2, trajectory="env/P000", timestamp=1.2, frame_index=9
    )
    first, parameters = corruptor(image, **kwargs)
    second, _ = corruptor(image, **kwargs)
    different, _ = corruptor(image, **{**kwargs, "realization": 3})
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.mean() < image.mean() * 0.75
    assert parameters["exposure_gain"] <= 0.55


def test_named_image_corruption_groups_are_isolated():
    corruptor = LowLightImageCorruptor(
        LowLightImageCorruptionConfig(clean_probability=0.0), master_seed=3
    )
    image = np.random.default_rng(1).random((24, 24, 3), dtype=np.float32)
    common = dict(split="test", realization=0, trajectory="T", timestamp=0.0, frame_index=0)
    clean, _ = corruptor(image, mode="clean", **common)
    dark, _ = corruptor(image, mode="low_light_only", **common)
    blur, _ = corruptor(image, mode="blur_only", **common)
    assert np.array_equal(clean, image)
    assert dark.mean() < image.mean()
    assert not np.array_equal(blur, image)


def test_imu_overlaps_share_the_exact_same_corruption_trace():
    count = 220
    times = np.arange(count, dtype=np.float64) * 0.01
    clean = np.stack([np.sin(times * (axis + 1)) for axis in range(6)], axis=-1)
    corruptor = TrajectoryImuCorruptor(
        ImuCorruptionConfig(clean_probability=0.0), master_seed=11
    )
    first, _ = corruptor.window(
        clean, times, 10, 138, split="train", realization=0, trajectory="T", mode="full"
    )
    second, _ = corruptor.window(
        clean, times, 50, 178, split="train", realization=0, trajectory="T", mode="full"
    )
    assert np.array_equal(first[40:], second[:88])
    replica = TrajectoryImuCorruptor(
        ImuCorruptionConfig(clean_probability=0.0), master_seed=11
    )
    repeated, _ = replica.window(
        clean, times, 10, 138, split="train", realization=0, trajectory="T", mode="full"
    )
    assert np.array_equal(first, repeated)


def test_imu_vibration_adds_narrowband_tones_below_nyquist():
    rate, count = 100.0, 1024
    times = np.arange(count, dtype=np.float64) / rate
    clean = np.zeros((count, 6))
    config = ImuCorruptionConfig(
        clean_probability=0.0,
        accel_white_noise_std=(0.0, 0.0),
        gyro_white_noise_std=(0.0, 0.0),
        accel_bias_random_walk=0.0,
        gyro_bias_random_walk=0.0,
        spike_rate_hz=0.0,
        dropout_rate_hz=0.0,
        quantization_step_accel=(0.0, 0.0),
        quantization_step_gyro=(0.0, 0.0),
        vibration_tones=(2, 2),
        accel_vibration_amplitude=(0.2, 0.2),
        accel_wander_std=(0.0, 0.0),
        gyro_wander_std=(0.0, 0.0),
    )
    corrupted, parameters = TrajectoryImuCorruptor(config, master_seed=5).trajectory(
        clean, times, split="train", realization=0, trajectory="T", mode="full"
    )
    tones = parameters["vibration_tones"]
    assert len(tones) == 2
    frequencies = np.fft.rfftfreq(count, 1.0 / rate)
    spectrum = np.abs(np.fft.rfft(corrupted[:, 0] - corrupted[:, 0].mean()))
    for tone in tones:
        assert tone["frequency_hz"] < rate / 2.0
        nearest = int(np.argmin(np.abs(frequencies - tone["frequency_hz"])))
        # Mot tone bang hep phai troi han nen pho quanh no.
        assert spectrum[nearest] > 20 * np.median(spectrum)


def test_imu_vibration_frequency_is_clamped_for_slow_sampling():
    rate, count = 40.0, 512      # Nyquist 20 Hz, duoi day tan so cau hinh
    times = np.arange(count, dtype=np.float64) / rate
    clean = np.zeros((count, 6))
    config = ImuCorruptionConfig(
        clean_probability=0.0, vibration_tones=(3, 3), vibration_frequency_hz=(30.0, 45.0)
    )
    _, parameters = TrajectoryImuCorruptor(config, master_seed=5).trajectory(
        clean, times, split="train", realization=0, trajectory="T", mode="full"
    )
    assert parameters["vibration_tones"]
    assert all(tone["frequency_hz"] <= 0.45 * rate for tone in parameters["vibration_tones"])


def test_imu_wander_is_bounded_and_does_not_diverge_like_a_random_walk():
    rate, count = 100.0, 20_000        # 200 s: du dai de random walk lo ro
    times = np.arange(count, dtype=np.float64) / rate
    clean = np.zeros((count, 6))
    config = ImuCorruptionConfig(
        clean_probability=0.0,
        accel_white_noise_std=(0.0, 0.0),
        gyro_white_noise_std=(0.0, 0.0),
        accel_bias_bound=0.0,
        gyro_bias_bound=0.0,
        accel_bias_random_walk=0.0,
        gyro_bias_random_walk=0.0,
        spike_rate_hz=0.0,
        dropout_rate_hz=0.0,
        vibration_tones=(0, 0),
        quantization_step_accel=(0.0, 0.0),
        quantization_step_gyro=(0.0, 0.0),
        wander_probability=1.0,
        accel_wander_std=(0.4, 0.4),
        wander_seconds=(1.0, 1.0),
    )
    corrupted, parameters = TrajectoryImuCorruptor(config, master_seed=13).trajectory(
        clean, times, split="train", realization=0, trajectory="T", mode="full"
    )
    assert parameters["wander"] and parameters["wander_seconds"] == 1.0
    channel = corrupted[:, 0]
    assert channel.std() == pytest.approx(0.4, rel=0.05)
    # Nua sau khong on hon nua dau: dao dong co gioi han, khong phai random walk.
    first, second = channel[: count // 2], channel[count // 2 :]
    assert second.std() == pytest.approx(first.std(), rel=0.2)
    assert np.abs(channel).max() < 6.0 * 0.4


def test_imu_wander_only_affects_a_minority_of_trajectories():
    rate, count = 100.0, 1024
    times = np.arange(count, dtype=np.float64) / rate
    clean = np.zeros((count, 6))
    config = ImuCorruptionConfig(clean_probability=0.0, wander_probability=0.25)
    flags = []
    for index in range(600):
        _, parameters = TrajectoryImuCorruptor(config, master_seed=3).trajectory(
            clean, times, split="train", realization=0, trajectory=f"T{index:04d}", mode="full"
        )
        flags.append(parameters["wander"])
    assert 0.20 < np.mean(flags) < 0.30
