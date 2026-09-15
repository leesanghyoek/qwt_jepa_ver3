import numpy as np

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

