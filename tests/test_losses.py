import torch

from qjepa.training.losses import (
    dense_positions,
    first_difference_l1,
    phase2_reconstruction_loss,
    variance_covariance_loss,
)


def test_constant_and_position_only_features_are_penalized_across_batch():
    constant = torch.zeros(8, 4, 16, requires_grad=True)
    variance, covariance = variance_covariance_loss(constant)
    assert variance > 0.98
    assert covariance == 0

    position_pattern = torch.arange(4.0)[None, :, None].expand(8, 4, 16).clone()
    variance, _ = variance_covariance_loss(position_pattern)
    assert variance > 0.98


def test_covariance_uses_b_minus_one_and_excludes_diagonal():
    values = torch.tensor(
        [
            [[1.0, 2.0]],
            [[2.0, 4.0]],
            [[3.0, 6.0]],
        ],
        requires_grad=True,
    )
    _, covariance_loss = variance_covariance_loss(values)
    # Sample covariance is [[1,2],[2,4]]. Off-diagonal square sum / D = 4.
    assert torch.allclose(covariance_loss, torch.tensor(4.0))
    covariance_loss.backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_dense_positions_never_flattens_positions_into_batch():
    feature = torch.randn(8, 16, 3, 3)
    selected = dense_positions(feature, torch.tensor([0, 4, 8]))
    assert selected.shape == (8, 3, 16)



def test_first_difference_catches_jitter_that_per_sample_loss_cannot_see():
    """Hai du doan co CUNG L1 tren tung mau, chi khac o do rung."""
    clean = torch.zeros(2, 6, 32)
    offset = torch.full_like(clean, 0.1)          # lech deu, hoan toan muot
    jitter = torch.full_like(clean, 0.1)
    jitter[..., 1::2] *= -1                       # cung |sai so|, doi dau moi mau

    assert torch.allclose(
        torch.nn.functional.l1_loss(offset, clean),
        torch.nn.functional.l1_loss(jitter, clean),
    )
    # ...nhung sai phan bac mot tach duoc chung, va do la ca ly do so hang nay ton tai.
    assert first_difference_l1(offset, clean) == 0.0
    assert torch.allclose(first_difference_l1(jitter, clean), torch.tensor(0.2))


def test_phase2_variation_term_is_opt_in_and_adds_exactly_its_weight():
    torch.manual_seed(0)
    image, image_clean = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    imu, imu_clean = torch.randn(2, 6, 16), torch.randn(2, 6, 16)

    base, base_parts = phase2_reconstruction_loss(image, image_clean, imu, imu_clean, beta=0.05)
    assert "imu_accel_variation_l1" not in base_parts

    total, parts = phase2_reconstruction_loss(
        image, image_clean, imu, imu_clean, beta=0.05, variation_weight=0.5
    )
    expected = base + 0.5 * (parts["imu_accel_variation_l1"] + parts["imu_gyro_variation_l1"])
    assert torch.allclose(total, expected)
    assert parts["imu_accel_variation_l1"] > 0
