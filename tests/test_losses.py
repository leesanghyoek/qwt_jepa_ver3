import torch

from qjepa.training.losses import dense_positions, variance_covariance_loss


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

