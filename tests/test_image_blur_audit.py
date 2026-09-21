"""The blur audit must reward edges at the correct position, not just energy."""

import torch
import torch.nn.functional as F

from tools.image_blur_audit import edge_components


def test_edge_error_detects_true_recovery_and_misplaced_edges():
    clean = torch.zeros(1, 3, 32, 32)
    clean[..., 16:] = 1
    blurred = F.avg_pool2d(clean, 5, stride=1, padding=2)
    shifted = torch.zeros_like(clean)
    shifted[..., 18:] = 1

    recovered = edge_components(clean, blurred, clean)
    misplaced = edge_components(clean, blurred, shifted)

    assert recovered["edge_count"] > 0
    assert recovered["edge_restored_error_sum"] == 0
    assert recovered["edge_input_error_sum"] > 0
    assert misplaced["edge_restored_error_sum"] > recovered["edge_restored_error_sum"]
    assert misplaced["edge_restored_sum"] < recovered["edge_restored_sum"]
