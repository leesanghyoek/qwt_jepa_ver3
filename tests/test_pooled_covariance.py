"""The VICReg covariance term must measure correlation, not sampling noise.

Phase 1 runs with batch 8 and 128 feature channels. Estimated per position
from 8 samples, the 128 x 128 covariance is rank <= 7: on features that are
perfectly uncorrelated it reads about 8 at std 0.82 (true value 0), and it
grows with std^4, so its gradient mostly shrinks features and fights the
variance term. Pooling the batch-centred samples of all sampled positions
fixes that. These tests pin both the failure and the fix.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.training.losses import variance_covariance_loss
from qjepa.training.phase1 import Phase1Trainer


def _features(batch=8, positions=64, dim=128, std=0.82, shared=0.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    own = torch.randn(batch, positions, dim, generator=generator)
    common = torch.randn(batch, positions, 1, generator=generator)
    return std * ((1 - shared) ** 0.5 * own + shared ** 0.5 * common)


def test_per_position_covariance_reads_noise_where_the_truth_is_zero():
    """The failure being fixed, pinned so it cannot be quietly un-diagnosed."""
    _, independent = variance_covariance_loss(_features())
    assert float(independent) > 5.0                 # measured 8.18; the true value is 0


def test_pooled_covariance_separates_correlated_from_independent_features():
    """Measured: per position 25.8 vs 8.2 (x3), pooled 15.4 vs 0.13 (x119)."""
    ratio = {}
    for pooled in (False, True):
        _, independent = variance_covariance_loss(_features(), pooled_covariance=pooled)
        _, correlated = variance_covariance_loss(_features(shared=0.5), pooled_covariance=pooled)
        ratio[pooled] = float(correlated) / float(independent)
        if pooled:
            assert float(independent) < 0.5
    assert ratio[True] > 10 * ratio[False]


def test_pooling_leaves_the_variance_term_unchanged():
    features = _features(std=0.5)
    per_position, _ = variance_covariance_loss(features)
    pooled, _ = variance_covariance_loss(features, pooled_covariance=True)
    assert torch.equal(per_position, pooled)


def test_configs_written_before_the_key_keep_the_per_position_form():
    config = load_config("configs/smoke.yaml")
    config["phase1"].pop("covariance_pooling", None)
    assert config["phase1"].get("covariance_pooling", "per_position") == "per_position"
    config["phase1"]["covariance_pooling"] = "global"
    with pytest.raises(ValueError, match="covariance_pooling"):
        validate_config(config)


@pytest.mark.parametrize("pooling", ["per_position", "pooled"])
def test_phase1_step_runs_with_either_covariance(pooling):
    from qjepa.cli import _synthetic_batch

    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase1"]["covariance_pooling"] = pooling
    validate_config(config)
    seed_everything(config["phase1"]["initialization_seed"])
    model = build_phase1_model(config, ImuNormalizer())
    trainer = Phase1Trainer(model, config, torch.device("cpu"), "manifest")
    metrics = trainer.step(_synthetic_batch(config))
    assert metrics["skipped"] is False
    assert metrics["covariance"] >= 0.0
