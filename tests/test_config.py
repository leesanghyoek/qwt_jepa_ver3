import copy

import pytest

from qjepa.cli import _latent_gate
from qjepa.config import load_config, validate_config


def test_main_config_enforces_real_batch_eight():
    config = load_config("configs/pipeline_v3.yaml")
    bad = copy.deepcopy(config)
    bad["phase1"]["batch_size"] = 4
    with pytest.raises(ValueError, match="physical batch"):
        validate_config(bad)


def test_phase_contracts_reject_decoder_or_bypass():
    config = load_config("configs/smoke.yaml")
    for section, key, value in (
        ("phase1", "decoder_enabled", True),
        ("phase1", "reconstruction_loss_weight", 1.0),
        ("phase2", "encoder_skips", True),
        ("phase2", "input_coefficient_residual", True),
    ):
        bad = copy.deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_latent_gate_detects_collapsed_diversity():
    reference = {
        "validation_clean_ZI_same_position_std": 1.0,
        "validation_clean_ZI_pooled_effective_rank": 4.0,
        "validation_clean_ZI_raw_rms": 1.0,
    }
    monitor = load_config("configs/smoke.yaml")["monitor"]
    passed, reasons = _latent_gate(reference, dict(reference), monitor)
    assert passed and not reasons
    collapsed = dict(reference)
    collapsed["validation_clean_ZI_same_position_std"] = 0.01
    passed, reasons = _latent_gate(reference, collapsed, monitor)
    assert not passed and reasons
