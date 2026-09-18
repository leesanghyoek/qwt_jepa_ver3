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


def test_phase2_rejects_negative_variation_weight_and_non_positive_beta():
    config = load_config("configs/pipeline_v3.yaml")
    for key, value in (
        ("imu_variation_weight", -0.1),
        ("smooth_l1_beta", 0.0),
        ("smooth_l1_beta", -1.0),
    ):
        bad = copy.deepcopy(config)
        bad["phase2"][key] = value
        with pytest.raises(ValueError, match=key):
            validate_config(bad)


def test_phase_contracts_reject_unimplemented_paths_and_lying_config():
    config = load_config("configs/smoke.yaml")
    for section, key, value in (
        ("phase1", "decoder_enabled", True),          # bat decoder ma trong so 0
        ("phase1", "reconstruction_loss_weight", 1.0),  # khong co duong pixel-space
        ("phase2", "encoder_skips", True),            # chua duoc cai dat
        ("phase2", "reconstruction_detail_weight", -1.0),
    ):
        bad = copy.deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_residual_mode_must_match_its_declared_output():
    config = load_config("configs/smoke.yaml")
    for residual, declared in ((True, "absolute_prediction"), (False, "input_residual")):
        bad = copy.deepcopy(config)
        bad["phase2"]["input_coefficient_residual"] = residual
        bad["phase2"]["output_coefficients"] = declared
        with pytest.raises(ValueError, match="output_coefficients"):
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
