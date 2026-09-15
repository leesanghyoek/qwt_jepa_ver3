import copy

import pytest
import torch
from torch import nn

from qjepa.cli import _synthetic_batch, _system_from_phase2
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.execution import select_device_ids
from qjepa.models import RestorationSystem
from qjepa.training.checkpoints import atomic_torch_save, configuration_hash, restore_rng_state, rng_state
from qjepa.training.losses import variance_covariance_loss
from qjepa.training.phase1 import Phase1Trainer
from qjepa.training.phase2 import Phase2Trainer


@pytest.mark.parametrize("available,requested,primary,expected", [
    (1, "auto", 0, [0]), (2, "auto", 0, [0, 1]), (4, "auto", 0, [0, 1]),
    (2, 1, 1, [1]), (2, 2, 1, [1, 0]),
])
def test_gpu_selection(monkeypatch, available, requested, primary, expected):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: available)
    assert select_device_ids(torch.device(f"cuda:{primary}"), requested) == expected


def test_invalid_gpu_requests_and_cpu_fallback(monkeypatch):
    assert select_device_ids(torch.device("cpu"), "auto") == []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="only 1 visible"):
        select_device_ids(torch.device("cuda:0"), 2)
    with pytest.raises(ValueError, match="require a CUDA"):
        select_device_ids(torch.device("cpu"), 2)
    config = load_config("configs/smoke.yaml")
    config["runtime"]["gpu_count"] = 3
    with pytest.raises(ValueError, match="gpu_count"):
        validate_config(config)


def test_gpu_topology_is_not_part_of_training_semantics_hash():
    config = load_config("configs/smoke.yaml")
    before = [configuration_hash(config, phase) for phase in ("phase1", "phase2")]
    config["runtime"]["gpu_count"] = 2
    assert before == [configuration_hash(config, phase) for phase in ("phase1", "phase2")]


def test_rng_resume_ignores_states_for_missing_devices(monkeypatch):
    state = rng_state()
    state["cuda"] = [torch.zeros(8, dtype=torch.uint8), torch.ones(8, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    calls = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: calls.append(device))
    restore_rng_state(state)
    assert calls == [0]


class _TwoCPUChunks(nn.Module):
    """Exercise the scatter/gather contract without pretending to emulate CUDA."""
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *args, **kwargs):
        size = args[0].shape[0]
        outputs = []
        for start, end in ((0, size // 2), (size // 2, size)):
            sliced = [value[start:end] for value in args]
            options = {key: value[start:end] if isinstance(value, torch.Tensor) else value
                       for key, value in kwargs.items()}
            outputs.append(self.wrapped(*sliced, **options))
        return {key: torch.cat([output[key] for output in outputs], dim=0) for key in outputs[0]}


def test_batch_statistics_need_cross_gpu_samples():
    left, right = torch.zeros(4, 2, 3), torch.full((4, 2, 3), 4.0)
    local = (variance_covariance_loss(left)[0] + variance_covariance_loss(right)[0]) / 2
    global_value = variance_covariance_loss(torch.cat((left, right)))[0]
    assert local > 0.9 and global_value == 0


def test_phase1_gather_preserves_global_b8_loss_and_gradients(monkeypatch):
    torch.set_num_threads(1)
    seed_everything(19)
    config = load_config("configs/smoke.yaml")
    config["phase1"]["batch_size"] = 8
    config["phase1"]["minimum_statistics_batch"] = 8
    batch = _synthetic_batch(config)
    model = build_phase1_model(config, ImuNormalizer())
    first = Phase1Trainer(model, config, torch.device("cpu"))
    second = Phase1Trainer(copy.deepcopy(model), config, torch.device("cpu"))
    second.forward_model = _TwoCPUChunks(second.forward_model)
    # Both the image and IMU finite-difference paths must traverse the wrapper.
    import qjepa.training.phase1 as module
    observed = []
    original = module.variance_covariance_loss

    def record_shape(value, **kwargs):
        observed.append(value.shape[0])
        return original(value, **kwargs)

    monkeypatch.setattr(module, "variance_covariance_loss", record_shape)
    for source in ("image", "imu"):
        a, b = first.step(batch), second.step(batch)
        assert a["encoder_source"] == b["encoder_source"] == source
        for key in ("loss", "jepa", "variance", "covariance", "encoder_sensitivity"):
            assert a[key] == pytest.approx(b[key], rel=3e-4, abs=1e-5)
        for pa, pb in zip(first.parameters, second.parameters):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-3, atol=2e-5)
    assert observed == [8] * 32
    assert not any(key.startswith("module.") for key in second.checkpoint_payload(config)["model"])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two real CUDA GPUs")
def test_actual_two_gpu_training_and_single_gpu_checkpoint_loading(tmp_path):
    seed_everything(20)
    config = load_config("configs/smoke.yaml")
    config["runtime"]["gpu_count"] = 2
    config["phase1"].update(batch_size=8, minimum_statistics_batch=8)
    config["phase2"].update(batch_size=4, gradient_accumulation=1)
    device = torch.device("cuda:0")
    batch = _synthetic_batch(config)
    model = build_phase1_model(config, ImuNormalizer())
    trainer1 = Phase1Trainer(model, config, device)
    assert trainer1.device_ids == [0, 1]
    assert isinstance(trainer1.forward_model, nn.DataParallel)
    for _ in range(2):
        metrics = trainer1.step(batch)
        assert not metrics["skipped"]
    assert all(parameter.grad is None for parameter in model.teachers.parameters())
    system = RestorationSystem(model.backbone, model.normalizer, build_decoders(config))
    trainer2 = Phase2Trainer(system, config, device, "synthetic")
    metrics = trainer2.step(batch)
    assert not metrics["skipped"]
    trainer2.assert_backbone_frozen()
    path = tmp_path / "dual_gpu.pt"
    atomic_torch_save(trainer2.checkpoint_payload(config), path)
    loaded, _ = _system_from_phase2(str(path), device)
    with torch.no_grad():
        inputs = [batch[key].to(device) for key in ("image_noisy", "imu_noisy_phys", "image_time", "imu_times")]
        expected = system(*inputs)
        actual = loaded(*inputs)
    torch.testing.assert_close(expected.image, actual.image)
    torch.testing.assert_close(expected.imu_physical, actual.imu_physical)
