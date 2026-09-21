import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from qjepa.cli import _fixed_validation_bank, _read_imu_csv, main
from qjepa.config import load_config, serializable_config
from qjepa.data.dataset import collate_paired
from qjepa.evaluation.metrics import latent_diagnostics
from qjepa.models.fusion import build_time_metadata
from qjepa.training.checkpoints import load_checkpoint


def test_unix_timestamp_precision_survives_collation_and_metadata():
    times = 1_750_000_000 + np.arange(128) * 0.01
    image_time = (times[0] + times[-1]) / 2
    item = {
        "image_clean": torch.zeros(3, 8, 8), "image_noisy": torch.zeros(3, 8, 8),
        "imu_clean_phys": torch.zeros(128, 6), "imu_noisy_phys": torch.zeros(128, 6),
        "image_time": torch.tensor(image_time, dtype=torch.float64),
        "imu_times": torch.from_numpy(times), "imu_start": torch.tensor(0),
        "sample_id": "sample", "trajectory_key": "trajectory", "corruption": {},
    }
    batch = collate_paired([item])
    assert batch["imu_times"].dtype == torch.float64
    metadata = build_time_metadata(batch["image_time"], batch["imu_times"])
    assert torch.isfinite(metadata).all()
    assert abs(float(metadata[0, 0])) < 1e-6
    bad = batch["imu_times"].clone()
    bad[0, 5] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_time_metadata(batch["image_time"], bad)


def test_constant_rank_is_zero_and_validation_bank_spans_trajectories():
    assert latent_diagnostics(torch.ones(8, 16, 4))["pooled_effective_rank"] == 0
    dataset = SimpleNamespace(samples=[
        SimpleNamespace(trajectory_key=key, image_time=index)
        for key in ("A", "B") for index in range(100)
    ])
    _fixed_validation_bank(dataset, 8)
    assert len(dataset.samples) == 8
    for key in ("A", "B"):
        times = [sample.image_time for sample in dataset.samples if sample.trajectory_key == key]
        assert min(times) == 0 and max(times) == 99


def test_csv_export_header_can_be_read_back(tmp_path):
    path = tmp_path / "imu.csv"
    data = np.column_stack((1_750_000_000 + np.arange(32) * 0.01, np.ones((32, 6))))
    np.savetxt(path, data, delimiter=",", header="timestamp,ax,ay,az,gx,gy,gz", comments="")
    values, times = _read_imu_csv(str(path), 32)
    assert values.shape == (32, 6)
    np.testing.assert_array_equal(times, data[:, 0])


def _write_dataset(root: Path):
    for split_index, split in enumerate(("train", "valid", "test")):
        path = root / split / f"env_{split}" / "Data_easy" / "P000"
        (path / "imu").mkdir(parents=True)
        (path / "image_lcam_front").mkdir()
        relative = np.arange(96) * 0.01
        times = 1_750_000_000 + relative
        imu = np.stack([np.sin(relative * (axis + 1) * 3 + split_index) for axis in range(6)], axis=-1)
        np.save(path / "imu/imu_time.npy", times)
        np.save(path / "imu/acc.npy", imu[:, :3])
        np.save(path / "imu/gyro.npy", imu[:, 3:])
        starts = (0, 16, 32, 48)
        np.save(path / "imu/cam_time.npy", [(times[i] + times[i + 31]) / 2 for i in starts])
        rng = np.random.default_rng(split_index)
        for index in range(4):
            pixels = rng.integers(20, 240, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path / f"image_lcam_front/{index:06d}_lcam_front.png")


def test_two_phase_cli_generates_kaggle_reports_and_preserves_best_on_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase2"]["max_successful_updates"] = 2
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))
    common = ["--config", str(config_path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(config_path), "--data-root", str(root), "--output", str(manifest)])
    main(["train-phase1", *common])
    parent = output / "phase1/last.pt"
    import qjepa.cli as cli
    original_save, original_evaluate = cli.atomic_torch_save, cli._evaluate_with_overlap

    def interrupt_after_first_phase2_checkpoint(payload, path):
        original_save(payload, path)
        if path.parent.name == "phase2" and path.name == "last.pt" and payload["successful_updates"] == 1:
            raise RuntimeError("Simulated session interruption after a saved update")

    scores = iter((1.0, 2.0))

    def validation_with_worsening_score(*args, **kwargs):
        values = original_evaluate(*args, **kwargs)
        values["joint_validation_score"] = next(scores)
        return values

    monkeypatch.setattr(cli, "atomic_torch_save", interrupt_after_first_phase2_checkpoint)
    monkeypatch.setattr(cli, "_evaluate_with_overlap", validation_with_worsening_score)
    with pytest.raises(SystemExit):
        main(["train-phase2", *common, "--backbone-checkpoint", str(parent)])
    monkeypatch.setattr(cli, "atomic_torch_save", original_save)
    last = output / "phase2/last.pt"
    payload = load_checkpoint(last)
    assert np.isfinite(payload["best_joint_validation_score"])
    best_path = output / "phase2/best_joint_validation.pt"
    best_bytes = best_path.read_bytes()
    main(["train-phase2", *common, "--backbone-checkpoint", str(parent), "--resume", str(last)])
    assert best_path.read_bytes() == best_bytes
    resumed = load_checkpoint(last)
    assert resumed["successful_updates"] == 2
    assert resumed["best_joint_validation_score"] == 1.0
    monkeypatch.setattr(cli, "_evaluate_with_overlap", original_evaluate)
    evaluation = output / "test_results"
    main(["evaluate", "--checkpoint", str(best_path), "--manifest", str(manifest),
          "--device", "cpu", "--output", str(evaluation), "--panels", "1"])
    metrics = json.loads((evaluation / "metrics.json").read_text())["requested"]
    assert metrics["image_count"] == 4
    assert metrics["imu_covered_unique_rows"] == 80  # four overlapping 32-row windows
    assert metrics["baseline_accel_rmse"] > 0
    assert len(list(evaluation.glob("requested/images/*.png"))) == 1
    merged = np.load(next(evaluation.glob("requested/imu/*.npz")))
    assert merged["coverage"].sum() == 80
    assert np.diff(merged["timestamp"][merged["coverage"]]).min() > 0
    for phase in ("phase1", "phase2"):
        assert (output / phase / "history.csv").exists()
        with Image.open(output / phase / "training_curves.png") as image:
            assert image.width > 1000
    with Image.open(evaluation / "comparison.png") as image:
        assert image.width > 1000
    with pytest.raises(SystemExit):
        main(["train-phase1", *common])  # cannot silently append another run


def test_guarded_phase2_starts_from_old_decoder_and_resumes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    base = serializable_config(load_config("configs/smoke.yaml"))
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    old_run, new_run = tmp_path / "old", tmp_path / "guarded"
    main(["build-manifest", "--config", str(base_path), "--data-root", str(root),
          "--output", str(manifest)])
    main(["train-phase1", "--config", str(base_path), "--manifest", str(manifest),
          "--output", str(old_run)])
    parent = old_run / "phase1/last.pt"
    main(["train-phase2", "--config", str(base_path), "--manifest", str(manifest),
          "--output", str(old_run), "--backbone-checkpoint", str(parent)])
    old_decoder = old_run / "phase2/best_joint_validation.pt"
    old_hash = load_checkpoint(old_decoder)["metadata"]["decoder_current_hash"]

    guarded = serializable_config(load_config(base_path))
    guarded["phase2"].update(
        max_successful_updates=2,
        blur_validation_samples=4,
        train_scenarios=[
            {"image_mode": "full", "imu_mode": "full", "weight": 0.8},
            {"image_mode": "blur_only", "imu_mode": "clean", "weight": 0.2},
        ],
        full_guard={"max_psnr_drop_db": 0.2, "max_ssim_drop": 0.01,
                    "max_accel_rmse_ratio": 1.03, "max_gyro_rmse_ratio": 1.03},
    )
    guarded_path = tmp_path / "guarded.yaml"
    guarded_path.write_text(yaml.safe_dump(guarded))
    import qjepa.cli as cli
    original_save = cli.atomic_torch_save

    def interrupt_after_first(payload, path):
        original_save(payload, path)
        if path == new_run / "phase2/last.pt" and payload["successful_updates"] == 1:
            raise RuntimeError("Simulated interruption after guarded checkpoint")

    monkeypatch.setattr(cli, "atomic_torch_save", interrupt_after_first)
    args = ["--config", str(guarded_path), "--manifest", str(manifest),
            "--output", str(new_run), "--backbone-checkpoint", str(parent)]
    with pytest.raises(SystemExit):
        main(["train-phase2", *args, "--decoder-init-checkpoint", str(old_decoder)])
    monkeypatch.setattr(cli, "atomic_torch_save", original_save)
    last = new_run / "phase2/last.pt"
    saved = load_checkpoint(last)
    assert saved["metadata"]["decoder_initialization_hash"] == old_hash
    assert saved["decoder_init_checkpoint"] == str(old_decoder.resolve())
    assert saved["guard_reference"]["full"]["image_count"] == 4
    assert saved["guard_reference"]["blur"]["active_frames"] > 0
    assert (new_run / "phase2/guard_reference.json").is_file()
    main(["train-phase2", *args, "--resume", str(last)])
    resumed = load_checkpoint(last)
    assert resumed["successful_updates"] == 2
    assert resumed["decoder_init_checkpoint"] == str(old_decoder.resolve())
    assert resumed["guard_reference"] == saved["guard_reference"]
