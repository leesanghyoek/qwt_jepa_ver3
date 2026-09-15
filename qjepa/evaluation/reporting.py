"""Headless PNG/CSV/JSON reports shared by the CLI and Kaggle notebooks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_training(run_dir: str | Path) -> list[Path]:
    """Plot sparse validation events at their actual optimizer steps; never interpolate."""
    plt = _pyplot()
    root = Path(run_dir)
    phases = [root] if (root / "train.jsonl").exists() else [root / "phase1", root / "phase2"]
    written = []
    for phase in phases:
        log = phase / "train.jsonl"
        if not log.exists():
            continue
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if not records:
            continue
        merged: dict[int, dict[str, Any]] = {}
        for record in records:
            step = int(record.get("successful_updates", 0))
            merged.setdefault(step, {"successful_updates": step}).update(record)
        rows = [merged[step] for step in sorted(merged)]
        write_csv(phase / "history.csv", rows)
        config_path = phase / "resolved_config.yaml"
        run_kind = "unknown"
        if config_path.exists():
            import yaml

            run_kind = yaml.safe_load(config_path.read_text()).get("run_kind", "unknown")
        phase1 = any("jepa" in row for row in rows)
        groups = (
            [
                ("Total and JEPA losses", ["loss", "jepa", "validation_jepa"]),
                ("Variance and covariance losses", ["variance", "covariance"]),
                ("Learning rate", ["learning_rate"]),
                ("Gradient norm", ["gradient_norm"]),
                ("Encoder sensitivity (normalized gain)", ["encoder_sensitivity"]),
                ("Sensitivity weight", ["encoder_sensitivity_weight"]),
                ("Teacher EMA momentum", ["teacher_momentum"]),
                ("Clipped image probe fraction", ["probe_clipped_fraction"]),
            ] if phase1 else [
                ("Reconstruction loss", ["loss", "image_l1", "imu_accel_smooth_l1", "imu_gyro_smooth_l1", "validation_joint_validation_score"]),
                ("Learning rate", ["learning_rate"]),
                ("Validation image PSNR (dB)", ["validation_image_psnr_db", "validation_baseline_image_psnr_db"]),
                ("Validation image SSIM", ["validation_image_ssim", "validation_baseline_image_ssim"]),
                ("Validation acceleration RMSE (m/s²)", ["validation_accel_rmse", "validation_baseline_accel_rmse"]),
                ("Validation gyro RMSE (rad/s)", ["validation_gyro_rmse", "validation_baseline_gyro_rmse"]),
                ("Gradient norm", ["gradient_norm"]),
            ]
        )
        fig, axes = plt.subplots(4, 2, figsize=(17, 19), constrained_layout=True)
        for ax, (title, keys) in zip(axes.flat, groups):
            for key in keys:
                points = [(row["successful_updates"], row[key]) for row in rows
                          if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])]
                if points:
                    x, y = zip(*points)
                    ax.plot(x, y, marker="." if len(x) < 30 else None,
                            linestyle="--" if key.startswith("validation_") else "-",
                            label=key.replace("validation_", "val: "))
            ax.set(title=title, xlabel="Successful optimizer update")
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend(fontsize=6)
            else:
                ax.text(0.5, 0.5, "No recorded values", ha="center", transform=ax.transAxes)
        for ax in list(axes.flat)[len(groups):]:
            ax.axis("off")
        last_validation = next((row for row in reversed(rows) if any(key.startswith("validation_") for key in row)), {})
        summary = {"phase": phase.name, "run_kind": run_kind,
                   "last_successful_update": max(merged, default=0),
                   "last_record": rows[-1] if rows else {}, "last_validation": last_validation}
        write_json(phase / "training_summary.json", summary)
        fig.suptitle(f"{phase.name} — {run_kind.upper()} — recorded training / validation", fontsize=16)
        destination = phase / "training_curves.png"
        fig.savefig(destination, dpi=140)
        plt.close(fig)
        written.append(destination)
        if phase1:
            diagnostics = ("same_position_std", "normalized_same_position_std",
                           "pooled_effective_rank", "normalized_pooled_effective_rank", "raw_rms")
            fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
            keys = list(dict.fromkeys(key for row in rows for key in row))
            for ax, metric in zip(axes.flat, diagnostics):
                for key in keys:
                    if not key.startswith("validation_") or not key.endswith("_" + metric):
                        continue
                    if "normalized" not in metric and "normalized" in key:
                        continue
                    points = [(row["successful_updates"], row[key]) for row in rows if key in row]
                    if points:
                        x, y = zip(*points)
                        label = key.removeprefix("validation_").removesuffix("_" + metric)
                        ax.plot(x, y, marker=".", label=label)
                ax.set(title=metric.replace("_", " "), xlabel="Successful optimizer update")
                ax.grid(alpha=0.25)
                if ax.lines:
                    ax.legend(fontsize=8, ncol=2)
            axes.flat[-1].axis("off")
            axes.flat[-1].text(0, 0.8, "Fixed validation bank; statistics across all bank samples.\n"
                              "Raw and LayerNorm-normalized features are reported separately.\n"
                              "Effective rank is limited by min(channels, bank size - 1).\n"
                              "Compare to step 0; gate PASS alone does not prove useful features.", fontsize=11)
            fig.suptitle(f"{phase.name} — {run_kind.upper()} — latent diversity and scale")
            destination = phase / "latent_diagnostics.png"
            fig.savefig(destination, dpi=140)
            plt.close(fig)
            written.append(destination)
    if not written:
        raise ValueError(f"No train.jsonl found under {root}")
    return written


def image_panel(path: Path, clean: np.ndarray, noisy: np.ndarray, restored: np.ndarray, title: str) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, data, label in zip(axes[:3], (clean, noisy, restored), ("Clean target", "Corrupted input", "Restored")):
        ax.imshow(np.clip(data, 0, 1))
        ax.set_title(label)
        ax.axis("off")
    error = np.abs(np.clip(restored, 0, 1) - clean).mean(axis=-1)
    im = axes[3].imshow(error, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Mean absolute RGB error [0,1]")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], shrink=0.7)
    fig.suptitle(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_imu_result(output: Path, trajectory: str, times: np.ndarray, clean: np.ndarray,
                    noisy: np.ndarray, restored: np.ndarray, coverage: np.ndarray,
                    *, make_plot: bool, label: str) -> str:
    """Preserve every row and a coverage mask; plots use at most 4000 rows."""
    plt = _pyplot()
    stem = hashlib.sha256(trajectory.encode()).hexdigest()[:16]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{stem}.npz", trajectory=trajectory, timestamp=times,
                        clean=clean, noisy=noisy, restored=restored, coverage=coverage)
    if make_plot and coverage.any():
        first, last = np.flatnonzero(coverage)[[0, -1]]
        index = np.unique(np.linspace(first, last, min(4000, last - first + 1), dtype=int))
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
        for channel, ax in enumerate(axes.T.flat):
            for values, name, alpha in ((clean, "Clean", 1), (noisy, "Corrupted", 0.5), (restored, "Restored", 0.9)):
                data = values[index, channel].copy()
                data[~coverage[index]] = np.nan
                ax.plot(times[index] - times[first], data, label=name, alpha=alpha, linewidth=0.8)
            ax.set(title=("ax", "ay", "az", "gx", "gy", "gz")[channel],
                   xlabel="Time from first covered row (s)", ylabel="m/s²" if channel < 3 else "rad/s")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{label} — {trajectory}\nMerged overlapping windows; plotted ≤4000 points")
        fig.savefig(output / f"{stem}.png", dpi=140)
        plt.close(fig)
    return stem


def evaluation_summary(output: Path, results: dict[str, dict[str, Any]], *, label: str = "Held-out evaluation") -> None:
    plt = _pyplot()
    write_json(output / "metrics.json", results)
    write_csv(output / "metrics.csv", [{"scenario": key, **value} for key, value in results.items()])
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    names = list(results)
    x = np.arange(len(names))
    metrics = (("image_psnr_db", "Image PSNR ↑ (dB)"), ("image_ssim", "Image SSIM ↑"),
               ("accel_rmse", "Acceleration RMSE ↓ (m/s²)"), ("gyro_rmse", "Gyro RMSE ↓ (rad/s)"))
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.bar(x - 0.2, [results[name][f"baseline_{key}"] for name in names], 0.4, label="Corrupted input")
        ax.bar(x + 0.2, [results[name][key] for name in names], 0.4, label="Restored")
        ax.set_xticks(x, names, rotation=55, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{label} — input baseline vs restored (identity PSNR capped at 120 dB)")
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)
