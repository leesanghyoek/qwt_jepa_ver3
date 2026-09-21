"""Profile which truly blurred validation frames improve or regress.

This is a read-only checkpoint diagnostic. Input blur severity is measured
against the clean target only to understand failures; it is not an inference
time signal or a proposed gating rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _loader, _system_from_phase2, _to_device
from qjepa.data import read_manifest
from tools.image_blur_audit import _spread_indices, edge_components


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No truly blurred frames")
    count = sum(row["edge_count"] for row in rows)
    return {
        "frames": len(rows),
        "input_mae": float(np.mean([row["input_mae"] for row in rows])),
        "restored_mae": float(np.mean([row["restored_mae"] for row in rows])),
        "fraction_mae_better": sum(row["restored_mae"] < row["input_mae"] for row in rows) / len(rows),
        "input_edge_error": sum(row["input_edge_error_sum"] for row in rows) / count,
        "restored_edge_error": sum(row["restored_edge_error_sum"] for row in rows) / count,
        "fraction_edge_better": sum(
            row["restored_edge_error_sum"] < row["input_edge_error_sum"] for row in rows
        ) / len(rows),
    }


@torch.no_grad()
def profile(checkpoint: str | Path, manifest_path: str | Path, *, output: str | Path,
            split: str = "valid", samples: int = 256, device: str = "cuda") -> dict:
    if samples < 4:
        raise ValueError("--samples must be at least 4 for quartiles")
    device_obj = torch.device(device)
    system, config = _system_from_phase2(str(checkpoint), device_obj)
    manifest = read_manifest(manifest_path)
    indices = _spread_indices(len(manifest["samples"][split]), samples)
    config["data"] = dict(config["data"], num_workers=0, pin_memory=False)
    dataset = _dataset(config, manifest, split, fixed_realization=True,
                       image_mode="blur_only", imu_mode="clean")
    loader = _loader(config, Subset(dataset, indices), config["phase2"]["batch_size"], train=False)
    rows: list[dict] = []
    for raw in loader:
        batch = _to_device(raw, device_obj)
        restored = system(
            batch["image_noisy"], batch["imu_noisy_phys"],
            batch["image_time"], batch["imu_times"],
        ).image.clamp(0, 1)
        for index, corruption in enumerate(raw["corruption"]):
            parameters = corruption["image"]
            if not any(parameters[key] for key in ("defocus", "motion", "downsample")):
                continue
            clean = batch["image_clean"][index:index + 1].clamp(0, 1)
            noisy = batch["image_noisy"][index:index + 1].clamp(0, 1)
            output_image = restored[index:index + 1]
            edges = edge_components(clean, noisy, output_image)
            rows.append({
                "sample_id": raw["sample_id"][index],
                "input_mae": float((noisy - clean).abs().mean()),
                "restored_mae": float((output_image - clean).abs().mean()),
                "edge_count": edges["edge_count"],
                "input_edge_error_sum": edges["edge_input_error_sum"],
                "restored_edge_error_sum": edges["edge_restored_error_sum"],
                "defocus": bool(parameters["defocus"]),
                "motion": bool(parameters["motion"]),
                "downsample": bool(parameters["downsample"]),
                "defocus_sigma": float(parameters["defocus_sigma"]),
                "motion_length": int(parameters["motion_length"]),
                "downsample_scale": float(parameters["downsample_scale"]),
            })
    rows.sort(key=lambda row: (row["input_mae"], row["sample_id"]))
    if len(rows) < 4:
        raise ValueError("Fewer than four frames have actual blur")
    quartiles = {}
    for index, selected in enumerate(np.array_split(np.arange(len(rows)), 4), start=1):
        group = [rows[int(item)] for item in selected]
        quartiles[f"q{index}"] = {
            "input_mae_range": [group[0]["input_mae"], group[-1]["input_mae"]],
            **_summarize(group),
        }
    report = {
        "checkpoint": str(checkpoint), "split": split,
        "spread_samples": len(indices), "actual_blur_frames": len(rows),
        "overall": _summarize(rows), "input_mae_quartiles": quartiles,
        "frames": rows,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Actual blur: {len(rows)}/{len(indices)} | saved: {path}")
    for name, result in (("all", report["overall"]), *quartiles.items()):
        print(
            f"{name:>3} n={result['frames']:>3}"
            f" | MAE {result['input_mae']:.5f} -> {result['restored_mae']:.5f}"
            f" | better {result['fraction_mae_better']:.1%}"
            f" | edge {result['input_edge_error']:.5f} -> {result['restored_edge_error']:.5f}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    profile(args.checkpoint, args.manifest, output=args.output,
            split=args.split, samples=args.samples, device=args.device)


if __name__ == "__main__":
    main()
