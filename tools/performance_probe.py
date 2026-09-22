"""Time the real phase-2 data loader and optimizer step without saving a checkpoint.

Run this in a separate process on Kaggle after training has stopped. The loaded
decoder is updated in memory for a few steps; the source checkpoint is untouched.
Use identical --workers and --updates with --gpus 1 and 2 to measure whether
DataParallel helps a four-image microbatch on the available hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _memory_mib, _system_from_phase2, _training_batch_stream
from qjepa.data import read_manifest
from qjepa.training.phase2 import Phase2Trainer


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_s": float(np.median(values)),
        "mean_s": float(np.mean(values)),
        "p90_s": float(np.percentile(values, 90)),
    }


def probe(checkpoint: Path, manifest_path: Path, *, gpus: int, workers: int,
          updates: int, warmup: int, output: Path | None = None) -> dict:
    if gpus not in (1, 2) or workers < 0 or updates < 1 or warmup < 0:
        raise ValueError("Use --gpus 1/2, --workers >= 0, --updates >= 1, --warmup >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("This training performance probe requires CUDA")
    device = torch.device("cuda")
    system, config = _system_from_phase2(str(checkpoint), device)
    config["data"] = dict(config["data"], num_workers=workers)
    config["runtime"] = dict(config["runtime"], gpu_count=gpus)
    manifest = read_manifest(manifest_path)
    dataset = _dataset(
        config, manifest, "train", fixed_realization=False,
        scenarios=config["phase2"].get("train_scenarios"),
    )
    stream = _training_batch_stream(
        config, dataset, config["phase2"]["batch_size"],
        start_microbatch=0, namespace="phase2",
    )
    trainer = Phase2Trainer(
        system, config, device,
        parent_checkpoint="performance_probe_only",
        manifest_hash=manifest["meta"]["manifest_hash"],
    )
    accumulation = config["phase2"]["gradient_accumulation"]
    waits: list[float] = []
    steps: list[float] = []
    for index in range(warmup + updates):
        start = time.perf_counter()
        batches = [next(stream) for _ in range(accumulation)]
        loaded = time.perf_counter()
        result = trainer.step(batches)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        if result.get("skipped"):
            raise RuntimeError(f"Optimizer step skipped during probe: {result}")
        if index >= warmup:
            waits.append(loaded - start)
            steps.append(finished - loaded)
    elapsed = sum(waits) + sum(steps)
    report = {
        "checkpoint": str(checkpoint),
        "gpus": gpus,
        "workers": workers,
        "pin_memory": bool(config["data"]["pin_memory"]),
        "batch_size_per_microbatch": config["phase2"]["batch_size"],
        "gradient_accumulation": accumulation,
        "warmup_updates": warmup,
        "measured_updates": updates,
        "data_wait": _summary(waits),
        "optimizer_step": _summary(steps),
        "wall_seconds_per_update": elapsed / updates,
        "data_wait_fraction": sum(waits) / elapsed,
        "memory_mib": _memory_mib(),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpus", required=True, type=int, choices=(1, 2))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    probe(args.checkpoint, args.manifest, gpus=args.gpus, workers=args.workers,
          updates=args.updates, warmup=args.warmup, output=args.output)


if __name__ == "__main__":
    main()
