"""Show fresh paired camera/IMU restorations from an existing checkpoint.

This is a qualitative preview. The fixed validation audit remains unchanged so
its metrics can be compared across checkpoints and runs.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _system_from_phase2
from qjepa.data import read_manifest
from qjepa.evaluation.metrics import image_metrics, imu_metrics


CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


def choose_indices(
    sample_ids: list[str], eligible: list[int], count: int,
    previous_ids: set[str], rng: np.random.Generator,
) -> list[int]:
    """Draw without replacement and never repeat an ID from the previous run."""
    if count < 1 or not eligible:
        raise ValueError("Need a positive count and at least one eligible sample")
    fresh = [index for index in eligible if sample_ids[index] not in previous_ids]
    if not fresh:
        raise ValueError("No new paired samples remain; use a larger split or fewer panels")
    return rng.choice(fresh, size=min(count, len(fresh)), replace=False).tolist()


def _eligible_indices(dataset, image_mode: str) -> list[int]:
    """Frames where some optical degradation actually fired, so a panel shows one."""
    if image_mode not in ("blur_only", "blur_low_light"):
        return list(range(len(dataset)))
    eligible = []
    for index, sample in enumerate(dataset.samples):
        parameters = dataset.image_corruptor._parameters(
            sample.split, dataset.realization, sample.trajectory_key,
            sample.image_time, image_mode,
        )
        if parameters["defocus"] or parameters["motion"] or parameters["downsample"]:
            eligible.append(index)
    return eligible


def _plot_pair(path: Path, sample: dict, restored, image_input: dict,
               image_output: dict, imu_input: dict, imu_output: dict,
               realization: int) -> None:
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(2.3, 1, 1))
    images = (
        ("Ảnh sạch", sample["image_clean"]),
        ("Ảnh hư hại", sample["image_noisy"]),
        ("Ảnh khôi phục", restored.image[0]),
    )
    for column, (title, tensor) in enumerate(images):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")

    times = sample["imu_times"].numpy()
    times = times - times[0]
    traces = (
        ("Sạch", sample["imu_clean_phys"].numpy(), "#333333"),
        ("Hư hại", sample["imu_noisy_phys"].numpy(), "#e87531"),
        ("Khôi phục", restored.imu_physical[0].detach().cpu().T.numpy(), "#2476c4"),
    )
    for channel, label in enumerate(CHANNELS):
        ax = fig.add_subplot(grid[1 + channel // 3, channel % 3])
        for name, values, colour in traces:
            ax.plot(times, values[:, channel], label=name, color=colour, linewidth=1.1)
        ax.set_title(f"IMU {label}")
        ax.set_xlabel("Giây")
        ax.set_ylabel("m/s²" if channel < 3 else "rad/s")
        ax.grid(alpha=0.2)
        if channel == 0:
            ax.legend(loc="best", fontsize=8)

    image_flags = sample["corruption"]["image"]
    blur = "+".join(name for name in ("defocus", "motion", "downsample") if image_flags[name]) or "none"
    fig.suptitle(
        f"{sample['sample_id']} | realization={realization} | blur={blur}\n"
        f"Ảnh PSNR {image_input['image_psnr_db']:.2f} → {image_output['image_psnr_db']:.2f} dB"
        f"  |  IMU accel RMSE {imu_input['accel_rmse']:.3g} → {imu_output['accel_rmse']:.3g}"
        f"  |  gyro {imu_input['gyro_rmse']:.3g} → {imu_output['gyro_rmse']:.3g}",
        fontsize=11,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def preview(
    checkpoint: str | Path, manifest_path: str | Path, *, output: str | Path,
    state: str | Path, count: int = 4, split: str = "valid",
    image_mode: str = "blur_only", imu_mode: str = "full",
    device: str = "cuda", seed: int | None = None, light_scale: float = 1.0,
) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    state = Path(state)
    manifest = read_manifest(manifest_path)
    system, config = _system_from_phase2(str(checkpoint), torch.device(device))
    if light_scale != 1.0:
        # Brightening only the preview, never the checkpoint's own recipe: the
        # panels are for looking at, and metrics stay comparable only while the
        # corruption matches what the run was measured on.
        if light_scale <= 0:
            raise ValueError("light_scale must be positive")
        low, high = config["corruption"]["image"]["exposure_gain"]
        config["corruption"]["image"]["exposure_gain"] = [
            min(1.0, low * light_scale), min(1.0, high * light_scale)
        ]
        print(f"[preview] exposure_gain {low:.3f}..{high:.3f} -> "
              f"{config['corruption']['image']['exposure_gain'][0]:.3f}.."
              f"{config['corruption']['image']['exposure_gain'][1]:.3f} "
              f"(chỉ ảnh hưởng panel này, không đổi checkpoint)")
    previous = json.loads(state.read_text()) if state.is_file() else {}
    identity = (manifest["meta"]["manifest_hash"], split, image_mode, imu_mode)
    previous_ids = set(previous.get("sample_ids", [])) if tuple(previous.get("identity", ())) == identity else set()
    explicit_seed = seed is not None
    seed = int(seed) if explicit_seed else secrets.randbits(63)
    rng = np.random.default_rng(seed)
    realization = int(rng.integers(1, 2**31))
    if not explicit_seed and realization == previous.get("realization"):
        realization = 1 + realization % (2**31 - 1)
    dataset = _dataset(config, manifest, split, fixed_realization=True,
                       image_mode=image_mode, imu_mode=imu_mode)
    dataset.set_realization(realization)
    eligible = _eligible_indices(dataset, image_mode)
    indices = choose_indices(
        [sample.sample_id for sample in dataset.samples], eligible, count,
        set() if explicit_seed else previous_ids, rng,
    )
    output.mkdir(parents=True)
    items = []
    for position, index in enumerate(indices):
        sample = dataset[index]
        with torch.no_grad():
            restored = system(
                sample["image_noisy"].unsqueeze(0).to(device),
                sample["imu_noisy_phys"].T.unsqueeze(0).to(device),
                sample["image_time"].unsqueeze(0).to(device),
                sample["imu_times"].unsqueeze(0).to(device),
            )
            image_clean = sample["image_clean"].unsqueeze(0).to(device)
            imu_clean = sample["imu_clean_phys"].T.unsqueeze(0).to(device)
            imu_noisy = sample["imu_noisy_phys"].T.unsqueeze(0).to(device)
            image_input = image_metrics(sample["image_noisy"].unsqueeze(0).to(device), image_clean)
            image_output = image_metrics(restored.image, image_clean)
            imu_input = imu_metrics(imu_noisy, imu_clean)
            imu_output = imu_metrics(restored.imu_physical, imu_clean)
        panel_name = f"pair_{position:02d}.png"
        _plot_pair(output / panel_name, sample, restored, image_input,
                   image_output, imu_input, imu_output, realization)
        items.append({
            "sample_id": sample["sample_id"], "index": index, "panel": panel_name,
            "image_corruption": sample["corruption"]["image"],
            "imu_corruption": sample["corruption"]["imu"],
            "image_input_metrics": image_input, "image_restored_metrics": image_output,
            "imu_input_metrics": imu_input, "imu_restored_metrics": imu_output,
        })
        print(f"{position + 1}/{len(indices)} {sample['sample_id']} → {output / panel_name}", flush=True)

    report = {
        "checkpoint": str(checkpoint), "manifest_hash": identity[0],
        "split": split, "image_mode": image_mode, "imu_mode": imu_mode,
        "light_scale": light_scale,
        "seed": seed, "realization": realization, "items": items,
    }
    (output / "preview.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"identity": identity, "sample_ids":
                                  [item["sample_id"] for item in items],
                                  "realization": realization}, indent=2), encoding="utf-8")
    print(f"Đã lưu {len(items)} cặp ảnh + IMU | seed={seed} | realization={realization}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--image-mode",
                        choices=("full", "blur_only", "blur_low_light"),
                        default="blur_only")
    parser.add_argument("--light-scale", type=float, default=1.0,
                        help="nhân vào exposure_gain: >1 sáng hơn, 1.0 giữ nguyên recipe train")
    parser.add_argument("--imu-mode", choices=("full", "clean"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    preview(args.checkpoint, args.manifest, output=args.output, state=args.state,
            count=args.count, split=args.split, image_mode=args.image_mode,
            imu_mode=args.imu_mode, device=args.device, seed=args.seed,
            light_scale=args.light_scale)


if __name__ == "__main__":
    main()
