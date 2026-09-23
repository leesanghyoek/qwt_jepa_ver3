"""Walk one random sample through the corruption pipeline, stage by stage.

Each run picks a different frame unless ``--seed`` pins one, and shows what
happens to it in the order the pipeline actually applies things:

    ảnh sạch  ->  + mờ do chuyển động (tích phân gyro SẠCH)  ->  + nhiễu
    IMU sạch  ->  + nhiễu

The middle image is the one worth looking at. It is the clean frame with
*nothing* applied except the blur the camera's own motion produced during the
exposure, so any softening you see there is the gyro's doing and nothing else.
The third image then adds what a real sensor adds: defocus, resolution loss, low
light, shot and read noise, quantisation, JPEG.

The two stages share one parameter draw, so the middle image is a genuine prefix
of the right-hand one rather than a separate render that happens to look similar.

    python3 tools/corruption_stages_demo.py --data-root <root>

Torch is not imported; numpy, scipy, Pillow and matplotlib are enough.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported before pyplot on purpose: this module picks the backend, and
# matplotlib.use() only takes effect before pyplot is loaded.
from imu_blur_preview import (
    AXIS_COLORS,
    CAN_SHOW,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    SURFACE,
    _style,
    find_trajectories,
    load_rgb,
    resolve_data_root,
    show_or_close,
    load_trajectory,
    pick_window,
)

import matplotlib.pyplot as plt

from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor
from qjepa.corruptions.imu import ImuCorruptionConfig, TrajectoryImuCorruptor
from qjepa.corruptions.motion import exposure_path
from qjepa.corruptions.rng import generator


def psnr(reference: np.ndarray, other: np.ndarray) -> float:
    error = float(np.mean((reference.astype(np.float64) - other.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(error, 1e-12))


def sharpness(image: np.ndarray) -> float:
    """Mean absolute gradient of luma: falls as the frame softens."""
    grey = image.astype(np.float64) @ (0.299, 0.587, 0.114)
    return float(np.abs(np.diff(grey, axis=1)).mean() + np.abs(np.diff(grey, axis=0)).mean())


def draw_stage(axis, image, index, title, caption):
    axis.imshow(np.clip(image, 0, 1))
    axis.set_xticks([]); axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color(GRID)
    axis.set_title(f"{index}  {title}", color=INK, fontsize=11, loc="left", pad=6)
    axis.set_xlabel(caption, color=INK_SOFT, fontsize=8.5, linespacing=1.5)


def draw_trace(axis, time, values, labels, *, ghost=None, title, ylabel, limits, capture):
    _style(axis)
    if ghost is not None:
        for index in range(values.shape[1]):
            axis.plot(time, ghost[:, index], color=AXIS_COLORS[index], linewidth=2.6,
                      alpha=0.22, zorder=2, solid_capstyle="round")
    for index, label in enumerate(labels):
        axis.plot(time, values[:, index], color=AXIS_COLORS[index], linewidth=1.5,
                  zorder=3, solid_capstyle="round", label=label)
        axis.annotate(label, xy=(time[-1], values[-1, index]), xytext=(4, 0),
                      textcoords="offset points", color=INK_SOFT, fontsize=8, va="center")
    axis.axvline(capture, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    axis.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
    axis.set_ylabel(ylabel, color=INK_SOFT, fontsize=8)
    axis.set_xlabel("giây", color=INK_SOFT, fontsize=8)
    axis.set_ylim(*limits)
    axis.margins(x=0.02)


def build_figure(sample, destination: Path) -> None:
    figure = plt.figure(figsize=(16.0, 8.8), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.86], hspace=0.36,
                               wspace=0.20, left=0.045, right=0.97, top=0.865, bottom=0.085)

    report = sample["report"]
    draw_stage(figure.add_subplot(grid[0, 0]), sample["clean"], "①", "ẢNH SẠCH",
               f"nét: {sample['sharp_clean']:.4f}\ntham chiếu, chỉ tồn tại lúc train")
    draw_stage(
        figure.add_subplot(grid[0, 1]), sample["blurred"], "②", "+ MỜ DO IMU",
        f"quét {report['path_span_px']:.2f} px · phơi sáng "
        f"{report['exposure_seconds'] * 1000:.1f} ms\n"
        f"nét: {sample['sharp_blur']:.4f}  ({sample['drop_blur']:+.1f}%)  ·  "
        f"PSNR {sample['psnr_blur']:.2f} dB",
    )
    draw_stage(
        figure.add_subplot(grid[0, 2]), sample["noisy"], "③", "+ NHIỄU (ảnh cuối)",
        f"thiếu sáng ×{report['exposure_gain']:.2f} · gamma {report['tone_gamma']:.2f} · "
        f"{report['quantization_bits']} bit\n"
        f"nét: {sample['sharp_noisy']:.4f}  ({sample['drop_noisy']:+.1f}%)  ·  "
        f"PSNR {sample['psnr_noisy']:.2f} dB",
    )

    time = sample["time"] - sample["time"][0]
    capture = float(sample["capture"] - sample["time"][0])
    gyro_clean, gyro_noisy = sample["imu_clean"][:, 3:6], sample["imu_noisy"][:, 3:6]
    limits = (min(gyro_clean.min(), gyro_noisy.min()) - 0.15,
              max(gyro_clean.max(), gyro_noisy.max()) + 0.15)

    axis_clean = figure.add_subplot(grid[1, 0])
    draw_trace(axis_clean, time, gyro_clean, ("gx", "gy", "gz"), capture=capture,
               title="① IMU SẠCH — vận tốc góc (thứ sinh ra vệt mờ ở ②)",
               ylabel="rad/s", limits=limits)
    axis_clean.legend(loc="upper left", fontsize=8, frameon=False,
                      labelcolor=INK_SOFT, ncol=3, columnspacing=1.1)
    draw_trace(figure.add_subplot(grid[1, 1]), time, gyro_noisy, ("gx", "gy", "gz"),
               ghost=gyro_clean, capture=capture,
               title="③ IMU + NHIỄU — đây mới là thứ model đọc",
               ylabel="rad/s", limits=limits)

    axis_kernel = figure.add_subplot(grid[1, 2])
    _style(axis_kernel)
    axis_kernel.grid(False)
    axis_kernel.imshow(sample["kernel"], cmap="Blues", interpolation="nearest")
    axis_kernel.set_xticks([]); axis_kernel.set_yticks([])
    axis_kernel.set_title("② Kernel mờ — tích phân gyro SẠCH",
                          color=INK, fontsize=10, loc="left", pad=6)
    axis_kernel.set_xlabel(
        f"roll {np.degrees(report['roll_radians']):+.2f}° (không nằm trong kernel)\n"
        "blur do gyro SẠCH; model chỉ được thấy bản nhiễu ở ③",
        color=INK_SOFT, fontsize=8, linespacing=1.5)

    figure.suptitle(
        f"{sample['name']}  ·  frame {sample['frame']}  ·  seed {sample['seed']}\n"
        "ảnh sạch  →  + mờ do IMU  →  + nhiễu        (IMU: sạch  →  + nhiễu)",
        color=INK, fontsize=12.5, x=0.045, ha="left", y=0.975)
    figure.savefig(destination, dpi=125, facecolor=SURFACE)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="mặc định: tự dò trong ~/Datasets và /kaggle/input")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent.parent / "configs/pipeline_v3.yaml")
    # Relative to the repository, not the shell's working directory: run from an
    # editor or another folder and a relative default silently scatters output
    # wherever that happened to be.
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs/corruption_stages")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,
                        help="mặc định: ngẫu nhiên mỗi lần chạy; đặt để lặp lại đúng mẫu cũ")
    parser.add_argument("--imu-window", type=int, default=128)
    parser.add_argument("--min-span-px", type=float, default=2.0,
                        help="chỉ nhận frame mà gyro quét được ít nhất bằng này")
    parser.add_argument("--no-show", action="store_true",
                        help="chỉ ghi file, không mở cửa sổ")
    parser.add_argument("--attempts", type=int, default=60,
                        help="số lần bốc frame trước khi chấp nhận frame ít chuyển động nhất")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = LowLightImageCorruptionConfig(**raw["corruption"]["image"])
    # Stage 2 must be blur and nothing else, so the softening there is
    # attributable to the gyro alone. Only the probabilities differ, and
    # `_parameters` draws in a fixed order regardless of their values, so both
    # corruptors see the identical parameter stream -- the exposure that shapes
    # the kernel in stage 2 is the same one stage 3 was built from.
    motion_only = LowLightImageCorruptor(
        replace(base, clean_probability=0.0, defocus_probability=0.0,
                downsample_probability=0.0),
        master_seed=raw["data"]["corruption_seed"],
    )
    full = LowLightImageCorruptor(replace(base, clean_probability=0.0),
                                  master_seed=raw["data"]["corruption_seed"])
    imu_corruptor = TrajectoryImuCorruptor(
        ImuCorruptionConfig(**raw["corruption"]["imu"]), master_seed=raw["data"]["corruption_seed"]
    )

    args.data_root = resolve_data_root(args.data_root)
    trajectories = find_trajectories(args.data_root)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = generator(seed, "corruption_stages")
    figures = []
    print(f"{len(trajectories)} trajectory · seed {seed} "
          f"(chạy lại đúng mẫu này bằng --seed {seed})\n")

    for number in range(args.samples):
        path = trajectories[int(rng.integers(len(trajectories)))]
        imu, imu_time, cam_time, images = load_trajectory(path)
        name = "/".join(path.parts[-3:])
        usable = min(len(cam_time), len(images))

        # Sample frames until one is moving enough to be worth looking at,
        # instead of scanning the whole trajectory: a still frame produces a 3x3
        # kernel and stage 2 would look identical to stage 1.
        best = None
        for _ in range(args.attempts):
            frame = int(rng.integers(1, max(2, usable - 1)))
            start, end = pick_window(imu_time, float(cam_time[frame]), args.imu_window)
            parameters = motion_only._parameters("demo", 0, name, float(cam_time[frame]), "blur_only")
            u, v, _ = exposure_path(
                imu[start:end, 3:6], imu_time[start:end], float(cam_time[frame]),
                float(parameters["exposure_seconds"]),
                focal_length_px=base.focal_length_px, angular_gain=base.angular_gain,
                samples=base.motion_path_samples)
            span = float(np.hypot(u.max() - u.min(), v.max() - v.min()))
            if best is None or span > best[1]:
                best = (frame, span, start, end)
            if span >= args.min_span_px:
                break
        frame, span, start, end = best

        clean = load_rgb(images[frame])
        window, window_time = imu[start:end], imu_time[start:end]
        shared = dict(split="demo", realization=0, trajectory=name,
                      timestamp=float(cam_time[frame]), frame_index=frame,
                      gyro=window[:, 3:6], imu_times=window_time)
        blurred, report = motion_only(clean, mode="blur_only", **shared)
        noisy, _ = full(clean, mode="full", **shared)
        imu_noisy, _ = imu_corruptor.window(imu, imu_time, start, end, split="demo",
                                            realization=0, trajectory=name, mode="full")

        from qjepa.corruptions.motion import imu_blur_kernel
        kernel, _ = imu_blur_kernel(
            window[:, 3:6], window_time, float(cam_time[frame]),
            float(report["exposure_seconds"]), focal_length_px=base.focal_length_px,
            angular_gain=base.angular_gain, samples=base.motion_path_samples,
            max_radius_px=base.motion_max_radius_px)

        sharp_clean = sharpness(clean)
        sharp_blur, sharp_noisy = sharpness(blurred), sharpness(noisy)
        destination = args.out / f"{number:02d}_{name.replace('/', '_')}_f{frame:06d}.png"
        figures.append(build_figure({
            "name": name, "frame": frame, "seed": seed,
            "clean": clean, "blurred": blurred, "noisy": noisy, "kernel": kernel,
            "report": report, "imu_clean": window.astype(np.float64),
            "imu_noisy": imu_noisy.astype(np.float64), "time": window_time,
            "capture": float(cam_time[frame]),
            "sharp_clean": sharp_clean, "sharp_blur": sharp_blur, "sharp_noisy": sharp_noisy,
            "drop_blur": 100.0 * (sharp_blur / sharp_clean - 1.0),
            "drop_noisy": 100.0 * (sharp_noisy / sharp_clean - 1.0),
            "psnr_blur": psnr(clean, blurred), "psnr_noisy": psnr(clean, noisy),
        }, destination))

        print(f"{name} · frame {frame}")
        print(f"   ② mờ do IMU : quét {span:5.2f} px · phơi sáng "
              f"{report['exposure_seconds'] * 1000:4.1f} ms · "
              f"nét {100.0 * (sharp_blur / sharp_clean - 1.0):+6.1f}% · "
              f"PSNR {psnr(clean, blurred):5.2f} dB")
        print(f"   ③ + nhiễu   : thiếu sáng x{report['exposure_gain']:.2f} · "
              f"{report['quantization_bits']} bit · "
              f"nét {100.0 * (sharp_noisy / sharp_clean - 1.0):+6.1f}% · "
              f"PSNR {psnr(clean, noisy):5.2f} dB")
        print(f"   -> {destination}\n")
    show_or_close(figures, args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
