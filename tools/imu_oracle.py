"""Hoi thang loss: dau ra co dinh nao duoc cham diem tot nhat?

Decoder IMU hoi tu ve delta ~ 0 qua moi cau hinh da thu, tuc dau ra = dau vao =
tin hieu nhieu. Co hai kha nang, va chung doi hai cach xu ly nguoc nhau:

  loss cham delta=0 la tot nhat  -> LOSS SAI. Doi kien truc khong cuu duoc.
  loss cham cai khac tot hon     -> VAN DE TOI UU HOA. Loss on, model khong tim ra.

Khong can train gi: cham diem vai dau ra co dinh — giu nguyen input, loc thong
thap o vai muc, va oracle (chinh tin hieu sach). Neu mot bo loc tam thuong danh
bai "giu nguyen" theo chinh loss dang dung, thi model dang bo lo mot cai loi de
lay, va loi thuoc ve toi uu hoa chu khong phai ham muc tieu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _loader
from qjepa.config import build_backbone, build_normalizer, load_config
from qjepa.data import read_manifest
from qjepa.training.losses import phase2_reconstruction_loss


def low_pass(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """Loc Gauss doc truc thoi gian — bo loc khu nhieu don gian nhat co the."""
    if sigma <= 0:
        return signal
    radius = max(1, int(3 * sigma))
    positions = torch.arange(-radius, radius + 1, dtype=signal.dtype, device=signal.device)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1).expand(signal.shape[1], 1, -1)
    padded = F.pad(signal, (radius, radius), mode="replicate")
    return F.conv1d(padded, kernel, groups=signal.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_v3.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config(args.config)
    manifest = read_manifest(args.manifest)
    backbone = build_backbone(config).to(device).eval()
    normalizer = build_normalizer(manifest["meta"]).to(device).eval()
    phase2 = config["phase2"]
    print(f"loss dang dung: beta {phase2['smooth_l1_beta']}"
          f" | detail {phase2['reconstruction_detail_weight']}"
          f" | variation {phase2.get('imu_variation_weight')}")

    dataset = _dataset(config, manifest, args.split, fixed_realization=True)
    loader = _loader(config, dataset, phase2["batch_size"], train=False)

    sigmas = [0.0, 0.5, 1.0, 2.0, 4.0]
    labels = [f"loc thong thap sigma={s}" if s else "giu nguyen input (delta=0)" for s in sigmas]
    labels.append("oracle: tin hieu SACH")
    totals = np.zeros(len(labels))
    rmse = np.zeros(len(labels))
    variation = np.zeros(len(labels))
    seen = 0

    for index, raw in enumerate(loader):
        if index >= args.batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
        with torch.no_grad():
            noisy = normalizer.normalize(batch["imu_noisy_phys"])
            clean = normalizer.normalize(batch["imu_clean_phys"])
            image = batch["image_clean"]
            image_coeff, _ = backbone.image_transform.analysis(image)
            clean_imu_coeff, _ = backbone.imu_transform.analysis(clean)
            candidates = [low_pass(noisy, s) for s in sigmas] + [clean]
            for slot, candidate in enumerate(candidates):
                coeff, _ = backbone.imu_transform.analysis(candidate)
                loss, _ = phase2_reconstruction_loss(
                    image, image, candidate, clean,
                    beta=phase2["smooth_l1_beta"],
                    image_coefficients=image_coeff, image_coefficient_target=image_coeff,
                    imu_coefficients=coeff, imu_coefficient_target=clean_imu_coeff,
                    detail_weight=phase2["reconstruction_detail_weight"],
                    variation_weight=float(phase2.get("imu_variation_weight", 0.0)),
                )
                totals[slot] += float(loss)
                rmse[slot] += float((candidate - clean).square().mean().sqrt())
                step = candidate.diff(dim=-1) - clean.diff(dim=-1)
                variation[slot] += float(step.square().mean().sqrt())
        seen += 1

    if not seen:
        raise SystemExit("Loader khong tra ve batch nao.")
    totals, rmse, variation = totals / seen, rmse / seen, variation / seen
    baseline = totals[0]

    print(f"\nmau: {seen} batch x {phase2['batch_size']}")
    # Phan anh bi triet tieu: ta truyen image_clean lam ca du doan lan dich, nen
    # cot loss duoi day chi con cac so hang IMU.
    print(f"\n{'dau ra co dinh':<30}{'loss IMU':>10}{'so voi giu nguyen':>20}"
          f"{'RMSE chuan hoa':>17}{'sai phan':>11}")
    for slot, label in enumerate(labels):
        delta = 100.0 * (totals[slot] - baseline) / max(abs(baseline), 1e-12)
        mark = "" if slot == 0 else ("  TOT HON" if totals[slot] < baseline else "")
        print(f"  {label:<28}{totals[slot]:>10.5f}{delta:>19.2f}%"
              f"{rmse[slot]:>17.5f}{variation[slot]:>11.5f}{mark}")

    best = int(np.argmin(totals[:-1]))
    gain = 100.0 * (baseline - totals[best]) / max(abs(baseline), 1e-12)
    print("\nDoc ket qua:")
    if best == 0:
        print("  'Giu nguyen input' la tot nhat trong cac phuong an khong can hoc.")
        print("  -> LOSS dang THUONG viec khong lam gi. Doi kien truc se khong cuu duoc;")
        print("     phai doi ham muc tieu.")
    elif gain < 1.0:
        print(f"  '{labels[best]}' chi hon 'giu nguyen' {gain:.2f}% — coi nhu hoa.")
        print("  -> Loss gan nhu KHONG PHAN BIET giua khu nhieu va khong lam gi, du sai")
        print("     phan giam manh. Gradient day ve phia mut qua yeu de thang nhieu cua")
        print("     SGD, nen delta~0 la diem dung tu nhien. Phai tang trong so cac so")
        print("     hang tan so cao, hoac doi ham muc tieu.")
    else:
        print(f"  '{labels[best]}' danh bai 'giu nguyen' {gain:.1f}%")
        print("  -> Loss KHONG thuong viec khong lam gi: co mot cai loi that de lay ma")
        print("     model khong tim ra. Loi thuoc ve toi uu hoa (learning rate, dung")
        print("     luong decoder, so update), khong phai ham muc tieu.")


if __name__ == "__main__":
    main()
