"""Do truc tiep Delta = C_out - C_in cua decoder residual, tach theo bang.

Metric tong hop chi noi anh ra "tot hon" hay "te hon"; no khong noi decoder dang
LAM GI. Voi C_out = C_in + Delta, cau hoi that su la Delta co dau nao tren bang
chi tiet: cong them duong net, hay tru bot di.

Tru bot la nghiem ma L1 thuong. O mot vi tri co canh, xoa phan tan so cao giam
sai so chac chan, con doan them canh thi rui ro — nen trung vi co dieu kien chon
lam phang. Khi do anh ra MO HON chinh anh vao, du moi metric deu thang baseline.

Chay bang subprocess nen luon nap code moi nhat tu dia.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _loader, _system_from_phase2, _to_device
from qjepa.data import read_manifest

BANDS = ("LL  (do sang, bo cuc)", "LH  (net ngang)", "HL  (net doc)", "HH  (net cheo)")


def rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = read_manifest(args.manifest)
    system, config = _system_from_phase2(args.checkpoint, device)
    if not config["phase2"].get("input_coefficient_residual", False):
        raise SystemExit("Checkpoint nay khong dung residual; Delta khong dinh nghia duoc.")
    print("phase2:",
          "beta", config["phase2"]["smooth_l1_beta"],
          "| detail", config["phase2"]["reconstruction_detail_weight"],
          "| variation", config["phase2"].get("imu_variation_weight"))

    dataset = _dataset(config, manifest, args.split, fixed_realization=True)
    loader = _loader(config, dataset, config["phase2"]["batch_size"], train=False)

    # Tong binh phuong tich luy, chia ra o cuoi — tranh giu tensor cua moi batch.
    image_in = np.zeros(4)
    image_delta = np.zeros(4)
    image_gap_in = np.zeros(4)      # |C_in - C_clean| : sai so cua viec khong lam gi
    image_gap_out = np.zeros(4)     # |C_out - C_clean| : sai so sau khi sua
    imu_in = imu_delta = imu_gap_in = imu_gap_out = 0.0
    seen = 0

    for index, raw in enumerate(loader):
        if index >= args.batches:
            break
        batch = _to_device(raw, device)
        with torch.no_grad():
            latent = system.encode(batch["image_noisy"], batch["imu_noisy_phys"],
                                   batch["image_time"], batch["imu_times"])
            restored = system.decode(latent)
            clean_image, _ = system.backbone.image_transform.analysis(batch["image_clean"])
            clean_imu_signal = system.normalizer.normalize(batch["imu_clean_phys"])
            clean_imu, _ = system.backbone.imu_transform.analysis(clean_imu_signal)

        c_in, c_out = latent.image_coefficients, restored.image_coefficients
        shape = (c_in.shape[0], c_in.shape[1] // 16, 4, 4) + c_in.shape[2:]
        for band in range(4):
            a = c_in.reshape(shape)[:, :, band]
            b = c_out.reshape(shape)[:, :, band]
            t = clean_image.reshape(shape)[:, :, band]
            image_in[band] += rms(a) ** 2
            image_delta[band] += rms(b - a) ** 2
            image_gap_in[band] += rms(a - t) ** 2
            image_gap_out[band] += rms(b - t) ** 2

        half = latent.imu_coefficients.shape[1] // 2
        u_in = latent.imu_coefficients[:, half:]
        u_out = restored.imu_coefficients[:, half:]
        u_target = clean_imu[:, half:]
        imu_in += rms(u_in) ** 2
        imu_delta += rms(u_out - u_in) ** 2
        imu_gap_in += rms(u_in - u_target) ** 2
        imu_gap_out += rms(u_out - u_target) ** 2
        seen += 1

    if not seen:
        raise SystemExit("Loader khong tra ve batch nao.")
    root = lambda total: np.sqrt(np.asarray(total) / seen)
    image_in, image_delta = root(image_in), root(image_delta)
    image_gap_in, image_gap_out = root(image_gap_in), root(image_gap_out)

    print(f"\nmau: {seen} batch x {config['phase2']['batch_size']}\n")
    print("ANH — he so QWT theo bang")
    print(f"  {'bang':<24}{'RMS(C_in)':>11}{'RMS(delta)':>12}{'delta/in':>10}"
          f"{'|C_in-sach|':>13}{'|C_out-sach|':>14}{'':>6}")
    for band in range(4):
        share = image_delta[band] / max(image_in[band], 1e-12)
        better = image_gap_out[band] < image_gap_in[band]
        print(f"  {BANDS[band]:<24}{image_in[band]:>11.5f}{image_delta[band]:>12.5f}"
              f"{100 * share:>9.1f}%{image_gap_in[band]:>13.5f}{image_gap_out[band]:>14.5f}"
              f"{'   tot hon' if better else '   TE HON'}")

    detail_out_gap = np.sqrt(np.square(image_gap_out[1:]).mean())
    detail_in_gap = np.sqrt(np.square(image_gap_in[1:]).mean())
    print(f"\n  Bang chi tiet gop lai: |C_in-sach| {detail_in_gap:.5f}"
          f" -> |C_out-sach| {detail_out_gap:.5f}"
          f"  {'tot hon' if detail_out_gap < detail_in_gap else 'TE HON'}")

    imu_in, imu_delta = np.sqrt(imu_in / seen), np.sqrt(imu_delta / seen)
    imu_gap_in, imu_gap_out = np.sqrt(imu_gap_in / seen), np.sqrt(imu_gap_out / seen)
    print("\nIMU — nua bang detail cua Haar")
    print(f"  RMS(C_in) {imu_in:.5f} | RMS(delta) {imu_delta:.5f}"
          f" | delta/in {100 * imu_delta / max(imu_in, 1e-12):.1f}%")
    print(f"  |C_in-sach| {imu_gap_in:.5f} -> |C_out-sach| {imu_gap_out:.5f}"
          f"  {'tot hon' if imu_gap_out < imu_gap_in else 'TE HON'}")

    print("\nDoc ket qua:")
    print("  delta/in gan 0%      -> decoder khong lam gi, dau ra = dau vao.")
    print("  |C_out-sach| TE HON  -> decoder dang lam hong chinh bang do.")
    print("  Bang chi tiet TE HON nhung LL tot hon -> chinh phoi sang, xoa duong net.")


if __name__ == "__main__":
    main()
