"""Do luong thong tin con lai trong latent bang hoi quy tuyen tinh.

Decoder tuyen tinh la decoder yeu nhat co the. Neu no rut duoc nhieu thong tin
thi thong tin nam san trong latent va loi thuoc ve decoder phase 2; neu no cung
that bai thi latent khong mang thong tin va loi thuoc ve phase 1.

Chay bang subprocess nen luon nap code moi nhat tu dia, khong dinh sys.modules.
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


def ridge_probe(features: np.ndarray, targets: np.ndarray, ratio: float = 0.7) -> tuple[float, float]:
    """MAE tren held-out cua ridge tot nhat, kem moc 'doan trung binh'."""
    split = int(len(features) * ratio)
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = targets[:split], targets[split:]
    mean = train_x.mean(0, keepdims=True)
    deviation = train_x.std(0, keepdims=True) + 1e-8
    train_x = (train_x - mean) / deviation
    test_x = (test_x - mean) / deviation
    train_x = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    test_x = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    gram = train_x.T @ train_x
    right = train_x.T @ train_y
    eye = np.eye(gram.shape[0])
    best = np.inf
    for penalty in (1e1, 1e2, 1e3, 1e4, 1e5):
        weights = np.linalg.solve(gram + penalty * eye, right)
        best = min(best, float(np.abs(test_x @ weights - test_y).mean()))
    baseline = float(np.abs(train_y.mean(0, keepdims=True) - test_y).mean())
    return best, baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--cells-per-image", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    system, config = _system_from_phase2(args.checkpoint, device)
    manifest = read_manifest(args.manifest)
    dataset = _dataset(config, manifest, args.split, fixed_realization=True,
                       image_mode="clean", imu_mode="clean")
    loader = _loader(config, dataset, config["phase2"]["batch_size"], train=False)

    rng = np.random.default_rng(0)
    imu_features, imu_targets, cells, patches = [], [], [], []
    bin_features, bin_targets = [], []
    decoded_imu_error = decoded_image_error = 0.0
    decoded_imu_count = decoded_image_count = 0
    seen = 0
    for raw in loader:
        if seen >= args.samples:
            break
        batch = _to_device(raw, device)
        with torch.no_grad():
            latent = system.encode(batch["image_noisy"], batch["imu_noisy_phys"],
                                   batch["image_time"], batch["imu_times"])
            imu_clean = system.normalizer.normalize(batch["imu_clean_phys"])
            restored = system.decode(latent)
        # Decoder that da train, do tren cung mau -> so sanh thang voi probe.
        decoded_imu_error += float((restored.imu_normalized - imu_clean).abs().sum())
        decoded_imu_count += imu_clean.numel()

        imu_features.append(latent.ZU.flatten(1).cpu().numpy())
        imu_targets.append(imu_clean.flatten(1).cpu().numpy())

        # Them phep do theo tung bin thoi gian: 128 chieu thay vi 1024, nen so
        # hang gap 8 lan va ridge khong con chay o vung thieu du lieu.
        bins = latent.ZU.shape[2]
        span = imu_clean.shape[2] // bins
        bin_features.append(latent.ZU.permute(0, 2, 1).reshape(-1, latent.ZU.shape[1]).cpu().numpy())
        sliced = imu_clean[:, :, : bins * span].reshape(imu_clean.shape[0], imu_clean.shape[1], bins, span)
        bin_targets.append(sliced.permute(0, 2, 1, 3).reshape(-1, imu_clean.shape[1] * span).cpu().numpy())

        # Chi giu vai o moi anh: ca anh se ton ~3 GB RAM cho 4000 mau.
        image = batch["image_clean"]
        count, channels, height, width = image.shape
        rows, columns = latent.ZI.shape[2], latent.ZI.shape[3]
        tile_h, tile_w = height // rows, width // columns
        tiles = image.unfold(2, tile_h, tile_h).unfold(3, tile_w, tile_w)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(count, rows * columns, -1)
        grid = latent.ZI.permute(0, 2, 3, 1).reshape(count, rows * columns, -1)
        picked = rng.choice(rows * columns, size=min(args.cells_per_image, rows * columns), replace=False)
        cells.append(grid[:, picked].reshape(-1, grid.shape[-1]).cpu().numpy())
        patches.append(tiles[:, picked].reshape(-1, tiles.shape[-1]).cpu().numpy())

        out_tiles = restored.image.clamp(0.0, 1.0).unfold(2, tile_h, tile_h).unfold(3, tile_w, tile_w)
        out_tiles = out_tiles.permute(0, 2, 3, 1, 4, 5).reshape(count, rows * columns, -1)
        decoded_image_error += float((out_tiles[:, picked] - tiles[:, picked]).abs().sum())
        decoded_image_count += out_tiles[:, picked].numel()
        seen += count

    imu_features = np.concatenate(imu_features).astype(np.float64)
    imu_targets = np.concatenate(imu_targets).astype(np.float64)
    bin_features = np.concatenate(bin_features).astype(np.float64)
    bin_targets = np.concatenate(bin_targets).astype(np.float64)
    cells = np.concatenate(cells).astype(np.float64)
    patches = np.concatenate(patches).astype(np.float64)
    print(f"mau = {seen} | ZU -> {imu_features.shape[1]}d | o anh -> {cells.shape[1]}d"
          f" -> mang {patches.shape[1]}d | hang anh = {len(cells)}")

    imu_baseline = float(np.abs(imu_targets - imu_targets.mean(0, keepdims=True)).mean())
    image_baseline = float(np.abs(patches - patches.mean(0, keepdims=True)).mean())
    for label, features, targets in (("IMU ca cua so", imu_features, imu_targets),
                                     ("IMU theo bin ", bin_features, bin_targets),
                                     ("ANH theo o   ", cells, patches)):
        if len(features) <= features.shape[1] * 2:
            print(f"{label} : chi {len(features)} hang cho {features.shape[1]} chieu"
                  f" — tang --samples, ket qua khong dang tin")
        probe, baseline = ridge_probe(features, targets)
        share = 100.0 * (1.0 - probe / baseline) if baseline > 0 else float("nan")
        print(f"{label} : probe {probe:.4f} | doan trung binh {baseline:.4f} | rut duoc {share:.0f}%")

    print()
    print("Decoder DA TRAIN, do tren cung mau va cung don vi:")
    for label, total, count, baseline in (
        ("IMU", decoded_imu_error, decoded_imu_count, imu_baseline),
        ("ANH", decoded_image_error, decoded_image_count, image_baseline),
    ):
        value = total / max(count, 1)
        share = 100.0 * (1.0 - value / baseline) if baseline > 0 else float("nan")
        print(f"  {label} : decoder {value:.4f} | doan trung binh {baseline:.4f} | rut duoc {share:.0f}%")


if __name__ == "__main__":
    main()
