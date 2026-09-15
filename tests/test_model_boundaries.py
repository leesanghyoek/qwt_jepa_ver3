import inspect

import torch

from qjepa.config import build_backbone, build_decoders, build_phase1_model, load_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.evaluation.metrics import image_metrics


def _inputs(batch=2):
    image = torch.rand(batch, 3, 32, 32)
    imu = torch.randn(batch, 6, 32)
    times = torch.arange(32).float().mul(0.01).repeat(batch, 1)
    image_time = times.mean(dim=1)
    return image, imu, image_time, times


def test_phase1_has_dense_shapes_teacher_stop_gradient_and_no_decoder():
    config = load_config("configs/smoke.yaml")
    model = build_phase1_model(config, ImuNormalizer())
    image, imu, image_time, times = _inputs()
    latent = model.encode_online(image, imu, image_time, times)
    assert latent.FI.shape == (2, 32, 2, 2)
    assert latent.FU.shape == (2, 32, 2)
    assert latent.ZI.shape == latent.FI.shape
    assert latent.ZU.shape == latent.FU.shape
    target_image, target_imu = model.targets(image, imu)
    assert not target_image.requires_grad and not target_imu.requires_grad
    assert not hasattr(model, "decoder")
    assert not any("decoder" in name for name, _ in model.named_modules())


def test_decoder_api_accepts_latents_only_and_preserves_batch_permutation():
    config = load_config("configs/smoke.yaml")
    decoder = build_decoders(config).eval()
    signature = inspect.signature(decoder.forward)
    assert list(signature.parameters) == ["ZI", "ZU"]
    zi = torch.randn(3, 32, 2, 2)
    zu = torch.randn(3, 32, 2)
    image_coeff, imu_coeff = decoder(zi, zu)
    permutation = torch.tensor([2, 0, 1])
    image_permuted, imu_permuted = decoder(zi[permutation], zu[permutation])
    assert torch.allclose(image_permuted, image_coeff[permutation])
    assert torch.allclose(imu_permuted, imu_coeff[permutation])


def test_restoration_shapes_come_from_absolute_coefficients():
    config = load_config("configs/smoke.yaml")
    system = RestorationSystem(build_backbone(config), ImuNormalizer(), build_decoders(config))
    image, imu, image_time, times = _inputs(batch=1)
    restored = system(image, imu, image_time, times)
    assert restored.image.shape == image.shape
    assert restored.imu_physical.shape == imu.shape
    assert restored.image_coefficients.shape == (1, 48, 16, 16)
    assert restored.imu_coefficients.shape == (1, 12, 16)


def test_ssim_is_one_for_identical_images():
    image = torch.rand(2, 3, 32, 32)
    assert abs(image_metrics(image, image)["image_ssim"] - 1.0) < 1e-5
