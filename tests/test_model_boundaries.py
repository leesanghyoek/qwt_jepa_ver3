import copy
import inspect

import pytest
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


def test_absolute_decoder_ignores_the_input_and_preserves_batch_permutation():
    config = load_config("configs/smoke.yaml")
    decoder = build_decoders(config, residual=False, skips=False).eval()
    signature = inspect.signature(decoder.forward)
    assert list(signature.parameters) == ["ZI", "ZU", "image_base", "imu_base",
                                      "image_skips", "imu_skips"]
    zi = torch.randn(3, 32, 2, 2)
    zu = torch.randn(3, 32, 2)
    image_coeff, imu_coeff = decoder(zi, zu)
    # Che do tuyet doi phai BO QUA base, neu khong day la mot skip lot vao.
    other_image = torch.randn_like(image_coeff)
    other_imu = torch.randn_like(imu_coeff)
    same_image, same_imu = decoder(zi, zu, other_image, other_imu)
    assert torch.equal(same_image, image_coeff)
    assert torch.equal(same_imu, imu_coeff)
    permutation = torch.tensor([2, 0, 1])
    image_permuted, imu_permuted = decoder(zi[permutation], zu[permutation])
    assert torch.allclose(image_permuted, image_coeff[permutation])
    assert torch.allclose(imu_permuted, imu_coeff[permutation])


def test_residual_decoder_starts_at_identity_and_needs_the_base():
    config = load_config("configs/smoke.yaml")
    decoder = build_decoders(config, residual=True, skips=False).eval()
    zi = torch.randn(3, 32, 2, 2)
    zu = torch.randn(3, 32, 2)
    image_base = torch.randn(3, 48, 16, 16)
    imu_base = torch.randn(3, 12, 16)
    with torch.no_grad():
        image_coeff, imu_coeff = decoder(zi, zu, image_base, imu_base)
    # Head zero-init: san dam bao bang identity truoc khi hoc bat cu dieu gi.
    assert torch.equal(image_coeff, image_base)
    assert torch.equal(imu_coeff, imu_base)
    with pytest.raises(ValueError, match="input coefficients"):
        decoder(zi, zu)


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


def test_encoder_only_reveals_stages_when_asked_and_keeps_the_same_output():
    config = load_config("configs/smoke.yaml")
    encoder = build_backbone(config).image_encoder.eval()
    coefficients = torch.randn(2, 48, 16, 16)
    with torch.no_grad():
        plain = encoder(coefficients)
        final, stages = encoder(coefficients, return_stages=True)
    assert torch.equal(plain, final)                      # xin stage khong doi ket qua
    assert len(stages) == 3                               # stage cuoi la FI, khong ke
    assert tuple(s.shape[1] for s in stages) == encoder.skip_channels
    # Tho -> min, dung thu tu decoder tieu thu.
    assert stages[0].shape[-1] < stages[1].shape[-1] < stages[2].shape[-1]


def test_skip_decoder_keeps_the_identity_floor_and_refuses_a_mismatched_call():
    config = load_config("configs/smoke.yaml")
    system = RestorationSystem(build_backbone(config), ImuNormalizer(),
                               build_decoders(config, residual=True, skips=True)).eval()
    image = torch.rand(2, 3, 32, 32)
    imu = torch.randn(2, 6, 32)
    times = torch.arange(32).float().mul(0.01).repeat(2, 1)
    with torch.no_grad():
        latent = system.encode(image, imu, times.mean(dim=1), times)
        restored = system.decode(latent)
    assert latent.image_skips is not None and latent.imu_skips is not None
    # Head zero-init VA SkipMerge khoi tao identity: san identity phai song sot
    # qua ca ba diem noi, neu khong residual mat y nghia ngay o update 0.
    assert torch.allclose(restored.image, image, atol=1e-5)

    without = build_decoders(config, residual=True, skips=False).eval()
    with pytest.raises(ValueError, match="refusing to use them"):
        without(latent.ZI, latent.ZU, latent.image_coefficients, latent.imu_coefficients,
                latent.image_skips, latent.imu_skips)
    with pytest.raises(ValueError, match="pass the encoder stages"):
        build_decoders(config, residual=True, skips=True)(latent.ZI, latent.ZU,
                                                          latent.image_coefficients,
                                                          latent.imu_coefficients)


def test_phase1_anchor_decoder_never_receives_skips():
    config = load_config("configs/smoke.yaml")
    enabled = copy.deepcopy(config)
    enabled["phase1"]["decoder_enabled"] = True
    enabled["phase1"]["coefficient_reconstruction_loss_weight"] = 0.45
    model = build_phase1_model(enabled, ImuNormalizer())
    # Neo phai ep thong tin VAO latent. Mot duong vong tu encoder cho phep no dat
    # loss thap ma latent van rong — dung that bai ma neo sinh ra de chan.
    assert model.decoders.uses_skips is False
    assert model.decoders.image.uses_skips is False
