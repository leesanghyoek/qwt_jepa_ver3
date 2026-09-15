import torch

from qjepa.transforms import HaarTransform1D, QuaternionWaveletTransform2D


def test_qwt_roundtrip_and_synthesis_gradient():
    transform = QuaternionWaveletTransform2D().double()
    image = torch.rand(2, 3, 16, 18, dtype=torch.float64)
    coefficients, layout = transform.analysis(image)
    restored = transform.synthesis(coefficients, layout)
    assert coefficients.shape == (2, 48, 8, 9)
    assert torch.allclose(restored, image, atol=1e-11, rtol=1e-11)

    independent_coefficients = coefficients.detach().requires_grad_(True)
    transform.synthesis(independent_coefficients, layout).square().mean().backward()
    assert independent_coefficients.grad is not None
    assert torch.isfinite(independent_coefficients.grad).all()
    assert independent_coefficients.grad.abs().sum() > 0


def test_haar_roundtrip_and_layout_guard():
    transform = HaarTransform1D(channels=6)
    imu = torch.randn(3, 6, 128)
    coefficients, layout = transform.analysis(imu)
    assert coefficients.shape == (3, 12, 64)
    assert torch.allclose(transform.synthesis(coefficients, layout), imu, atol=1e-6)

