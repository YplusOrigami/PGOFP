import math

import torch

from pgofp.math.action_transform import (atanh_clamped, log_abs_det_affine, log_abs_det_tanh, to_normalized,
                                         to_physical)


def test_log_det_tanh_matches_autograd_jacobian():
    z = torch.randn(5, 3, dtype=torch.float64)
    ld = log_abs_det_tanh(z)
    for b in range(5):
        J = torch.autograd.functional.jacobian(torch.tanh, z[b])
        assert abs(float(torch.linalg.slogdet(J)[1]) - float(ld[b])) < 1e-9


def test_log_det_tanh_stable_at_extremes():
    z = torch.tensor([[-40.0, 0.0, 40.0, 1e3]], dtype=torch.float64)
    ld = log_abs_det_tanh(z)
    assert torch.isfinite(ld).all()
    # log(1 - tanh^2 z) = log 4 - 2|z| + O(e^{-2|z|}) for large |z|; at z=0 it is 0
    expected = (math.log(4) - 80) + 0.0 + (math.log(4) - 80) + (math.log(4) - 2e3)
    assert abs(float(ld) - expected) < 1e-9


def test_atanh_roundtrip_and_clamp_ratio():
    a = torch.tensor([[0.3, -0.7], [1.0, -1.0]], dtype=torch.float64)
    z, ratio = atanh_clamped(a, eps=1e-3)
    assert torch.isfinite(z).all()
    assert abs(float(ratio) - 0.5) < 1e-12  # two of four coordinates hit the clamp
    assert torch.allclose(torch.tanh(z[0]), a[0], atol=1e-12)


def test_affine_roundtrip_and_shapes():
    bias, scale = torch.tensor([1.0, -2.0]), torch.tensor([3.0, 0.5])
    a = torch.rand(7, 2) * 2 - 1
    assert torch.allclose(to_normalized(to_physical(a, bias, scale), bias, scale), a, atol=1e-6)
    assert abs(float(log_abs_det_affine(scale)) - (math.log(3.0) + math.log(0.5))) < 1e-6
    assert to_physical(a, bias, scale).shape == (7, 2)
