import pytest
import torch
from torch import nn

from pgofp.math.meanflow_target import meanflow_jvp, meanflow_target


class PolyNet(nn.Module):
    """Smooth nonlinear u(z, r, t, c) with learnable params, to check JVP vs finite differences."""

    def __init__(self, d=3, dc=2):
        super().__init__()
        self.l1 = nn.Linear(d + 2 + dc, 16)
        self.l2 = nn.Linear(16, d)

    def forward(self, z, r, t, c):
        return self.l2(torch.tanh(self.l1(torch.cat([z, r, t, c], -1))))


def _setup(N=6, d=3, dc=2, dtype=torch.float64):
    torch.manual_seed(0)
    net = PolyNet(d, dc).to(dtype)
    z = torch.randn(N, d, dtype=dtype)
    t = torch.rand(N, 1, dtype=dtype) * 0.8 + 0.1
    r = t * torch.rand(N, 1, dtype=dtype)
    v = torch.randn(N, d, dtype=dtype)
    c = torch.randn(N, dc, dtype=dtype)
    return net, z, r, t, v, c


def test_jvp_matches_finite_difference():
    net, z, r, t, v, c = _setup()
    fn = lambda zz, rr, tt: net(zz, rr, tt, c)
    _, du = meanflow_jvp(fn, z, r, t, v)
    h = 1e-6
    fd = (fn(z + h * v, r, t + h) - fn(z - h * v, r, t - h)) / (2 * h)  # tangent (v, 0, 1): r is FIXED
    assert torch.allclose(du, fd, atol=1e-7)


def test_r_is_held_fixed_not_differentiated():
    net, z, r, t, v, c = _setup()
    fn = lambda zz, rr, tt: net(zz, rr, tt, c)
    _, du = meanflow_jvp(fn, z, r, t, v)
    h = 1e-6
    wrong = (fn(z + h * v, r + h, t + h) - fn(z - h * v, r - h, t - h)) / (2 * h)
    assert not torch.allclose(du, wrong, atol=1e-5)


def test_target_reduces_to_v_when_r_equals_t():
    net, z, r, t, v, c = _setup()
    tgt, _ = meanflow_target(lambda zz, rr, tt: net(zz, rr, tt, c), z, t.clone(), t, v)
    assert torch.allclose(tgt, v, atol=1e-12)


def test_target_formula():
    net, z, r, t, v, c = _setup()
    fn = lambda zz, rr, tt: net(zz, rr, tt, c)
    tgt, _ = meanflow_target(fn, z, r, t, v)
    _, du = meanflow_jvp(fn, z, r, t, v)
    assert torch.allclose(tgt, v - (t - r) * du, atol=1e-12)


def test_target_carries_no_gradient():
    net, z, r, t, v, c = _setup()
    z.requires_grad_(True)
    v = v.clone().requires_grad_(True)
    tgt, _ = meanflow_target(lambda zz, rr, tt: net(zz, rr, tt, c), z, r, t, v)
    assert not tgt.requires_grad
    assert all(p.grad is None for p in net.parameters())


def test_meanflow_identity_on_exact_linear_flow():
    """For v(z,t) = A z the exact average velocity is known; target must equal u at the same point."""
    d = 2
    A = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float64)
    # exact trajectory z(tau) = expm(A (tau - t)) z_t ;  u(z_t, r, t) = (z_t - z(r)) / ... (average of dz/dtau over [r,t])
    def u_exact(z, r, t, c):
        out = []
        for i in range(z.shape[0]):
            zr = torch.linalg.matrix_exp(A * (r[i] - t[i])) @ z[i]
            out.append((z[i] - zr) / (t[i] - r[i]))
        return torch.stack(out)

    N = 5
    torch.manual_seed(1)
    z = torch.randn(N, d, dtype=torch.float64)
    t = torch.rand(N, 1, dtype=torch.float64) * 0.5 + 0.4
    r = t * 0.3
    v = z @ A.T  # instantaneous velocity at (z, t)
    tgt, _ = meanflow_target(lambda zz, rr, tt: u_exact(zz, rr, tt, None), z, r, t, v)
    assert torch.allclose(tgt, u_exact(z, r, t, None), atol=1e-6)
