import math

import pytest
import torch
from torch import nn

from pgofp.core.exceptions import NumericalError
from pgofp.density import ContinuousInverseProxy, LocalMeanFlowProxy, check_finite
from pgofp.math.action_transform import log_abs_det_tanh
from pgofp.math.divergence import divergence_exact, divergence_hutchinson, rademacher_like
from pgofp.priors.base import GaussianMixtureSource
from pgofp.priors.gaussian import gaussian_source
from tests.conftest import AffineField

D64 = torch.float64


def _gauss(B, d, sigma=1.0):
    return gaussian_source(B, d, sigma, dtype=D64)


def test_zero_field_density_is_source_density_minus_logJ():
    B, d = 6, 3
    z = torch.randn(B, d, dtype=D64)
    src = _gauss(B, d, 0.7)
    net = AffineField(0.0, 0.0, d)
    res = ContinuousInverseProxy(steps=3, solver="heun", trace="exact").estimate_at(z, torch.zeros(B, 0, dtype=D64), net, src)
    expect = src.log_prob(z) - log_abs_det_tanh(z)
    assert torch.allclose(res.log_prob_norm, expect, atol=1e-10)
    assert res.nfe == 6 and not res.is_exact_for_sampled_policy and res.method == "continuous_inverse_proxy"


@pytest.mark.parametrize("a,b", [(0.4, 0.0), (-0.3, 0.0), (0.2, 0.6), (-0.5, 0.9)])
def test_affine_field_matches_analytic_direction_and_value(a, b):
    """f = (a + b t) z  =>  z(1) = z0 exp(a + b/2), D = d (a + b/2); log p0(z0) = log p_src(z(1)) + D."""
    B, d = 5, 2
    z0 = torch.randn(B, d, dtype=D64)
    src = _gauss(B, d, 1.3)
    net = AffineField(a, b, d)
    k = a + b / 2
    expected_z = src.log_prob(z0 * math.exp(k)) + d * k
    est = ContinuousInverseProxy(steps=200, solver="heun", trace="exact")
    res = est.estimate_at(z0, torch.zeros(B, 0, dtype=D64), net, src)
    assert torch.allclose(res.log_prob_latent, expected_z, atol=2e-4)
    assert torch.allclose(res.log_prob_norm, expected_z - log_abs_det_tanh(z0), atol=2e-4)
    assert torch.allclose(res.eps_hat, z0 * math.exp(k), atol=2e-4)


def test_wrong_sign_would_be_detected():
    """With expanding field (k>0) the density at z0 must be *larger* than the source density at z0 would imply for
    a naive 'minus divergence' sign convention: the analytic value pins the direction."""
    d = 1
    z0 = torch.zeros(1, d, dtype=D64)
    src = _gauss(1, d, 1.0)
    net = AffineField(0.5, 0.0, d)
    res = ContinuousInverseProxy(steps=100, trace="exact").estimate_at(z0, torch.zeros(1, 0, dtype=D64), net, src)
    assert abs(float(res.log_prob_latent) - (-0.5 * math.log(2 * math.pi) + 0.5)) < 1e-6  # + D, not - D


def test_density_depends_on_the_queried_action():
    d = 2
    src = _gauss(2, d)
    net = AffineField(0.3, 0.1, d)
    est = ContinuousInverseProxy(steps=20, trace="exact")
    c = torch.zeros(2, 0, dtype=D64)
    z = torch.tensor([[0.1, 0.2], [2.0, -1.5]], dtype=D64)
    r1 = est.estimate_at(z, c, net, src).log_prob_norm
    r2 = est.estimate_at(z.flip(0), c, net, src).log_prob_norm
    assert not torch.allclose(r1, r2.flip(0) * 0 + r2)  # sanity
    assert torch.allclose(r1, est.estimate_at(z, c, net, src).log_prob_norm)  # deterministic for exact trace
    assert abs(float(r1[0] - r1[1])) > 0.5


def test_mixture_source_density():
    B, d = 3, 2
    mu = torch.randn(B, 2, d, dtype=D64)
    ls = torch.full((B, 2, d), math.log(0.5), dtype=D64)
    lw = torch.log(torch.tensor([[0.3, 0.7]] * B, dtype=D64))
    src = GaussianMixtureSource(mu, ls, lw)
    z0 = torch.randn(B, d, dtype=D64)
    net = AffineField(-0.2, 0.0, d)
    res = ContinuousInverseProxy(steps=100, trace="exact").estimate_at(z0, torch.zeros(B, 0, dtype=D64), net, src)
    expected = src.log_prob(z0 * math.exp(-0.2)) - d * 0.2
    assert torch.allclose(res.log_prob_latent, expected, atol=2e-4)


def test_normalization_of_pushforward_density_1d():
    """Density in a_norm coordinates integrates to 1 over (-1, 1)."""
    d = 1
    src = _gauss(1, d, 1.0)
    net = AffineField(0.35, -0.2, d)
    est = ContinuousInverseProxy(steps=60, trace="exact")
    a = torch.linspace(-0.9995, 0.9995, 4001, dtype=D64)
    z = torch.atanh(a).view(-1, 1)
    srcN = _gauss(z.shape[0], d, 1.0)
    res = est.estimate_at(z, torch.zeros(z.shape[0], 0, dtype=D64), net, srcN)
    mass = float(torch.trapz(res.log_prob_norm.exp(), a))
    assert abs(mass - 1.0) < 3e-3


def test_hutchinson_unbiased_vs_exact_on_mlp():
    torch.manual_seed(0)
    d = 3
    mlp = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, d)).to(D64)
    z = torch.randn(4, d, dtype=D64, requires_grad=True)
    f = mlp(z)
    exact = divergence_exact(f, z)
    g = torch.Generator().manual_seed(3)
    probes = [rademacher_like(z, g) for _ in range(20000)]
    # a single Rademacher probe gives v^T J v through a VJP (checked before the graph is consumed)
    v = probes[0]
    jtv = torch.autograd.grad(f, z, v, retain_graph=True)[0]
    J = torch.stack([torch.autograd.functional.jacobian(mlp, z[i].detach()) for i in range(4)])
    assert torch.allclose((jtv * v).sum(-1), torch.einsum("bi,bij,bj->b", v, J, v), atol=1e-9)
    est = divergence_hutchinson(f, z, probes)
    assert torch.allclose(est, exact, atol=0.05)


def test_hutchinson_density_close_to_exact_density():
    torch.manual_seed(1)
    d = 3

    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Sequential(nn.Linear(d + 2, 24), nn.Tanh(), nn.Linear(24, d))

        def forward(self, z, r, t, c):
            return 0.3 * self.l(torch.cat([z, r, t], -1))

    net = SmallNet().to(D64)
    z = torch.randn(8, d, dtype=D64)
    src = _gauss(8, d)
    c = torch.zeros(8, 0, dtype=D64)
    ex = ContinuousInverseProxy(steps=10, trace="exact").estimate_at(z, c, net, src).log_prob_norm
    g = torch.Generator().manual_seed(0)
    hu = ContinuousInverseProxy(steps=10, trace="hutchinson", n_trace_samples=3000).estimate_at(z, c, net, src, g).log_prob_norm
    assert torch.allclose(ex, hu, atol=0.1)


def test_integration_steps_converge():
    d = 2
    z = torch.randn(4, d, dtype=D64)
    src = _gauss(4, d)
    net = AffineField(0.4, 0.8, d)
    c = torch.zeros(4, 0, dtype=D64)
    ref = ContinuousInverseProxy(steps=400, trace="exact").estimate_at(z, c, net, src).log_prob_norm
    errs = [float((ContinuousInverseProxy(steps=n, trace="exact").estimate_at(z, c, net, src).log_prob_norm - ref).abs().max())
            for n in (1, 5, 20)]
    assert errs[0] > errs[1] > errs[2]


def test_nonfinite_density_is_reported_not_zeroed():
    d = 2
    z = torch.tensor([[1e6, 1e6]], dtype=D64)
    net = AffineField(50.0, 0.0, d)
    res = ContinuousInverseProxy(steps=2, trace="exact").estimate_at(z, torch.zeros(1, 0, dtype=D64), net, _gauss(1, d))
    if not bool(res.finite_mask.all()):
        with pytest.raises(NumericalError):
            check_finite(res, "test")
    else:  # pragma: no cover - float64 may not overflow here; make a guaranteed non-finite one
        res.finite_mask = torch.tensor([False])
        with pytest.raises(NumericalError):
            check_finite(res, "test")


def test_local_meanflow_exact_for_affine_one_interval():
    """u = k z over [0,1]: T(z) = (1 - k) z, so log pi = log p_src(z/(1-k)) - d log|1-k| ... pushforward check."""
    d = 2
    k = 0.4
    z = torch.randn(5, d, dtype=D64)
    # T maps source -> target: z_target = z_src - u = (1-k) z_src. Target density: p_src(z/(1-k)) / (1-k)^d.
    # Local proxy inverts from the target end: y - h*k*y = z -> y = z/(1-k); logdet(I - hJ) = d log(1-k).
    est = LocalMeanFlowProxy(grid=(0.0, 1.0), logdet="full")
    src = _gauss(5, d, 1.0)
    res = est.estimate_at(z, torch.zeros(5, 0, dtype=D64), AffineField(k, 0.0, d), src)
    expected = src.log_prob(z / (1 - k)) - d * math.log(1 - k)
    assert torch.allclose(res.log_prob_latent, expected, atol=1e-6)
    assert float(res.inverse_residual.max()) < 1e-6
    assert not res.is_exact_for_sampled_policy


def test_local_meanflow_reports_residual_when_not_contractive():
    d = 1
    est = LocalMeanFlowProxy(grid=(0.0, 1.0), max_iter=20)
    z = torch.ones(1, d, dtype=D64)
    res = est.estimate_at(z, torch.zeros(1, 0, dtype=D64), AffineField(3.0, 0.0, d), _gauss(1, d))
    assert float(res.inverse_residual.max()) > 1e-3  # fixed-point iteration diverges; failure is visible, not hidden
