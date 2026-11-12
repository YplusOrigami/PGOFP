"""Integration-step and trace-sample convergence of the continuous density proxy against float64 references."""
import numpy as np
import torch
from torch import nn

from pgofp.density import ContinuousInverseProxy
from pgofp.priors.gaussian import gaussian_source

D64 = torch.float64


class TimeMLP(nn.Module):
    def __init__(self, d=3, scale=0.4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.l = nn.Sequential(nn.Linear(d + 1, 32), nn.Tanh(), nn.Linear(32, d)).to(D64)
        self.scale = scale

    def forward(self, z, r, t, c):
        return self.scale * self.l(torch.cat([z, t], -1))


def _setup(B=16, d=3):
    torch.manual_seed(1)
    return TimeMLP(d), torch.randn(B, d, dtype=D64), gaussian_source(B, d, 1.0, dtype=D64), torch.zeros(B, 0, dtype=D64)


def test_integration_error_decreases_with_steps_heun_second_order():
    net, z, src, c = _setup()
    ref = ContinuousInverseProxy(steps=400, solver="heun", trace="exact").estimate_at(z, c, net, src).log_prob_norm
    err = {n: float((ContinuousInverseProxy(steps=n, solver="heun", trace="exact").estimate_at(z, c, net, src).log_prob_norm - ref).abs().max())
           for n in (1, 2, 4, 8)}
    assert err[1] > err[2] > err[4] > err[8]
    assert err[8] < err[1] / 8  # faster than first order over two doublings
    # the default N_est=5 is an approximation: its error is reported, not assumed away
    assert float((ContinuousInverseProxy(steps=5, trace="exact").estimate_at(z, c, net, src).log_prob_norm - ref).abs().max()) > 0


def test_euler_vs_heun_nfe_accounting():
    net, z, src, c = _setup()
    assert ContinuousInverseProxy(steps=5, solver="euler", trace="exact").estimate_at(z, c, net, src).nfe == 5
    assert ContinuousInverseProxy(steps=5, solver="heun", trace="exact").estimate_at(z, c, net, src).nfe == 10


def test_hutchinson_variance_shrinks_with_probe_count_and_is_unbiased():
    net, z, src, c = _setup(B=8)
    ref = ContinuousInverseProxy(steps=5, trace="exact").estimate_at(z, c, net, src).log_prob_norm
    sd = {}
    for n in (1, 4, 16):
        g = torch.Generator().manual_seed(0)
        runs = torch.stack([ContinuousInverseProxy(steps=5, trace="hutchinson", n_trace_samples=n).estimate_at(z, c, net, src, g).log_prob_norm
                            for _ in range(300)])
        sd[n] = float(runs.std(0).mean())
        assert float((runs.mean(0) - ref).abs().max()) < 4 * sd[n] / np.sqrt(300) + 1e-3  # unbiased within MC error
    assert sd[1] > sd[4] > sd[16]
    assert abs(sd[1] / sd[16] - 4.0) < 1.5  # ~1/sqrt(n)


def test_eps_reference_residual_is_reported_for_a_consistent_field():
    """Using the same field to generate z (source->target, backwards in t) and to estimate: recovered source ~ original eps."""
    net, _, src, c = _setup(B=8)
    eps = src.sample((), torch.Generator().manual_seed(0))
    z = eps.clone()
    h = 1.0 / 200
    for i in range(200):
        t = torch.full((8, 1), 1.0 - i * h, dtype=D64)
        z = z - h * net(z, t, t, c)
    res = ContinuousInverseProxy(steps=200, trace="exact").estimate_at(z, c, net, src, eps_ref=eps)
    assert float(res.inverse_residual.max()) < 5e-3
