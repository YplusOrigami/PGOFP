"""Q-reweighted transport validated on low-dimensional problems with analytic references."""
import math

import numpy as np
import pytest
import torch

from pgofp.algorithms.objectives import FlowSettings, actor_loss_endpoint_is, actor_loss_reverse
from pgofp.math.reverse_sampling import mixture_reverse_candidates, reverse_candidates
from pgofp.networks.mean_flow import MeanFlowNet
from pgofp.priors.gaussian import gaussian_source
from pgofp.utils.seed import RngStreams

D64 = torch.float64


def _energy(alpha, astar):
    def E(z):  # log q*(z) up to a constant: Q/alpha + log|J_tanh|
        a = torch.tanh(z)
        return (-(a - astar) ** 2) / alpha + 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))
    return E


def _exact_velocity(E, zg, zt, t):
    s = t / (1 - t)
    lphi = -0.5 * ((zg[None, :] - zt[:, None] / (1 - t)) / s) ** 2
    w = torch.softmax(E(zg)[None, :] + lphi, 1)
    return (zt - (w * zg[None, :]).sum(1)) / t


@pytest.mark.parametrize("t", [0.3, 0.7])
def test_pure_reverse_snis_matches_exact_posterior_velocity(t):
    torch.manual_seed(0)
    alpha, astar = 0.2, 0.5
    E = _energy(alpha, astar)
    zg = torch.linspace(-9, 9, 6001, dtype=D64)
    zt = torch.tensor([0.4], dtype=D64)
    v_exact = float(_exact_velocity(E, zg, zt, t))
    src = gaussian_source(1, 1, 1.0, dtype=D64)
    tt = torch.tensor([[t]], dtype=D64)
    est = []
    for _ in range(40):
        eps, za, v = reverse_candidates(zt.view(1, 1), tt, src, 20000)
        w = torch.softmax(E(za[..., 0]), -1)
        est.append(float((w * v[..., 0]).sum()))
    assert abs(np.mean(est) - v_exact) < 0.02


def test_mixture_proposal_is_consistent_and_reduces_variance_at_large_t():
    """The MIS-corrected local candidates target the SAME posterior (unbiased in the limit) with a far smaller spread."""
    torch.manual_seed(0)
    alpha, astar = 0.05, 0.5
    E = _energy(alpha, astar)
    zg = torch.linspace(-9, 9, 6001, dtype=D64)
    t = 0.95
    zt = torch.tensor([[0.4]], dtype=D64)
    tt = torch.tensor([[t]], dtype=D64)
    v_exact = float(_exact_velocity(E, zg, zt[:, 0], t))
    src = gaussian_source(1, 1, 1.0, dtype=D64)
    g = torch.Generator().manual_seed(1)
    pure, mix = [], []
    for _ in range(300):
        eps, za, v = reverse_candidates(zt, tt, src, 16, g)
        pure.append(float((torch.softmax(E(za[..., 0]), -1) * v[..., 0]).sum()))
        cen = (0.55 + 0.3 * torch.randn(1, 16, 1, generator=g, dtype=D64))
        eps, za, v, corr = mixture_reverse_candidates(zt, tt, src, 16, cen, 0.3, g)
        mix.append(float((torch.softmax(E(za[..., 0]) + corr, -1) * v[..., 0]).sum()))
    assert abs(np.mean(mix) - v_exact) < 0.03
    assert np.std(mix) < 0.2 * np.std(pure)


def test_exact_velocity_field_transports_source_to_the_weighted_target():
    """Integrating v*(z,t) from the Gaussian source must give the Q-weighted action distribution exp(Q/alpha) (the theory behind 6.2)."""
    torch.manual_seed(0)
    alpha, astar = 0.05, 0.5
    E = _energy(alpha, astar)
    zg = torch.linspace(-8, 8, 2001, dtype=D64)
    z = torch.randn(1000, dtype=D64)
    ts = torch.linspace(0.995, 0.005, 101, dtype=D64)
    for i in range(100):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        h = t1 - t0
        k1 = _exact_velocity(E, zg, z, t0)
        k2 = _exact_velocity(E, zg, z + h * k1, t1)
        z = z + 0.5 * h * (k1 + k2)
    a = torch.tanh(z)
    ag = torch.linspace(-0.9999, 0.9999, 20001, dtype=D64)
    pa = torch.exp(-((ag - astar) ** 2) / alpha)
    pa /= torch.trapz(pa, ag)
    mt = float(torch.trapz(ag * pa, ag))
    st = float(torch.sqrt(torch.trapz((ag - mt) ** 2 * pa, ag)))
    assert abs(float(a.mean()) - mt) < 0.02 and abs(float(a.std()) - st) < 0.02


class _BimodalQ:
    """Q(a) = alpha * log( 0.7 N(a; +0.6, s) + 0.3 N(a; -0.6, s) ) so that exp(Q/alpha) is the bimodal density (modes 70/30)."""

    def __init__(self, alpha, s=0.12):
        self.alpha, self.s = alpha, s

    def min_q(self, cond, a):
        a0 = a[..., 0]
        m = 0.7 * torch.exp(-0.5 * ((a0 - 0.6) / self.s) ** 2) + 0.3 * torch.exp(-0.5 * ((a0 + 0.6) / self.s) ** 2)
        return self.alpha * torch.log(m + 1e-30)


def _train(objective, steps=1200, K=32, K_local=16, seed=0, alpha=0.1, B=256, lr=1e-3):
    torch.manual_seed(seed)
    net = MeanFlowNet(1, 1, (96, 96), 16, "silu")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = RngStreams(seed)
    src = gaussian_source(B, 1, 1.0)
    q = _BimodalQ(alpha)
    flow = FlowSettings(0.02, 0.5, "uniform")
    cond = torch.zeros(B, 1)
    for _ in range(steps):
        if objective == "endpoint_is":
            loss, _ = actor_loss_endpoint_is(net, q, cond, src, alpha, K=K, flow=flow, rng=rng)
        else:
            loss, _ = actor_loss_reverse(net, q, cond, src, alpha, K=K, flow=flow, objective=objective, zt_proposal="standard_normal",
                                          a_replay_latent=None, rng=rng, K_local=K_local)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
    with torch.no_grad():
        a = torch.tanh(net.one_step(torch.randn(20000, 1), torch.zeros(20000, 1)))[:, 0]
    return net, a


@pytest.mark.slow
def test_q_reweighting_moves_mass_to_high_q_mode_and_no_q_does_not():
    _, a_q = _train("reverse_qrmf")
    _, a_noq = _train("reverse_no_q")
    frac_q = float((a_q > 0).float().mean())
    frac_noq = float((a_noq > 0).float().mean())
    assert frac_q > 0.6, frac_q  # weights tilt toward the 70% mode
    assert 0.35 < frac_noq < 0.65, frac_noq  # without Q only the Jacobian term acts (≈ uniform in a)
    assert frac_q > frac_noq + 0.1


@pytest.mark.slow
def test_reverse_and_endpoint_is_objectives_agree_roughly():
    _, a_rev = _train("reverse_qrmf")
    _, a_end = _train("endpoint_is", K=64)
    assert abs(float((a_rev > 0).float().mean()) - float((a_end > 0).float().mean())) < 0.2
    assert abs(float(a_rev.mean()) - float(a_end.mean())) < 0.3
