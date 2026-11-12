import math

import numpy as np
import pytest
import torch

from pgofp.core.specs import SourceSpec
from pgofp.priors.base import GaussianMixtureSource
from pgofp.priors.gaussian import gaussian_source
from pgofp.priors.mixture import build_source


def _mix_spec(beta, **kw):
    return SourceSpec(kind="prior_mixture", beta=beta, sigma_prior=0.3, **kw)


def test_gaussian_moments(gen):
    src = gaussian_source(1, 3, sigma=1.5)
    z = src.sample((200000,), gen)
    assert z.shape == (1, 200000, 3)
    assert abs(float(z.mean())) < 0.02
    assert abs(float(z.std()) - 1.5) < 0.02


def test_beta_endpoints_are_exact_single_component(ctx_factory):
    ctx = ctx_factory(B=3, d=2, M=1)
    s0 = build_source(_mix_spec(0.0), ctx, 2)
    assert s0.n_components == 1 and torch.all(s0.means == 0)
    s1 = build_source(_mix_spec(1.0), ctx, 2)
    assert s1.n_components == 1  # base component dropped entirely
    assert torch.allclose(s1.means[:, 0], ctx.prior_mu[:, 0])
    z = torch.randn(3, 5, 2)
    exp = (-0.5 * ((z - ctx.prior_mu) / 0.3) ** 2 - math.log(0.3)).sum(-1) - math.log(2 * math.pi)
    assert torch.allclose(s1.log_prob(z), exp, atol=1e-5)


def test_mixture_ratio_and_full_density(ctx_factory, gen):
    ctx = ctx_factory(B=1, d=1, M=1)
    beta = 0.3
    src = build_source(_mix_spec(beta), ctx, 1)
    assert src.n_components == 2
    z = src.sample((100000,), gen)
    usage = src.component_usage()
    assert abs(float(usage[1]) - beta) < 0.01
    # full-mixture density integrates to 1 and matches analytic formula
    grid = torch.linspace(-8, 8, 20001).view(1, -1, 1)
    pdf = src.log_prob(grid).exp()[0]
    integral = float(torch.trapz(pdf, grid[0, :, 0]))
    assert abs(integral - 1.0) < 1e-3
    mu = float(ctx.prior_mu[0, 0, 0])
    gx = grid[0, :, 0]
    ref = (1 - beta) * torch.exp(-0.5 * gx**2) / math.sqrt(2 * math.pi) + beta * torch.exp(-0.5 * ((gx - mu) / 0.3) ** 2) / (0.3 * math.sqrt(2 * math.pi))
    assert torch.allclose(pdf, ref, atol=1e-5)
    # empirical histogram matches density (samples are selections, not interpolations)
    h, edges = np.histogram(z.flatten().numpy(), bins=60, range=(-4, 4), density=True)
    centers = torch.as_tensor((edges[1:] + edges[:-1]) / 2, dtype=torch.float32)
    ref_c = ((1 - beta) * torch.exp(-0.5 * centers**2) / math.sqrt(2 * math.pi)
             + beta * torch.exp(-0.5 * ((centers - mu) / 0.3) ** 2) / (0.3 * math.sqrt(2 * math.pi))).numpy()
    assert np.abs(h - ref_c).max() < 0.08


def test_invalid_controller_falls_back_consistently(ctx_factory, gen):
    ctx = ctx_factory(B=2, d=2, M=1)
    ctx.prior_valid[1] = 0.0
    src = build_source(_mix_spec(1.0), ctx, 2)
    # sample 1 is invalid: pure base Gaussian in BOTH sampling and density
    z = src.sample((50000,), gen)
    assert float((z[1].std(0) - 1.0).abs().max()) < 0.02
    assert float((z[0].std(0) - 0.3).abs().max()) < 0.01
    lp = src.log_prob(torch.zeros(2, 1, 2))
    assert abs(float(lp[1]) - (-math.log(2 * math.pi))) < 1e-5


def test_sigma_clamped_positive(ctx_factory):
    ctx = ctx_factory(B=2, d=2, M=1)
    ctx.prior_logstd.fill_(-20.0)
    spec = SourceSpec(kind="prior_mixture", beta=1.0, prior_std_mode="native", sigma_min=0.05, sigma_max=2.0)
    src = build_source(spec, ctx, 2)
    assert torch.allclose(src.log_stds.exp(), torch.full_like(src.log_stds, 0.05), atol=1e-6)
    assert torch.isfinite(src.log_prob(torch.zeros(2, 2))).all()


def test_generator_reproducible(ctx_factory):
    ctx = ctx_factory(B=3, d=2, M=1)
    src = build_source(_mix_spec(0.5), ctx, 2)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(src.sample((8,), g1), src.sample((8,), g2))


def test_multi_controller_weights(ctx_factory, gen):
    ctx = ctx_factory(B=1, d=1, M=2)
    spec = SourceSpec(kind="prior_mixture", beta=1.0, component_weights=(0.25, 0.75), sigma_prior=0.1)
    src = build_source(spec, ctx, 1)
    src.sample((40000,), gen)
    u = src.component_usage()
    assert abs(float(u[0]) - 0.25) < 0.01 and abs(float(u[1]) - 0.75) < 0.01


def test_log_prob_rejects_bad_shapes(ctx_factory):
    src = gaussian_source(3, 2, 1.0)
    with pytest.raises(ValueError):
        src.log_prob(torch.zeros(4, 2))


def test_spec_validation():
    with pytest.raises(ValueError):
        SourceSpec(kind="prior_mixture", beta=1.5)
    with pytest.raises(ValueError):
        SourceSpec(kind="prior_mixture", beta=0.5, component_weights=(0.5, 0.2))
    assert SourceSpec(beta=0.0).hash == SourceSpec(beta=0.0, version=5).hash  # version excluded from content hash
