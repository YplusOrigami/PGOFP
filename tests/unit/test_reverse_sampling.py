import math

import numpy as np
import torch

from pgofp.core.specs import SourceSpec
from pgofp.math.reverse_sampling import reconstruct_zt, reverse_candidates
from pgofp.priors.mixture import build_source


def _phi_density_1d(za, zt, t, src):
    """phi(z_a | z_t, c, t) = ((1-t)/t)^d p_src((z_t - (1-t) z_a)/t)  with d = 1."""
    eps = (zt - (1 - t) * za) / t
    return ((1 - t) / t) * src.log_prob(eps.view(1, -1, 1)).exp()[0]


def test_path_reconstruction_exact(ctx_factory, gen):
    ctx = ctx_factory(B=4, d=3, M=1)
    src = build_source(SourceSpec(kind="prior_mixture", beta=0.6, sigma_prior=0.2), ctx, 3)
    zt = torch.randn(4, 3, generator=gen, dtype=torch.float64).float()
    t = torch.rand(4, 1, generator=gen) * 0.9 + 0.05
    eps, za, v = reverse_candidates(zt, t, src, 16, gen)
    assert eps.shape == za.shape == v.shape == (4, 16, 3)
    assert torch.allclose(reconstruct_zt(za, eps, t), zt.unsqueeze(1).expand(-1, 16, -1), atol=1e-5)
    assert torch.allclose(v, eps - za, atol=1e-6)
    assert torch.allclose(v, (eps - zt.unsqueeze(1)) / (1 - t.unsqueeze(1)), atol=1e-4)


def test_kernel_normalizes_and_matches_samples_shifted_hetero(ctx_factory, gen):
    """Non-zero-mean, non-unit-variance, mixture source: the reverse kernel is phi, not a standard-normal kernel."""
    ctx = ctx_factory(B=1, d=1, M=1)
    spec = SourceSpec(kind="prior_mixture", beta=0.5, sigma_prior=0.2)
    src = build_source(spec, ctx, 1)
    t = torch.tensor([[0.6]])
    zt = torch.tensor([[0.35]])
    za_grid = torch.linspace(-12, 12, 40001)
    phi = _phi_density_1d(za_grid, 0.35, 0.6, src)
    assert abs(float(torch.trapz(phi, za_grid)) - 1.0) < 2e-3  # kernel integrates to one
    _, za, _ = reverse_candidates(zt, t, src, 400000, gen)
    h, edges = np.histogram(za.flatten().numpy(), bins=80, range=(-4, 4), density=True)
    centers = torch.as_tensor((edges[1:] + edges[:-1]) / 2, dtype=torch.float32)
    ref = _phi_density_1d(centers, 0.35, 0.6, src).numpy()
    assert np.abs(h - ref).max() < 0.06 * max(ref.max(), 1.0)


def test_standard_normal_closed_form(gen):
    from pgofp.priors.gaussian import gaussian_source

    src = gaussian_source(1, 1, 1.0)
    t = torch.tensor([[0.3]])
    zt = torch.tensor([[1.2]])
    _, za, _ = reverse_candidates(zt, t, src, 300000, gen)
    assert abs(float(za.mean()) - 1.2 / 0.7) < 0.01
    assert abs(float(za.std()) - 0.3 / 0.7) < 0.01


def test_shape_validation():
    import pytest

    from pgofp.priors.gaussian import gaussian_source

    src = gaussian_source(3, 2, 1.0)
    with pytest.raises(ValueError):
        reverse_candidates(torch.zeros(3, 2), torch.zeros(3), src, 4)
