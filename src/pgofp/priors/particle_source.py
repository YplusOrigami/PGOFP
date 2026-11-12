"""Particle (MPPI/CEM) priors.

An empirical weighted particle set is a Dirac mixture: it has NO ordinary continuous density. For the density
path we therefore convert particles into a Gaussian mixture with a positive kernel bandwidth. A raw particle
set may only be registered as a sample-only source and is rejected by any configuration needing log_prob.
"""
from __future__ import annotations

import numpy as np

from ..core.exceptions import ConfigError


def particles_to_components(
    particles_norm: np.ndarray,  # [n, P, d] first-step actions in normalized coordinates
    weights: np.ndarray,  # [n, P] non-negative
    n_components: int,
    bandwidth: float,
    sigma_min: float = 0.05,
    atanh_eps: float = 1e-3,
):
    """Keep the n_components heaviest particles per env as kernel components.

    Returns (mu [n, M, d] pre-tanh, std [n, M, d] latent, comp_weights [n, M]).
    """
    if bandwidth < sigma_min:
        raise ConfigError(f"kernel bandwidth {bandwidth} < sigma_min {sigma_min}: Dirac sources have no log_prob")
    n, P, d = particles_norm.shape
    M = min(n_components, P)
    order = np.argsort(-weights, axis=1)[:, :M]
    top = np.take_along_axis(particles_norm, order[:, :, None].repeat(d, 2), axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    w = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
    lim = 1.0 - atanh_eps
    mu = np.arctanh(np.clip(top, -lim, lim))
    std = np.full_like(mu, bandwidth)
    return mu.astype(np.float32), std.astype(np.float32), w.astype(np.float32)


def require_density(source_kind: str) -> None:
    if source_kind == "particles_raw":
        raise ConfigError("raw particle (Dirac) sources have no log_prob; use particles_to_components() with a positive bandwidth")
