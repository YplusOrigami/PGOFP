"""Standard / diagonal Gaussian sources."""
from __future__ import annotations

import math

import torch

from ..core.specs import SourceSpec
from .base import GaussianMixtureSource


def gaussian_source(batch: int, dim: int, sigma: float, device="cpu", dtype=torch.float32) -> GaussianMixtureSource:
    """Uninformative p_base(z) = N(0, sigma^2 I)."""
    means = torch.zeros(batch, 1, dim, device=device, dtype=dtype)
    log_stds = torch.full((batch, 1, dim), math.log(sigma), device=device, dtype=dtype)
    log_w = torch.zeros(batch, 1, device=device, dtype=dtype)
    return GaussianMixtureSource(means, log_stds, log_w)


def diag_gaussian_source(mu: torch.Tensor, std: torch.Tensor) -> GaussianMixtureSource:
    """mu, std: [B, d]."""
    return GaussianMixtureSource(mu.unsqueeze(1), std.clamp_min(1e-8).log().unsqueeze(1),
                                 torch.zeros(mu.shape[0], 1, device=mu.device, dtype=mu.dtype))


def base_spec(sigma_base: float = 1.0) -> SourceSpec:
    return SourceSpec(kind="gaussian", beta=0.0, sigma_base=sigma_base)
