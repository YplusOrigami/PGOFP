"""Build the conditional SourceDistribution from (SourceSpec, frozen prior context).

p_src(z|c) = (1-beta) p_base(z) + beta * sum_m w_m q_m(z|c)
  * sampling picks a component per sample (Bernoulli/categorical selection, NOT interpolation of samples)
  * log_prob is the full logsumexp mixture density
  * beta in {0,1} take exact single-component branches (no log(0) arithmetic in the common case)
  * an invalid controller output (valid=0) moves that component's weight to the base component, and both
    sampling and log_prob see the same fallback distribution
"""
from __future__ import annotations

import math

import torch

from ..core.specs import SourceSpec
from ..core.types import ContextBatch
from .base import GaussianMixtureSource
from .gaussian import gaussian_source


def prior_stds(spec: SourceSpec, ctx: ContextBatch) -> torch.Tensor:
    """[B, M, d] positive stds of the prior components in latent units, clamped to [sigma_min, sigma_max]."""
    if spec.prior_std_mode == "native":
        std = ctx.prior_logstd.exp() * spec.std_multiplier
    else:
        std = torch.full_like(ctx.prior_mu, spec.sigma_prior)
    return std.clamp(spec.sigma_min, spec.sigma_max)


def build_source(spec: SourceSpec, ctx: ContextBatch, action_dim: int) -> GaussianMixtureSource:
    B = ctx.batch_size
    dev, dt = ctx.obs_raw.device, ctx.obs_raw.dtype
    M = ctx.prior_mu.shape[1]
    if spec.kind == "gaussian" or spec.beta == 0.0 or M == 0:
        return gaussian_source(B, action_dim, spec.sigma_base, dev, dt)
    if len(spec.component_weights) != M:
        raise ValueError(f"spec has {len(spec.component_weights)} component weights but context has {M} prior components")

    comp_w = torch.as_tensor(spec.component_weights, device=dev, dtype=dt).unsqueeze(0)  # [1, M]
    w_prior = spec.beta * comp_w * ctx.prior_valid  # [B, M]; invalid -> weight 0
    w_base = (1.0 - w_prior.sum(-1, keepdim=True)).clamp_min(0.0)  # [B, 1]

    p_means = ctx.prior_mu
    p_logstd = prior_stds(spec, ctx).log()
    drop_base = spec.beta >= 1.0 and bool((w_base <= 0).all())
    if drop_base:
        return GaussianMixtureSource(p_means, p_logstd, torch.log(w_prior.clamp_min(1e-38)).masked_fill(w_prior <= 0, float("-inf")))

    b_means = torch.zeros(B, 1, action_dim, device=dev, dtype=dt)
    b_logstd = torch.full((B, 1, action_dim), math.log(spec.sigma_base), device=dev, dtype=dt)
    means = torch.cat([b_means, p_means], 1)
    log_stds = torch.cat([b_logstd, p_logstd], 1)
    w = torch.cat([w_base, w_prior], 1)
    log_w = torch.where(w > 0, w.clamp_min(1e-38).log(), torch.full_like(w, float("-inf")))
    return GaussianMixtureSource(means, log_stds, log_w)
