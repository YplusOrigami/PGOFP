"""Conditional reverse sampling kernel (docs/algorithm_contract.md section 6.2).

For a fixed condition c, flow time t and intermediate point z_t, the interpolation
z_t = (1-t) z_a + t * eps  gives the reverse kernel
    phi(z_a | z_t, c, t) = ((1-t)/t)^d p_src((z_t - (1-t) z_a)/t | c),
which is sampled by drawing eps_k ~ p_src(.|c) and setting z_a_k = (z_t - t eps_k)/(1-t).
This works for ANY source with a sampler (shifted, heteroscedastic, mixture): no standard-normal
reverse kernel is assumed.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def reverse_candidates(
    z_t: torch.Tensor,  # [B, d]
    t: torch.Tensor,  # [B, 1]
    source,  # SourceDistribution with sample((K,), gen) -> [B, K, d]
    K: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (eps [B,K,d], z_a [B,K,d], v [B,K,d]) with v = eps - z_a = (eps - z_t)/(1-t)."""
    if z_t.dim() != 2 or t.shape != (z_t.shape[0], 1):
        raise ValueError(f"expected z_t [B,d] and t [B,1], got {tuple(z_t.shape)} / {tuple(t.shape)}")
    eps = source.sample((K,), generator)  # [B, K, d]
    tt = t.unsqueeze(1)  # [B,1,1]
    z_a = (z_t.unsqueeze(1) - tt * eps) / (1.0 - tt)
    v = eps - z_a
    return eps, z_a, v


def reconstruct_zt(z_a: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """z_t = (1-t) z_a + t eps for candidate tensors [B,K,d] and t [B,1]; used for path-residual checks."""
    tt = t.unsqueeze(1)
    return (1.0 - tt) * z_a + tt * eps


def mixture_reverse_candidates(
    z_t: torch.Tensor,  # [B, d]
    t: torch.Tensor,  # [B, 1]
    source,
    K_reverse: int,
    centers: torch.Tensor,  # [B, J, d] latent samples of the current policy (local proposal centers), J may be 0
    sigma_local: float,
    generator: Optional[torch.Generator] = None,
):
    """Defensive mixture proposal for the reverse posterior  q(z_a | z_t) ∝ exp(E(z_a)) * phi(z_a | z_t, c, t).

    Techniques (balance-heuristic multiple importance sampling):
      * K_reverse candidates from the reverse kernel phi (the spec's main proposal),
      * J local candidates z_c_j + sigma_local * xi, one per policy-sample center z_c_j.
    g_mix(z) = (K_reverse*phi(z) + sum_j N(z; z_c_j, sigma_local^2)) / (K_reverse + J)
    The SNIS weight is  exp(E(z) + log phi(z) - log g_mix(z)); with J = 0 it equals exp(E(z)) exactly (main path).
    Returns (eps [B,K,d], z_a [B,K,d], v [B,K,d], log_corr [B,K]) with log_corr = log phi - log g_mix and K = K_reverse + J.
    """
    B, d = z_t.shape
    eps_r, z_r, _ = reverse_candidates(z_t, t, source, K_reverse, generator)
    J = centers.shape[1]
    if J == 0:
        v = eps_r - z_r
        return eps_r, z_r, v, torch.zeros(B, K_reverse, device=z_t.device, dtype=z_t.dtype)
    xi = torch.randn(B, J, d, generator=generator, device=z_t.device, dtype=z_t.dtype)
    z_l = centers + sigma_local * xi
    z_a = torch.cat([z_r, z_l], 1)  # [B, K, d]
    tt = t.unsqueeze(1)
    eps = (z_t.unsqueeze(1) - (1.0 - tt) * z_a) / tt  # unique eps consistent with (z_t, z_a); equals eps_r on the first block
    v = eps - z_a
    log_phi = d * torch.log((1.0 - tt.squeeze(-1)) / tt.squeeze(-1)) + source.log_prob(eps)  # [B, K]
    diff = z_a.unsqueeze(2) - centers.unsqueeze(1)  # [B, K, J, d]
    log_n = -0.5 * (diff**2).sum(-1) / sigma_local**2 - d * math.log(sigma_local) - 0.5 * d * math.log(2 * math.pi)
    parts = torch.cat([(log_phi + math.log(K_reverse)).unsqueeze(-1), log_n], dim=-1)  # [B, K, 1+J]
    log_g = torch.logsumexp(parts, dim=-1) - math.log(K_reverse + J)
    return eps, z_a, v, log_phi - log_g
