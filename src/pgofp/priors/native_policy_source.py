"""Frozen tanh-Gaussian PPO as a *native* source.

A tanh-Gaussian PPO already is a pre-tanh Gaussian N(mu_psi(s), std_psi^2): export (mu, log_std) as-is and
use prior_std_mode="native" (no extra noise on top). A Gaussian+clip PPO has boundary mass after clipping and
is NOT a native pre-tanh source; only its center may be reused as `ppo_center_gaussian`
(prior_std_mode="fixed", explicit sigma_prior), and it must be labelled that way in experiments.
"""
from __future__ import annotations

from ..core.specs import SourceSpec


def native_ppo_spec(beta: float, std_multiplier: float = 1.0, **kw) -> SourceSpec:
    return SourceSpec(kind="prior_mixture", beta=beta, prior_std_mode="native", std_multiplier=std_multiplier, **kw)


def ppo_center_gaussian_spec(beta: float, sigma_prior: float, **kw) -> SourceSpec:
    return SourceSpec(kind="prior_mixture", beta=beta, prior_std_mode="fixed", sigma_prior=sigma_prior, **kw)
