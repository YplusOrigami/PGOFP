"""Controller outputs -> frozen latent prior parameters cached in the replay/context.

Deterministic controller: a_prior_norm -> mu = atanh(clamp(a_prior_norm, +-(1-eps)))  (pre-tanh center),
then q(z|c) = N(mu, diag(sigma_prior^2)).  This is a pre-tanh center + latent perturbation, NOT an
action-space Gaussian: with nonzero sigma the action mean generally differs from the controller center.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.types import ControllerPrediction


def atanh_np(a: np.ndarray, eps: float) -> tuple:
    lim = 1.0 - eps
    clamped = np.abs(a) > lim
    return np.arctanh(np.clip(a, -lim, lim)), float(clamped.mean()) if a.size else 0.0


def prediction_to_prior_arrays(
    pred: ControllerPrediction, n_envs: int, n_prior: int, action_dim: int, atanh_eps: float = 1e-3
) -> Dict[str, np.ndarray]:
    """Normalize a ControllerPrediction into replay arrays prior_mu/prior_logstd/prior_valid ([n, M, d], [n, M])."""
    if pred.center_latent is not None:
        mu = np.asarray(pred.center_latent, dtype=np.float32)
        clamp_ratio = 0.0
    elif pred.center_norm is not None:
        mu, clamp_ratio = atanh_np(np.asarray(pred.center_norm, dtype=np.float64), atanh_eps)
        mu = mu.astype(np.float32)
    else:
        raise ValueError("ControllerPrediction needs center_norm or center_latent")
    if mu.shape != (n_envs, n_prior, action_dim):
        raise ValueError(f"prediction center shape {mu.shape} != {(n_envs, n_prior, action_dim)}")
    if not np.isfinite(mu).all():
        raise ValueError("non-finite controller center")
    if pred.log_std_latent is not None:
        logstd = np.asarray(pred.log_std_latent, dtype=np.float32)
    else:
        logstd = np.zeros_like(mu)
    valid = np.ones((n_envs, n_prior), dtype=np.float32) if pred.valid is None else np.asarray(pred.valid, dtype=np.float32)
    return {"prior_mu": mu, "prior_logstd": logstd, "prior_valid": valid, "atanh_clamp_ratio": np.float32(clamp_ratio)}


def empty_prior_arrays(n_envs: int, action_dim: int) -> Dict[str, np.ndarray]:
    return {
        "prior_mu": np.zeros((n_envs, 0, action_dim), np.float32),
        "prior_logstd": np.zeros((n_envs, 0, action_dim), np.float32),
        "prior_valid": np.zeros((n_envs, 0), np.float32),
        "atanh_clamp_ratio": np.float32(0.0),
    }
