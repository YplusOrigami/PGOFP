"""tanh / affine action chart. Entropy and Critic live in a_norm = tanh(z) coordinates."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def log_abs_det_tanh(z: torch.Tensor) -> torch.Tensor:
    """log|J_tanh(z)| = sum_i log(1 - tanh(z_i)^2) in the numerically stable form
    sum_i 2*(log 2 - z_i - softplus(-2 z_i)).  Returns shape z.shape[:-1]."""
    return (2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))).sum(-1)


def atanh_clamped(a_norm: torch.Tensor, eps: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Import an external bounded action into latent space.

    Returns (z, clamp_ratio) where clamp_ratio is the fraction of coordinates that hit the clamp.
    Only for importing external actions / controller centers; never used to hide policy saturation.
    """
    lim = 1.0 - eps
    clamped = a_norm.abs() > lim
    z = torch.atanh(a_norm.clamp(-lim, lim))
    return z, clamped.float().mean()


def to_physical(a_norm: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return bias + scale * a_norm


def to_normalized(a_phys: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (a_phys - bias) / scale


def log_abs_det_affine(scale: torch.Tensor) -> torch.Tensor:
    """sum log(c) -- subtract from a norm-coordinate log-density to get a physical-coordinate one."""
    return scale.abs().log().sum(-1)


def saturation_fraction(a_norm: torch.Tensor, thresh: float = 0.99) -> torch.Tensor:
    return (a_norm.abs() > thresh).float().mean()
