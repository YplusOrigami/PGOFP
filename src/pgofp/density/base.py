"""Density estimator interface and result semantics.

A DensityResult is a *proxy* unless `is_exact_for_sampled_policy` is True. The deployed policy is the one-step map
z = eps - u(eps,0,1); the continuous-flow density (diagonal field u(z,t,t)) only approximates it, and more
integration steps reduce numerical error, not the one-step/continuous policy gap.
"""
from __future__ import annotations

from typing import Optional, Protocol

import torch

from ..core.types import DensityResult


class Estimator(Protocol):
    method: str

    def estimate_at(self, z_a: torch.Tensor, cond: torch.Tensor, net, source, generator: Optional[torch.Generator] = None,
                    eps_ref: Optional[torch.Tensor] = None) -> DensityResult: ...


def check_finite(result: DensityResult, where: str) -> None:
    """Non-finite densities must stop the update (never silently zeroed)."""
    from ..core.exceptions import NumericalError

    if result.finite_mask is not None and not bool(result.finite_mask.all()):
        bad = int((~result.finite_mask).sum())
        raise NumericalError(f"{bad}/{result.finite_mask.numel()} non-finite density values in {where}")
