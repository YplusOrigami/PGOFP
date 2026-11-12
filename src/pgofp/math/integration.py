"""Fixed-step Euler / Heun integration of the augmented ODE (z, D) with D = int div f dt.

Direction: integration runs from the TARGET end (t=0, the action latent) to the SOURCE end (t=1).
With dz/dt = f(z,t):  log p_target(z(0)) = log p_src(z(1)) + int_0^1 div_z f(z(t), t) dt.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

from .divergence import divergence_exact, divergence_hutchinson, rademacher_like


def integrate_with_divergence(
    field: Callable[[torch.Tensor, float], torch.Tensor],  # f(z, t_scalar) -> [B, d]
    z0: torch.Tensor,  # [B, d]  target-end latent
    *,
    steps: int,
    solver: str = "heun",
    trace: str = "hutchinson",
    n_trace_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (z(1), D, nfe). `nfe` counts field evaluations (Heun uses 2 per step)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be euler|heun")
    B = z0.shape[0]
    h = 1.0 / steps
    z = z0.detach()
    D = torch.zeros(B, device=z.device, dtype=z.dtype)
    probes: List[torch.Tensor] = []
    if trace == "hutchinson":
        probes = [rademacher_like(z, generator) for _ in range(n_trace_samples)]
    elif trace != "exact":
        raise ValueError("trace must be hutchinson|exact")

    def eval_at(zz: torch.Tensor, tt: float):
        with torch.enable_grad():
            zr = zz.detach().requires_grad_(True)
            f = field(zr, tt)
            div = divergence_exact(f, zr) if trace == "exact" else divergence_hutchinson(f, zr, probes)
        return f.detach(), div.detach()

    nfe = 0
    for i in range(steps):
        t0 = i * h
        k1, d1 = eval_at(z, t0)
        nfe += 1
        if solver == "euler":
            z = z + h * k1
            D = D + h * d1
        else:
            k2, d2 = eval_at(z + h * k1, t0 + h)
            nfe += 1
            z = z + 0.5 * h * (k1 + k2)
            D = D + 0.5 * h * (d1 + d2)
    return z.detach(), D.detach(), nfe
