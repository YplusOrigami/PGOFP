"""MeanFlow identity target -- the single implementation (never duplicated elsewhere).

    u(z_t, r, t) = v - (t - r) * ( J_z u . v + d_t u ),      r fixed
so the regression target is  v - (t-r) * D_u  with D_u the JVP of u along the tangent (v, 0, 1) on (z, r, t).
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.func import jvp


def meanflow_jvp(
    u_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    z_t: torch.Tensor,  # [N, d]
    r: torch.Tensor,  # [N, 1]
    t: torch.Tensor,  # [N, 1]
    v: torch.Tensor,  # [N, d]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (u(z_t,r,t), D_u). Condition/prior/params must be closed over by `u_fn` and are not differentiated."""
    tangents = (v, torch.zeros_like(r), torch.ones_like(t))
    return jvp(u_fn, (z_t, r, t), tangents)


@torch.no_grad()
def meanflow_target(u_fn, z_t, r, t, v) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully detached target and the JVP norm for diagnostics.

    No second-order graph is created: everything runs under no_grad, so the target carries no gradient
    to the network parameters, to the candidates/weights, or to any Critic / controller.
    """
    _, du = meanflow_jvp(u_fn, z_t, r, t, v)
    target = v - (t - r) * du
    return target.detach(), du.detach().norm(dim=-1)
