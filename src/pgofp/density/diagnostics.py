"""Diagnostics that never take part in Actor gradients: one-step vs multi-step gap, analytic references."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch


@torch.no_grad()
def generate_multistep(net, eps: torch.Tensor, cond: torch.Tensor, steps: int = 20, solver: str = "heun") -> torch.Tensor:
    """Integrate dz/dt = u(z,t,t) from t=1 (source) down to t=0 (action latent) with fixed steps."""
    z = eps
    h = 1.0 / steps
    B = eps.shape[0]

    def f(zz, t):
        tt = torch.full((B, 1), t, device=eps.device, dtype=eps.dtype)
        return net(zz, tt, tt, cond)

    for i in range(steps):
        t1 = 1.0 - i * h
        k1 = f(z, t1)
        if solver == "euler":
            z = z - h * k1
        else:
            k2 = f(z - h * k1, t1 - h)
            z = z - 0.5 * h * (k1 + k2)
    return z


@torch.no_grad()
def one_step_gap(net, eps: torch.Tensor, cond: torch.Tensor, steps: int = 20, critic=None) -> Dict[str, float]:
    """Compare the deployed one-step map with a multi-step integration of the diagonal field on shared eps."""
    z1 = net.one_step(eps, cond)
    zm = generate_multistep(net, eps, cond, steps=steps)
    a1, am = torch.tanh(z1), torch.tanh(zm)
    out = {
        "flow/one_step_gap_latent": float((z1 - zm).norm(dim=-1).mean()),
        "flow/one_step_gap_action": float((a1 - am).norm(dim=-1).mean()),
    }
    if critic is not None:
        out["flow/one_step_gap_q"] = float((critic.min_q(cond, a1) - critic.min_q(cond, am)).abs().mean())
    return out


def gaussian_logpdf(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return (-0.5 * ((x - mu) / std) ** 2 - std.log()).sum(-1) - 0.5 * d * math.log(2 * math.pi)
