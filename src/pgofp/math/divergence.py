"""Divergence (trace of the action Jacobian) estimators for the density path.

`hutchinson` returns v^T J v computed through a VJP:  grad(f . v, z) = J^T v, then (J^T v) . v = v^T J v.
(This is a VJP-based estimate, not a JVP.)  `exact` loops over action dims and is the correctness reference.
"""
from __future__ import annotations

from typing import List, Optional

import torch


def rademacher_like(z: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    b = torch.randint(0, 2, z.shape, generator=generator, device=z.device)
    return (b.to(z.dtype) * 2.0 - 1.0).detach()


def divergence_exact(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Exact trace(df/dz) per sample. f, z: [B, d] (graph from z to f required)."""
    d = z.shape[-1]
    tr = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    for i in range(d):
        g = torch.autograd.grad(f[:, i].sum(), z, retain_graph=True)[0]
        tr = tr + g[:, i]
    return tr


def divergence_hutchinson(f: torch.Tensor, z: torch.Tensor, probes: List[torch.Tensor]) -> torch.Tensor:
    """Mean over probes of v^T J v. Reusing the same probes along a trajectory keeps the estimator consistent."""
    acc = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    for k, v in enumerate(probes):
        jtv = torch.autograd.grad(f, z, grad_outputs=v, retain_graph=(k < len(probes) - 1))[0]
        acc = acc + (jtv * v).sum(-1)
    return acc / len(probes)
