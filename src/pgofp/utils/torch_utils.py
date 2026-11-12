from __future__ import annotations

import contextlib
from typing import Iterable, Tuple

import torch
from torch import nn


def select_device(name: str = "auto") -> torch.device:
    """`mps` is deliberately not auto-selected: JVP/density are not validated there."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


@contextlib.contextmanager
def frozen_params(module: nn.Module):
    """Temporarily disable parameter grads (e.g. Critic during Actor update)."""
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield module
    finally:
        for p, f in zip(module.parameters(), flags):
            p.requires_grad_(f)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """target <- (1 - tau) * target + tau * online   (tau = target_tau, e.g. 0.005)."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(1.0 - tau).add_(po.detach(), alpha=tau)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)


def clip_grad_norm(params: Iterable[torch.Tensor], max_norm: float) -> Tuple[float, bool]:
    """Returns (pre-clip norm, was_clipped)."""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0, False
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    n = float(norm)
    return n, n > max_norm


def grad_norm(params: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total**0.5


def assert_finite(name: str, *tensors: torch.Tensor) -> None:
    from ..core.exceptions import NumericalError

    for t in tensors:
        if not torch.isfinite(t).all():
            raise NumericalError(f"non-finite values in {name}")
