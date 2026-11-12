"""Running observation statistics with immutable snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


@dataclass(frozen=True)
class NormSnapshot:
    mean: torch.Tensor
    std: torch.Tensor
    clip: float
    version: int

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 0:
            return x
        out = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return out.clamp(-self.clip, self.clip)

    def to(self, device) -> "NormSnapshot":
        return NormSnapshot(self.mean.to(device), self.std.to(device), self.clip, self.version)


class RunningNorm:
    """Parallel (Chan et al.) running mean/var. Updated by training env only; eval uses a frozen snapshot."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-4, enabled: bool = True):
        self.dim = dim
        self.clip = clip
        self.eps = eps
        self.enabled = enabled
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps
        self.version = 0

    def update(self, x: np.ndarray) -> None:
        if not self.enabled or self.dim == 0:
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        n = x.shape[0]
        if n == 0:
            return
        bm, bv = x.mean(0), x.var(0)
        delta = bm - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        m2 = self.var * self.count + bv * n + delta**2 * self.count * n / tot
        self.var = m2 / tot
        self.count = tot
        self.version += 1

    def snapshot(self, device="cpu") -> NormSnapshot:
        if not self.enabled:
            mean = torch.zeros(self.dim)
            std = torch.ones(self.dim)
        else:
            mean = torch.as_tensor(self.mean, dtype=torch.float32)
            std = torch.as_tensor(np.sqrt(self.var + 1e-8), dtype=torch.float32)
        return NormSnapshot(mean.to(device), std.to(device), self.clip, self.version)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count,
                "version": self.version, "dim": self.dim, "enabled": self.enabled}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if sd["dim"] != self.dim:
            raise ValueError(f"normalizer dim mismatch {sd['dim']} vs {self.dim}")
        self.mean, self.var, self.count = sd["mean"].copy(), sd["var"].copy(), sd["count"]
        self.version = sd["version"]
