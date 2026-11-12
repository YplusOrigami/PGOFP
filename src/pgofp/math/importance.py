"""Self-normalized importance weights. Normalization is ALWAYS per condition (over the candidate dim)."""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def candidate_weights(logw: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """logw [B, K] -> (w [B, K] softmax over K, diagnostics). Never normalizes across B."""
    if logw.dim() != 2:
        raise ValueError(f"logw must be [B,K], got {tuple(logw.shape)}")
    logw = logw.detach()
    finite = torch.isfinite(logw)
    safe = torch.where(finite, logw, torch.full_like(logw, float("-inf")))
    # if a whole row is non-finite fall back to uniform weights for that row (counted in finite_ratio)
    row_ok = finite.any(-1, keepdim=True)
    safe = torch.where(row_ok, safe, torch.zeros_like(safe))
    w = torch.softmax(safe, dim=-1)
    ess = 1.0 / (w**2).sum(-1)
    diag = {
        "ess": ess,
        "max_weight": w.max(-1).values,
        "finite_ratio": finite.float().mean(),
        "logw_std": torch.where(finite, logw, torch.zeros_like(logw)).std(-1),
    }
    return w, diag
