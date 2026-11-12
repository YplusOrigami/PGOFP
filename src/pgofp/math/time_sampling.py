"""Flow-time pair sampling (t, r) for MeanFlow training. t=0 is the target action, t=1 the source."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TimeSample:
    t: torch.Tensor  # [B, 1]
    r: torch.Tensor  # [B, 1], r <= t
    is_diag: torch.Tensor  # [B] bool, True where r == t (instantaneous-velocity samples)
    kind: str


def sample_time_pairs(
    batch: int,
    *,
    delta: float,
    flow_ratio: float,
    kind: str = "uniform",
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device="cpu",
    dtype=torch.float32,
) -> TimeSample:
    """t in [delta, 1-delta] (never the endpoints; the reverse kernel blows up like 1/(1-t)).

    With probability `flow_ratio` set r = t (diagonal / instantaneous), otherwise r = t * q, q ~ U(0,1).
    Inference always uses (r, t) = (0, 1); the interior-to-endpoint generalization is a real error source.
    """
    if not 0.0 < delta < 0.5:
        raise ValueError("delta must be in (0, 0.5)")
    if not 0.0 <= flow_ratio <= 1.0:
        raise ValueError("flow_ratio must be in [0, 1]")
    u = torch.rand(batch, 1, generator=generator, device=device, dtype=dtype)
    if kind == "uniform":
        t = delta + (1.0 - 2.0 * delta) * u
    elif kind == "logit_normal":
        n = torch.randn(batch, 1, generator=generator, device=device, dtype=dtype)
        t = torch.sigmoid(logit_mean + logit_std * n).clamp(delta, 1.0 - delta)
    else:
        raise ValueError(f"unknown time sampling kind {kind!r}")
    q = torch.rand(batch, 1, generator=generator, device=device, dtype=dtype)
    diag = torch.rand(batch, 1, generator=generator, device=device, dtype=dtype) < flow_ratio
    r = torch.where(diag, t, t * q)
    return TimeSample(t=t, r=r, is_diag=diag.squeeze(-1), kind=kind)
