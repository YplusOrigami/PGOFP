"""Flow-time embeddings (r, t). Flow time is NOT the control/environment time."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """[B,1] in [0,1] -> [B,dim]; frequencies are geometric in [2*pi, 2*pi*max_freq]."""

    def __init__(self, dim: int = 32, max_freq: float = 4.0):
        super().__init__()
        if dim % 2:
            raise ValueError("embedding dim must be even")
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half)) * 2.0 * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t * self.freqs.to(t.dtype)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class PhaseEmbedding(nn.Module):
    """Optional episode-phase (remaining-time fraction) encoding for finite-horizon tasks."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.emb = SinusoidalEmbedding(dim, max_freq=16.0)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return self.emb(phase[..., :1])
