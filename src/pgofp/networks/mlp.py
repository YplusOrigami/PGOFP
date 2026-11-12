from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

_ACT = {"silu": nn.SiLU, "relu": nn.ReLU, "mish": nn.Mish, "tanh": nn.Tanh, "gelu": nn.GELU}


def make_activation(name: str) -> nn.Module:
    if name not in _ACT:
        raise ValueError(f"unknown activation {name!r}; known {sorted(_ACT)}")
    return _ACT[name]()


class MLP(nn.Module):
    """Plain MLP; hidden blocks are Linear -> (LayerNorm) -> activation. Algorithm losses never live here."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: str = "silu",
        layer_norm: bool = False,
        zero_init_last: bool = False,
        orthogonal_gain: float | None = None,
        out_gain: float = 0.01,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            lin = nn.Linear(d, h)
            if orthogonal_gain is not None:
                nn.init.orthogonal_(lin.weight, gain=orthogonal_gain)
                nn.init.zeros_(lin.bias)
            layers.append(lin)
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(make_activation(activation))
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim)
        if zero_init_last:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif orthogonal_gain is not None:
            nn.init.orthogonal_(self.out.weight, gain=out_gain)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))
