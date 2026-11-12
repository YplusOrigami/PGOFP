from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from .mlp import MLP


class TwinQ(nn.Module):
    """Two independent Q networks Q_i(condition, a_command_norm). Supports a [B,d] or candidates [B,K,d]."""

    def __init__(self, cond_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256, 256),
                 activation: str = "relu", layer_norm: bool = False):
        super().__init__()
        self.q1 = MLP(cond_dim + action_dim, hidden_dims, 1, activation=activation, layer_norm=layer_norm)
        self.q2 = MLP(cond_dim + action_dim, hidden_dims, 1, activation=activation, layer_norm=layer_norm)

    def forward(self, cond: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.dim() == 3:
            B, K, d = a.shape
            c = cond.unsqueeze(1).expand(B, K, cond.shape[-1])
            x = torch.cat([c, a], -1).reshape(B * K, -1)
            return self.q1(x).reshape(B, K), self.q2(x).reshape(B, K)
        x = torch.cat([cond, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, cond: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(cond, a)
        return torch.minimum(q1, q2)
