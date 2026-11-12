"""Average-velocity network u_theta(z, r, t; c) and the one-step policy wrapper.

Convention: t=0 target action, t=1 source sample.  Inference: z_out = eps - u(eps, 0, 1; c).
The last layer is zero-initialized, so at initialization the policy is exactly tanh(source sample).
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .embeddings import SinusoidalEmbedding
from .mlp import MLP


class MeanFlowNet(nn.Module):
    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        time_embed_dim: int = 32,
        activation: str = "silu",
        layer_norm: bool = False,
        time_max_freq: float = 4.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        # Low max frequency on purpose: high-frequency time features make d_t u (the JVP term of the MeanFlow target) explode.
        self.embed_r = SinusoidalEmbedding(time_embed_dim, time_max_freq)
        self.embed_t = SinusoidalEmbedding(time_embed_dim, time_max_freq)
        self.net = MLP(action_dim + cond_dim + 2 * time_embed_dim, hidden_dims, action_dim,
                       activation=activation, layer_norm=layer_norm, zero_init_last=True)

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z [N,d], r [N,1], t [N,1], cond [N,dc] -> average velocity [N,d] (all shapes asserted: no [B]x[B,1] broadcast)."""
        if z.dim() != 2 or r.shape != (z.shape[0], 1) or t.shape != r.shape or cond.shape[0] != z.shape[0]:
            raise ValueError(f"bad shapes z{tuple(z.shape)} r{tuple(r.shape)} t{tuple(t.shape)} cond{tuple(cond.shape)}")
        x = torch.cat([z, cond, self.embed_r(r), self.embed_t(t)], dim=-1)
        return self.net(x)

    def diag_field(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Instantaneous velocity field f(z, t) = u(z, t, t)."""
        return self(z, t, t, cond)

    def one_step(self, eps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z_out = eps - u(eps, r=0, t=1; c)."""
        n = eps.shape[0]
        zeros = torch.zeros(n, 1, device=eps.device, dtype=eps.dtype)
        return eps - self(eps, zeros, torch.ones_like(zeros), cond)
