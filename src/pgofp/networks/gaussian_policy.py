"""tanh-Gaussian policy (PPO prior / SAC baseline) in the pre-tanh latent space."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import nn

from ..math.action_transform import log_abs_det_tanh
from .mlp import MLP

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class TanhGaussianPolicy(nn.Module):
    """z ~ N(mu(s), std^2), a_norm = tanh(z).

    state_dependent_std=False (PPO): log_std is a free parameter, mu has a small final init.
    state_dependent_std=True (SAC): the network also outputs log_std.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (64, 64), activation: str = "tanh",
                 state_dependent_std: bool = False, init_log_std: float = -0.5, orthogonal: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        out = action_dim * (2 if state_dependent_std else 1)
        self.net = MLP(obs_dim, hidden_dims, out, activation=activation,
                       orthogonal_gain=math.sqrt(2) if orthogonal else None)
        if not state_dependent_std:
            self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        if self.state_dependent_std:
            mu, ls = h.chunk(2, -1)
        else:
            mu, ls = h, self.log_std.expand_as(h)
        return mu, ls.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor, generator=None, deterministic: bool = False):
        mu, ls = self.dist_params(obs)
        if deterministic:
            z = mu
        else:
            z = mu + ls.exp() * torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return z, torch.tanh(z), self.log_prob_z(z, mu, ls)

    @staticmethod
    def log_prob_z(z: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        return (-0.5 * ((z - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

    def log_prob_action(self, z, mu, log_std) -> torch.Tensor:
        """log density in a_norm coordinates (adds -log|J_tanh|)."""
        return self.log_prob_z(z, mu, log_std) - log_abs_det_tanh(z)

    @staticmethod
    def entropy_z(log_std: torch.Tensor) -> torch.Tensor:
        return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int] = (64, 64), activation: str = "tanh"):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dims, 1, activation=activation, orthogonal_gain=math.sqrt(2), out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
