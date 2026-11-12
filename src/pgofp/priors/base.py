"""Conditional source distributions over the flow latent z (pre-tanh).

Every supported source (standard Gaussian, controller+perturbation, frozen PPO, controller mixtures and
prior/Gaussian mixtures, kernelized particle sets) is a diagonal Gaussian mixture with per-sample
components, so `sample` and `log_prob` always refer to the SAME actual distribution (including fallback).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class GaussianMixtureSource:
    """p(z) = sum_m w_m N(mu_m, diag(std_m^2)), batched over conditions.

    means, log_stds: [B, M, d];  log_weights: [B, M] (log of normalized weights; -inf allowed for dropped comps).
    `sample(sample_shape)` returns [B, *sample_shape, d]; `log_prob(z)` accepts [B, *S, d] -> [B, *S].
    The log-density is the full mixture density (logsumexp over components), never just the chosen component.
    """

    def __init__(self, means: torch.Tensor, log_stds: torch.Tensor, log_weights: torch.Tensor):
        if means.dim() != 3 or means.shape != log_stds.shape or log_weights.shape != means.shape[:2]:
            raise ValueError(
                f"bad shapes means {tuple(means.shape)} log_stds {tuple(log_stds.shape)} "
                f"log_weights {tuple(log_weights.shape)}"
            )
        self.means = means
        self.log_stds = log_stds
        self.log_weights = log_weights
        self.last_component: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ shape helpers
    @property
    def batch_size(self) -> int:
        return self.means.shape[0]

    @property
    def n_components(self) -> int:
        return self.means.shape[1]

    @property
    def dim(self) -> int:
        return self.means.shape[2]

    @property
    def device(self):
        return self.means.device

    def index(self, idx) -> "GaussianMixtureSource":
        return GaussianMixtureSource(self.means[idx], self.log_stds[idx], self.log_weights[idx])

    # ------------------------------------------------------------------ sampling
    def sample(self, sample_shape: Tuple[int, ...] = (), generator: Optional[torch.Generator] = None) -> torch.Tensor:
        B, M, d = self.means.shape
        S = int(math.prod(sample_shape)) if len(sample_shape) else 1
        noise = torch.randn(B, S, d, generator=generator, device=self.device, dtype=self.means.dtype)
        if M == 1:
            comp = torch.zeros(B, S, dtype=torch.long, device=self.device)
            mu = self.means[:, :1].expand(B, S, d)
            ls = self.log_stds[:, :1].expand(B, S, d)
        else:
            w = torch.softmax(self.log_weights, dim=-1)
            comp = torch.multinomial(w, S, replacement=True, generator=generator)  # [B, S]
            gi = comp.unsqueeze(-1).expand(B, S, d)
            mu = torch.gather(self.means, 1, gi)
            ls = torch.gather(self.log_stds, 1, gi)
        self.last_component = comp
        z = mu + ls.exp() * noise
        return z.reshape(B, *sample_shape, d)

    # ------------------------------------------------------------------ density
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        B, M, d = self.means.shape
        if z.shape[0] != B or z.shape[-1] != d:
            raise ValueError(f"z {tuple(z.shape)} incompatible with source [B={B}, d={d}]")
        extra = z.shape[1:-1]
        zf = z.reshape(B, -1, d).unsqueeze(2)  # [B, S, 1, d]
        mu = self.means.unsqueeze(1)  # [B, 1, M, d]
        ls = self.log_stds.unsqueeze(1)
        lp = -0.5 * (((zf - mu) / ls.exp()) ** 2).sum(-1) - ls.sum(-1) - 0.5 * d * math.log(2.0 * math.pi)  # [B,S,M]
        out = torch.logsumexp(lp + self.log_weights.unsqueeze(1), dim=-1)  # [B, S]
        return out.reshape(B, *extra)

    # ------------------------------------------------------------------ misc
    def representative(self) -> torch.Tensor:
        """Mean of the highest-weight component [B, d] (only a `representative_action` diagnostic)."""
        idx = self.log_weights.argmax(-1)
        return self.means[torch.arange(self.batch_size, device=self.device), idx]

    def component_usage(self) -> Optional[torch.Tensor]:
        """Mean one-hot usage per component of the last sample call: [M]."""
        if self.last_component is None:
            return None
        oh = torch.nn.functional.one_hot(self.last_component, self.n_components).float()
        return oh.mean((0, 1))
