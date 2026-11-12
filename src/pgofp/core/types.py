"""Core data types shared across layers.

Naming (docs/algorithm_contract.md):
  z        flow latent (pre-tanh); source sample lives at flow time t=1, target action at t=0
  a_norm   tanh(z) in (-1, 1)^d   -- the action coordinate for Critic / entropy
  a_phys   bias + scale * a_norm  -- what the task adapter receives
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Fields that describe "the condition c" a policy/critic/source depends on.
CONTEXT_FIELDS = ("obs_raw", "reference", "ctrl_feat", "phase", "prior_mu", "prior_logstd", "prior_valid")


@dataclass(frozen=True)
class ContextSpec:
    """Per-sample shapes of every context field (zero-width fields allowed)."""

    obs_dim: int
    action_dim: int
    ref_dim: int = 0
    ctrl_dim: int = 0
    phase_dim: int = 0
    n_prior: int = 0  # number of frozen prior components M (0 = no controller)

    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "obs_raw": (self.obs_dim,),
            "reference": (self.ref_dim,),
            "ctrl_feat": (self.ctrl_dim,),
            "phase": (self.phase_dim,),
            "prior_mu": (self.n_prior, self.action_dim),
            "prior_logstd": (self.n_prior, self.action_dim),
            "prior_valid": (self.n_prior,),
        }


@dataclass
class ContextBatch:
    """Torch batch of conditions. All tensors share leading batch dim B."""

    obs_raw: torch.Tensor  # [B, d_o]
    reference: torch.Tensor  # [B, d_r]
    ctrl_feat: torch.Tensor  # [B, d_c]
    phase: torch.Tensor  # [B, d_p]
    prior_mu: torch.Tensor  # [B, M, d_a]   pre-tanh centers (frozen controller cache)
    prior_logstd: torch.Tensor  # [B, M, d_a]   pre-tanh log-std (only used if prior_std_mode == native)
    prior_valid: torch.Tensor  # [B, M]        1 = controller output valid, 0 = fallback to base

    def __len__(self) -> int:
        return self.obs_raw.shape[0]

    @property
    def device(self) -> torch.device:
        return self.obs_raw.device

    @property
    def batch_size(self) -> int:
        return self.obs_raw.shape[0]

    def to(self, device) -> "ContextBatch":
        return ContextBatch(**{k: getattr(self, k).to(device) for k in CONTEXT_FIELDS})

    def select(self, idx) -> "ContextBatch":
        return ContextBatch(**{k: getattr(self, k)[idx] for k in CONTEXT_FIELDS})

    @staticmethod
    def from_numpy(d: Dict[str, np.ndarray], device="cpu", prefix: str = "") -> "ContextBatch":
        return ContextBatch(
            **{k: torch.as_tensor(d[prefix + k], dtype=torch.float32, device=device) for k in CONTEXT_FIELDS}
        )


@dataclass
class ControllerPrediction:
    """Output of a (pure) controller query.

    center_norm:  [B, M, d_a] suggested action in normalized (tanh) coordinates, or
    center_latent [B, M, d_a] pre-tanh centers if the controller natively lives in latent coordinates.
    log_std_latent: [B, M, d_a] or None (deterministic controllers).
    valid: [B, M] bool.
    """

    center_norm: Optional[np.ndarray] = None
    center_latent: Optional[np.ndarray] = None
    log_std_latent: Optional[np.ndarray] = None
    valid: Optional[np.ndarray] = None
    latency_s: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    version: str = ""


@dataclass
class ActionResult:
    action_norm: torch.Tensor  # [B, d_a]  tanh(z)
    z: torch.Tensor  # [B, d_a]
    eps: torch.Tensor  # [B, d_a]  source sample
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DensityResult:
    log_prob_norm: torch.Tensor  # [B]  log pi(a_norm | c)   (proxy unless is_exact_for_sampled_policy)
    method: str
    is_exact_for_sampled_policy: bool
    nfe: int
    inverse_residual: Optional[torch.Tensor] = None  # [B]
    finite_mask: Optional[torch.Tensor] = None  # [B] bool
    log_prob_latent: Optional[torch.Tensor] = None
    eps_hat: Optional[torch.Tensor] = None


@dataclass
class ReplayBatch:
    ctx: ContextBatch
    next_ctx: ContextBatch
    action_policy_norm: torch.Tensor  # [B, d_a]
    action_command_norm: torch.Tensor  # [B, d_a]  what the env actually received (Critic input)
    reward: torch.Tensor  # [B]
    terminated: torch.Tensor  # [B] float
    truncated: torch.Tensor  # [B] float
    bootstrap_mask: torch.Tensor  # [B] float
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.reward.shape[0]


@dataclass
class MetricRecord:
    step: int  # env_transitions: the single x-axis for all backends
    scalars: Dict[str, float]
    wall_time: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateResult:
    metrics: Dict[str, float]
    did_actor_update: bool = False
