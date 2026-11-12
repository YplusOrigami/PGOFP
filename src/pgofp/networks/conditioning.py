"""Assemble the flat condition vector consumed by Actor / Critic / density, from a ContextBatch.

The SAME builder instance is used for acting, current/next states in updates, density and evaluation, so
the condition protocol cannot diverge between code paths.

condition_mode
  public_state    public normalized observation (+ reference/phase if present) (+ controller memory features
                  if use_controller_features). The prior center/scale are NOT inputs: the prior only
                  changes the source distribution.
  prior_features  additionally feeds the prior centers/validity (and log-std for native priors) to the networks.
                  Use paired with a Gaussian-source twin to separate input-information effects from the
                  source-distribution effect (ablation A13).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from ..core.exceptions import ConfigError
from ..core.specs import SourceSpec
from ..core.types import ContextBatch, ContextSpec
from ..utils.normalization import NormSnapshot


@dataclass(frozen=True)
class ConditionConfig:
    mode: str = "public_state"  # public_state | prior_features
    use_reference: bool = True
    use_phase: bool = True
    use_controller_features: bool = False
    source_features: str = "auto"  # auto | on | off   (auto: on iff the schedule is not constant)
    reference_scale: float = 1.0


class ConditionBuilder:
    def __init__(self, ctx_spec: ContextSpec, cfg: ConditionConfig, schedule_is_constant: bool = True):
        if cfg.mode not in ("public_state", "prior_features"):
            raise ConfigError(f"unknown condition mode {cfg.mode!r}")
        if cfg.mode == "prior_features" and ctx_spec.n_prior == 0:
            raise ConfigError("condition.mode=prior_features needs a configured controller/prior")
        self.spec = ctx_spec
        self.cfg = cfg
        self.use_source_features = (
            cfg.source_features == "on" or (cfg.source_features == "auto" and not schedule_is_constant)
        )
        parts: List[int] = [ctx_spec.obs_dim]
        if cfg.use_reference:
            parts.append(ctx_spec.ref_dim)
        if cfg.use_phase:
            parts.append(ctx_spec.phase_dim)
        if cfg.use_controller_features:
            parts.append(ctx_spec.ctrl_dim)
        if cfg.mode == "prior_features":
            parts.append(ctx_spec.n_prior * ctx_spec.action_dim)  # tanh(prior center)
            parts.append(ctx_spec.n_prior)  # valid flags
        if self.use_source_features:
            parts.append(2)
        self.dim = int(sum(parts))

    def build(self, ctx: ContextBatch, norm: NormSnapshot, spec: SourceSpec) -> torch.Tensor:
        B = ctx.batch_size
        feats = [norm.normalize(ctx.obs_raw)]
        if self.cfg.use_reference:
            feats.append(ctx.reference * self.cfg.reference_scale)
        if self.cfg.use_phase:
            feats.append(ctx.phase)
        if self.cfg.use_controller_features:
            feats.append(ctx.ctrl_feat)
        if self.cfg.mode == "prior_features":
            feats.append(torch.tanh(ctx.prior_mu).reshape(B, -1))
            feats.append(ctx.prior_valid)
        if self.use_source_features:
            sf = torch.as_tensor(spec.feature_vector(), device=ctx.obs_raw.device, dtype=ctx.obs_raw.dtype)
            feats.append(sf.unsqueeze(0).expand(B, -1))
        out = torch.cat(feats, dim=-1)
        assert out.shape[-1] == self.dim, (out.shape, self.dim)
        return out
