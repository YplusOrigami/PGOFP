"""SAC (tanh-Gaussian, twin Q, auto temperature) and controller-only baselines under the same agent interface/protocol."""
from __future__ import annotations

import copy
import math
from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..core.specs import EnvSpec, SourceSpec
from ..core.types import ActionResult, ContextBatch, ReplayBatch, UpdateResult
from ..networks.conditioning import ConditionBuilder
from ..networks.critics import TwinQ
from ..networks.gaussian_policy import TanhGaussianPolicy
from ..priors.mixture import build_source
from ..utils.normalization import NormSnapshot
from ..utils.torch_utils import assert_finite, clip_grad_norm, ema_update, freeze
from .base import BaseAgent, UpdateContext
from .temperature import Temperature


class SACAgent(BaseAgent):
    def __init__(self, cfg, env_spec: EnvSpec, builder: ConditionBuilder, device: torch.device):
        self.cfg, self.a = cfg, cfg.algorithm
        self.env_spec, self.builder, self.device = env_spec, builder, device
        self.d = env_spec.action_dim
        hidden = tuple(dict(cfg.algorithm.params).get("hidden", [256, 256]))
        self.actor = TanhGaussianPolicy(builder.dim, self.d, hidden, "relu", state_dependent_std=True, orthogonal=False).to(device)
        self.critic = TwinQ(builder.dim, self.d, hidden, "relu").to(device)
        self.critic_target = freeze(copy.deepcopy(self.critic))
        o, t = cfg.optimizer, cfg.algorithm.temperature
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=o.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=o.critic_lr)
        self.temp = Temperature(t.mode, t.alpha, t.target_entropy, self.d, o.alpha_lr, t.alpha_min, t.alpha_max, device)
        self.max_grad_norm = cfg.train.max_grad_norm
        self.actor_updates = self.critic_updates = 0

    @torch.no_grad()
    def act(self, ctx: ContextBatch, spec: SourceSpec, norm: NormSnapshot, generator, *, deterministic: bool = False,
            K: int = 1) -> ActionResult:
        cond = self.builder.build(ctx.to(self.device), norm, spec)
        z, a, _ = self.actor.sample(cond, generator, deterministic)
        return ActionResult(action_norm=a, z=z, eps=z)

    def update(self, batch: ReplayBatch, uctx: UpdateContext) -> UpdateResult:
        rng, spec, norm, g = uctx.rng, uctx.spec, uctx.norm, self.a.gamma
        cond = self.builder.build(batch.ctx.to(self.device), norm, spec)
        cond_n = self.builder.build(batch.next_ctx.to(self.device), norm, spec)
        alpha = self.temp.alpha
        with torch.no_grad():
            z_n, a_n, _ = self.actor.sample(cond_n, rng.torch("target_source"))
            mu, ls = self.actor.dist_params(cond_n)
            logp_n = self.actor.log_prob_action(z_n, mu, ls)
            y = batch.reward + g * batch.bootstrap_mask * (self.critic_target.min_q(cond_n, a_n) - alpha * logp_n)
        q1, q2 = self.critic(cond, batch.action_command_norm)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        assert_finite("sac critic loss", c_loss)
        self.opt_critic.zero_grad(set_to_none=True)
        c_loss.backward()
        clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()
        self.critic_updates += 1
        m = {"critic/loss": float(c_loss), "critic/q1_mean": float(q1.mean()), "critic/target_mean": float(y.mean())}
        did = False
        if uctx.do_actor:
            did = True
            mu, ls = self.actor.dist_params(cond)
            z = mu + ls.exp() * torch.randn(mu.shape, generator=rng.torch("alpha_source"), device=self.device)
            a = torch.tanh(z)
            logp = self.actor.log_prob_action(z, mu, ls)
            for p in self.critic.parameters():
                p.requires_grad_(False)
            q = self.critic.min_q(cond, a)
            for p in self.critic.parameters():
                p.requires_grad_(True)
            a_loss = (alpha * logp - q).mean()
            self.opt_actor.zero_grad(set_to_none=True)
            a_loss.backward()
            clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
            self.opt_actor.step()
            self.actor_updates += 1
            H = (-logp.detach()).mean()
            m.update({"loss/actor": float(a_loss), "entropy/proxy_mean": float(H), "entropy/exact": 1.0})
            if self.temp.mode == "auto":
                m.update(self.temp.step(H))
        m["alpha"] = self.temp.alpha
        ema_update(self.critic_target, self.critic, self.a.target_tau)
        return UpdateResult(m, did)

    def state_dict(self) -> Dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
                "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
                "temperature": self.temp.state_dict(),
                "counters": {"actor_updates": self.actor_updates, "critic_updates": self.critic_updates, "clip_counts": {}}}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])
        self.temp.load_state_dict(sd["temperature"])
        self.actor_updates, self.critic_updates = sd["counters"]["actor_updates"], sd["counters"]["critic_updates"]


class ControllerOnlyAgent(BaseAgent):
    """No learning. mode=deterministic: tanh(center of component 0); mode=noisy_source: sample the configured source
    (the zero-velocity flow = noisy-source-only control, isolating randomization from flow improvement)."""

    def __init__(self, cfg, env_spec: EnvSpec, builder: ConditionBuilder, device: torch.device):
        self.mode = dict(cfg.algorithm.params).get("mode", "deterministic")
        if self.mode not in ("deterministic", "noisy_source"):
            raise ValueError("controller_only mode must be deterministic|noisy_source")
        self.d, self.device = env_spec.action_dim, device
        self.actor_updates = self.critic_updates = 0

    @torch.no_grad()
    def act(self, ctx: ContextBatch, spec: SourceSpec, norm: NormSnapshot, generator, *, deterministic: bool = False,
            K: int = 1) -> ActionResult:
        ctx = ctx.to(self.device)
        if self.mode == "deterministic" or deterministic:
            z = ctx.prior_mu[:, 0] if ctx.prior_mu.shape[1] else torch.zeros(len(ctx), self.d, device=self.device)
        else:
            z = build_source(spec, ctx, self.d).sample((), generator)
        return ActionResult(action_norm=torch.tanh(z), z=z, eps=z)

    def update(self, batch, uctx) -> UpdateResult:
        return UpdateResult({}, False)

    def state_dict(self) -> Dict[str, Any]:
        return {"counters": {"actor_updates": 0, "critic_updates": 0, "clip_counts": {}}}

    def load_state_dict(self, sd) -> None:
        pass
