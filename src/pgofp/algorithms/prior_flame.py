"""Main agent: prior_flame_m (and gaussian_flame_m -- the SAME class with a Gaussian SourceSpec).

Update order (docs section 8.1): critic -> actor (every `actor_update_interval`) -> alpha -> target EMA, all with one
SourceSpec / normalizer snapshot.  Gradient contracts:
  * Critic update: target side (target actor, entropy density, target Q) is fully stop-gradient.
  * Actor update: Critic is not differentiated; candidates, weights and JVP targets are no_grad; only u_pred has grad.
  * Alpha update: uses the ONLINE actor's samples/density, detached; gradient only reaches log_alpha.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch

from ..core.exceptions import ConfigError, NumericalError
from ..core.specs import EnvSpec, SourceSpec
from ..core.types import ActionResult, ContextBatch, ReplayBatch, UpdateResult
from ..density import check_finite, make_estimator
from ..density.diagnostics import one_step_gap
from ..networks.conditioning import ConditionBuilder
from ..networks.critics import TwinQ
from ..networks.mean_flow import MeanFlowNet
from ..priors.mixture import build_source
from ..utils.normalization import NormSnapshot
from ..utils.torch_utils import assert_finite, clip_grad_norm, ema_update, freeze
from .base import BaseAgent, UpdateContext
from .objectives import FlowSettings, actor_loss_endpoint_is, actor_loss_reverse, soft_critic_loss
from .temperature import Temperature


class PriorFlameAgent(BaseAgent):
    def __init__(self, cfg, env_spec: EnvSpec, builder: ConditionBuilder, device: torch.device):
        self.cfg = cfg
        a = cfg.algorithm
        self.a = a
        self.env_spec = env_spec
        self.builder = builder
        self.device = device
        self.d = env_spec.action_dim
        n = cfg.network
        self.actor = MeanFlowNet(self.d, builder.dim, tuple(n.actor_hidden), n.time_embed_dim, n.actor_activation,
                                 n.layer_norm, n.time_max_freq).to(device)
        self.actor_target = freeze(copy.deepcopy(self.actor))
        self.critic = TwinQ(builder.dim, self.d, tuple(n.critic_hidden), n.critic_activation, n.layer_norm).to(device)
        self.critic_target = freeze(copy.deepcopy(self.critic))
        o = cfg.optimizer
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=o.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=o.critic_lr)
        t = a.temperature
        self.temp = Temperature(t.mode, t.alpha, t.target_entropy, self.d, o.alpha_lr, t.alpha_min, t.alpha_max, device)
        dcfg = a.density
        self.estimator = make_estimator(
            dcfg.method, steps=dcfg.steps, solver=dcfg.solver, trace=dcfg.trace, n_trace_samples=dcfg.n_trace_samples,
            chunk_size=dcfg.chunk_size, grid=tuple(dcfg.grid), logdet=dcfg.logdet)
        self.flow = FlowSettings(a.flow.time_delta, a.flow.flow_ratio, a.flow.time_sampling, a.flow.logit_mean,
                                 a.flow.logit_std)
        self.max_grad_norm = cfg.train.max_grad_norm
        self.actor_updates = 0
        self.critic_updates = 0
        self._last_clip = {"actor": 0, "critic": 0}
        self._clip_counts = {"actor": 0, "critic": 0}
        self.scope = set(a.source_scope)
        bad = self.scope - {"behavior", "actor", "target"}
        if bad:
            raise ConfigError(f"unknown source_scope entries {sorted(bad)}")
        if a.actor_objective not in ("reverse_qrmf", "endpoint_is", "reverse_no_q", "reverse_uniform"):
            raise ConfigError(f"unknown actor_objective {a.actor_objective!r}")

    # ------------------------------------------------------------------ source placement (ablation A9)
    def spec_for(self, place: str, spec: SourceSpec) -> SourceSpec:
        """Where the prior source is used. Non-listed places fall back to the plain Gaussian (diagnostic only)."""
        if place in self.scope or spec.kind == "gaussian":
            return spec
        return SourceSpec(kind="gaussian", sigma_base=spec.sigma_base, version=spec.version)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, ctx: ContextBatch, spec: SourceSpec, norm: NormSnapshot, generator, *, deterministic: bool = False,
            K: int = 1) -> ActionResult:
        """One source sample + one flow forward (K=1). K>1 is the separately-tagged best-of-K behavior variant."""
        spec = self.spec_for("behavior", spec)
        ctx = ctx.to(self.device)
        cond = self.builder.build(ctx, norm, spec)
        src = build_source(spec, ctx, self.d)
        info: Dict[str, Any] = {}
        if deterministic:
            eps = src.representative()
            z = self.actor.one_step(eps, cond)
        elif K <= 1:
            eps = src.sample((), generator)
            z = self.actor.one_step(eps, cond)
            usage = src.component_usage()
            if usage is not None and usage.numel() > 1:
                info["component_usage"] = usage.cpu().numpy()
        else:
            B = cond.shape[0]
            eps_k = src.sample((K,), generator)  # [B,K,d]
            cond_k = cond.unsqueeze(1).expand(B, K, cond.shape[-1]).reshape(B * K, -1)
            z_k = self.actor.one_step(eps_k.reshape(B * K, self.d), cond_k).reshape(B, K, self.d)
            q = self.critic.min_q(cond, torch.tanh(z_k))
            best = q.argmax(-1)
            ar = torch.arange(B, device=self.device)
            z, eps = z_k[ar, best], eps_k[ar, best]
            info["best_of_k"] = K
        return ActionResult(action_norm=torch.tanh(z), z=z, eps=eps, info=info)

    # ------------------------------------------------------------------ update
    def update(self, batch: ReplayBatch, uctx: UpdateContext) -> UpdateResult:
        a = self.a
        spec, norm, rng = uctx.spec, uctx.norm, uctx.rng
        ctx, nctx = batch.ctx.to(self.device), batch.next_ctx.to(self.device)
        cond = self.builder.build(ctx, norm, spec)
        cond_next = self.builder.build(nctx, norm, spec)
        a_cmd = batch.action_command_norm
        alpha = self.temp.alpha
        metrics: Dict[str, float] = {}

        # ---- Critic (target side fully detached)
        src_next = build_source(self.spec_for("target", spec), nctx, self.d)
        c_loss, cm = soft_critic_loss(
            self.critic, self.critic_target, self.actor_target, self.estimator,
            cond=cond, cond_next=cond_next, source_next=src_next, a_cmd=a_cmd, reward=batch.reward,
            bootstrap_mask=batch.bootstrap_mask, gamma=a.gamma, alpha=alpha if a.use_entropy else 0.0,
            use_entropy=a.use_entropy, rng=rng)
        assert_finite("critic loss", c_loss)
        self.opt_critic.zero_grad(set_to_none=True)
        c_loss.backward()
        gn, clipped = clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()
        self.critic_updates += 1
        self._clip_counts["critic"] += int(clipped)
        metrics.update(cm)
        metrics["critic/grad_norm"] = gn
        metrics["critic/clip_rate"] = self._clip_counts["critic"] / self.critic_updates

        did_actor = False
        if uctx.do_actor:
            did_actor = True
            src = build_source(self.spec_for("actor", spec), ctx, self.d)
            for p in self.critic.parameters():
                p.requires_grad_(False)
            try:
                if a.actor_objective == "endpoint_is":
                    a_loss, am = actor_loss_endpoint_is(self.actor, self.critic, cond, src, alpha, K=a.K_train,
                                                         flow=self.flow, rng=rng)
                else:
                    a_rep = None
                    if a.zt_proposal == "replay_action":
                        from ..math.action_transform import atanh_clamped

                        a_rep, _ = atanh_clamped(batch.action_policy_norm, a.atanh_eps)
                    a_loss, am = actor_loss_reverse(self.actor, self.critic, cond, src, alpha, K=a.K_train,
                                                    flow=self.flow, objective=a.actor_objective,
                                                    zt_proposal=a.zt_proposal, a_replay_latent=a_rep, rng=rng,
                                                    K_local=a.K_local, sigma_local=a.sigma_local)
            finally:
                for p in self.critic.parameters():
                    p.requires_grad_(True)
            assert_finite("actor loss", a_loss)
            self.opt_actor.zero_grad(set_to_none=True)
            a_loss.backward()
            gn, clipped = clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
            self.opt_actor.step()
            self.actor_updates += 1
            self._clip_counts["actor"] += int(clipped)
            metrics.update(am)
            metrics["actor/grad_norm"] = gn
            metrics["actor/clip_rate"] = self._clip_counts["actor"] / self.actor_updates

            # ---- alpha: ONLINE actor samples + online density, detached
            if self.temp.mode == "auto" or uctx.log_step:
                with torch.no_grad():
                    eps = src.sample((), rng.torch("alpha_source"))
                    z = self.actor.one_step(eps, cond)
                    dres = self.estimator.estimate_at(z, cond, self.actor, src, rng.torch("trace"), eps_ref=eps)
                    check_finite(dres, "alpha entropy")
                    H = (-dres.log_prob_norm).mean()
                    metrics["entropy/proxy_mean"] = float(H)
                    metrics["entropy/target_entropy"] = self.temp.target_entropy
                    metrics["entropy/nfe"] = float(dres.nfe)
                    if dres.inverse_residual is not None:
                        metrics["entropy/inverse_residual"] = float(dres.inverse_residual.mean())
                    metrics["actions/saturation_frac"] = float((torch.tanh(z).abs() > 0.99).float().mean())
                    metrics["actions/norm_mean"] = float(torch.tanh(z).norm(dim=-1).mean())
                    if uctx.log_step:
                        metrics.update(one_step_gap(self.actor, eps[: min(256, eps.shape[0])], cond[: min(256, cond.shape[0])],
                                                    steps=8, critic=self.critic))
                if self.temp.mode == "auto":
                    metrics.update({f"loss/{k}" if k == "alpha_loss" else k: v for k, v in self.temp.step(H).items()})
        metrics["alpha"] = self.temp.alpha

        # ---- target EMA (target_tau weight on the NEW parameters)
        ema_update(self.critic_target, self.critic, a.target_tau)
        if did_actor:
            ema_update(self.actor_target, self.actor, a.target_tau)
        return UpdateResult(metrics=metrics, did_actor_update=did_actor)

    # ------------------------------------------------------------------ state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(), "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "temperature": self.temp.state_dict(),
            "counters": {"actor_updates": self.actor_updates, "critic_updates": self.critic_updates,
                         "clip_counts": dict(self._clip_counts)},
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])
        self.temp.load_state_dict(sd["temperature"])
        c = sd["counters"]
        self.actor_updates, self.critic_updates = c["actor_updates"], c["critic_updates"]
        self._clip_counts = dict(c["clip_counts"])

    def policy_weights(self) -> Dict[str, Any]:
        """Weights only (NOT a resumable checkpoint)."""
        return {"actor": self.actor.state_dict()}
