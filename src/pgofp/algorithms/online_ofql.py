"""Online-OFQL baseline (docs: Online-OFQL设计.md): FBC + differentiable-Q guidance. NOT FLAME.

  L_actor = L_FBC(theta) - alpha_q * E[Q(c, tanh z_gen)],   alpha_q = eta / EMA(|Q|)
  L_FBC   : MeanFlow regression toward REPLAY actions (z_a = atanh(a_replay), eps ~ N(0, I), z_t = (1-t) z_a + t eps)
  critic  : y = r + gamma * mask * min Q'(c', a'),  a' from the target one-step policy (no entropy term)
  explore : external Gaussian noise on the executed action (annealed)
Uses its own coefficients (eta / q_ema / explore_*), never the alpha_ent entropy temperature.
Source is always the standard Gaussian (the OFQL definition); the SourceSpec argument is accepted but ignored.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..core.specs import EnvSpec, SourceSpec
from ..core.types import ActionResult, ContextBatch, ReplayBatch, UpdateResult
from ..math.action_transform import atanh_clamped
from ..math.meanflow_target import meanflow_target
from ..math.time_sampling import sample_time_pairs
from ..networks.conditioning import ConditionBuilder
from ..networks.critics import TwinQ
from ..networks.mean_flow import MeanFlowNet
from ..utils.normalization import NormSnapshot
from ..utils.torch_utils import assert_finite, clip_grad_norm, ema_update, freeze
from .base import BaseAgent, UpdateContext


class OnlineOFQLAgent(BaseAgent):
    def __init__(self, cfg, env_spec: EnvSpec, builder: ConditionBuilder, device: torch.device):
        self.cfg, self.a = cfg, cfg.algorithm
        self.env_spec, self.builder, self.device = env_spec, builder, device
        self.d = env_spec.action_dim
        n, p = cfg.network, dict(cfg.algorithm.params)
        self.eta = float(p.get("eta", 0.1))
        self.q_ema_decay = float(p.get("q_ema", 0.99))
        self.sig0 = float(p.get("explore_sigma", 0.1))
        self.sig1 = float(p.get("explore_sigma_final", 0.02))
        self.sig_steps = float(p.get("explore_anneal_steps", 200_000))
        self.actor = MeanFlowNet(self.d, builder.dim, tuple(n.actor_hidden), n.time_embed_dim, n.actor_activation, n.layer_norm, n.time_max_freq).to(device)
        self.actor_target = freeze(copy.deepcopy(self.actor))
        self.critic = TwinQ(builder.dim, self.d, tuple(n.critic_hidden), n.critic_activation, n.layer_norm).to(device)
        self.critic_target = freeze(copy.deepcopy(self.critic))
        o = cfg.optimizer
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=o.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=o.critic_lr)
        self.q_abs_ema = None
        self.max_grad_norm = cfg.train.max_grad_norm
        self.actor_updates = self.critic_updates = 0
        self._progress = 0

    def _sigma(self) -> float:
        frac = min(1.0, self._progress / max(self.sig_steps, 1.0))
        return self.sig0 + frac * (self.sig1 - self.sig0)

    @torch.no_grad()
    def act(self, ctx: ContextBatch, spec: SourceSpec, norm: NormSnapshot, generator, *, deterministic: bool = False,
            K: int = 1) -> ActionResult:
        ctx = ctx.to(self.device)
        cond = self.builder.build(ctx, norm, spec)
        eps = torch.zeros(cond.shape[0], self.d, device=self.device) if deterministic else \
            torch.randn(cond.shape[0], self.d, generator=generator, device=self.device)
        z = self.actor.one_step(eps, cond)
        a = torch.tanh(z)
        if not deterministic:
            a = (a + self._sigma() * torch.randn(a.shape, generator=generator, device=self.device)).clamp(-1.0, 1.0)
        return ActionResult(action_norm=a, z=z, eps=eps)

    def update(self, batch: ReplayBatch, uctx: UpdateContext) -> UpdateResult:
        a_cfg, rng, spec, norm = self.a, uctx.rng, uctx.spec, uctx.norm
        self._progress = uctx.env_transitions
        ctx, nctx = batch.ctx.to(self.device), batch.next_ctx.to(self.device)
        cond, cond_next = self.builder.build(ctx, norm, spec), self.builder.build(nctx, norm, spec)
        B = cond.shape[0]
        with torch.no_grad():
            eps_n = torch.randn(B, self.d, generator=rng.torch("target_source"), device=self.device)
            a_n = torch.tanh(self.actor_target.one_step(eps_n, cond_next))
            y = batch.reward + a_cfg.gamma * batch.bootstrap_mask * self.critic_target.min_q(cond_next, a_n)
        q1, q2 = self.critic(cond, batch.action_command_norm)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        assert_finite("ofql critic loss", c_loss)
        self.opt_critic.zero_grad(set_to_none=True)
        c_loss.backward()
        clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()
        self.critic_updates += 1
        m = {"critic/loss": float(c_loss.detach()), "critic/q1_mean": float(q1.detach().mean()), "critic/target_mean": float(y.mean()),
             "critic/td_error": float((q1.detach() - y).abs().mean())}

        did_actor = False
        if uctx.do_actor:
            did_actor = True
            fl = a_cfg.flow
            ts = sample_time_pairs(B, delta=fl.time_delta, flow_ratio=fl.flow_ratio, kind=fl.time_sampling, logit_mean=fl.logit_mean,
                                   logit_std=fl.logit_std, generator=rng.torch("time"), device=self.device)
            z_a, _ = atanh_clamped(batch.action_policy_norm, a_cfg.atanh_eps)
            eps = torch.randn(B, self.d, generator=rng.torch("candidates"), device=self.device)
            z_t = (1 - ts.t) * z_a + ts.t * eps
            v = eps - z_a
            tgt, jn = meanflow_target(lambda z, r, t: self.actor(z, r, t, cond), z_t, ts.r, ts.t, v)
            fbc = ((self.actor(z_t, ts.r, ts.t, cond) - tgt) ** 2).mean()
            eps_g = torch.randn(B, self.d, generator=rng.torch("alpha_source"), device=self.device)
            a_pi = torch.tanh(self.actor.one_step(eps_g, cond))
            for p in self.critic.parameters():
                p.requires_grad_(False)
            q_pi = self.critic.min_q(cond, a_pi)
            for p in self.critic.parameters():
                p.requires_grad_(True)
            q_abs = float(q_pi.detach().abs().mean())
            self.q_abs_ema = q_abs if self.q_abs_ema is None else self.q_ema_decay * self.q_abs_ema + (1 - self.q_ema_decay) * q_abs
            alpha_q = self.eta / (self.q_abs_ema + 1e-6)
            a_loss = fbc - alpha_q * q_pi.mean()
            assert_finite("ofql actor loss", a_loss)
            self.opt_actor.zero_grad(set_to_none=True)
            a_loss.backward()
            gn, _ = clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
            self.opt_actor.step()
            self.actor_updates += 1
            m.update({"loss/actor_fbc": float(fbc.detach()), "loss/actor_total": float(a_loss.detach()), "ofql/alpha_q": alpha_q,
                      "flow/jvp_norm": float(jn.mean()), "actor/grad_norm": gn, "ofql/explore_sigma": self._sigma()})
        ema_update(self.critic_target, self.critic, a_cfg.target_tau)
        if did_actor:
            ema_update(self.actor_target, self.actor, a_cfg.target_tau)
        return UpdateResult(m, did_actor)

    def state_dict(self) -> Dict[str, Any]:
        return {"actor": self.actor.state_dict(), "actor_target": self.actor_target.state_dict(), "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(), "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(), "q_abs_ema": self.q_abs_ema,
                "counters": {"actor_updates": self.actor_updates, "critic_updates": self.critic_updates, "clip_counts": {}},
                "progress": self._progress}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        for k in ("actor", "actor_target", "critic", "critic_target"):
            getattr(self, k).load_state_dict(sd[k])
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])
        self.q_abs_ema = sd["q_abs_ema"]
        self.actor_updates, self.critic_updates = sd["counters"]["actor_updates"], sd["counters"]["critic_updates"]
        self._progress = sd.get("progress", 0)
