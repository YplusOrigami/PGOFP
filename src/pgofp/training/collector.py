"""Collector: acting, controller advance, true next condition, replay storage (docs section 8.1).

Per step and per env slot:
  condition (obs, reference, phase, controller memory, FROZEN prior params) -> action (warmup | source+flow)
  -> env.step -> controller.advance (once) -> TRUE next condition (pre-reset final observation) -> store
  -> only finished slots reset (controller state + reference progress of other slots untouched).
terminated / truncated are stored separately; bootstrap_mask = 1 - terminated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..algorithms.base import BaseAgent
from ..controllers.manager import ControllerManager
from ..core.specs import EnvSpec, SourceSpec
from ..core.types import CONTEXT_FIELDS, ContextBatch
from ..data.replay_buffer import ReplayBuffer
from ..envs.vector import SyncEnvPool
from ..priors.mixture import build_source
from ..utils.normalization import NormSnapshot, RunningNorm
from ..utils.seed import RngStreams


@dataclass
class EpisodeRecord:
    ret: float
    length: int
    terminated: bool
    truncated: bool
    env_transitions: int
    env_id: int


class Collector:
    def __init__(self, cfg, pool: SyncEnvPool, manager: ControllerManager, env_spec: EnvSpec, replay: ReplayBuffer,
                 normalizer: RunningNorm, rng: RngStreams, device: torch.device):
        self.cfg = cfg
        self.pool = pool
        self.manager = manager
        self.spec = env_spec
        self.replay = replay
        self.norm = normalizer
        self.rng = rng
        self.device = device
        self.n = pool.n
        self.d = env_spec.action_dim
        self.cur: Dict[str, np.ndarray] = {}
        self.ep_return = np.zeros(self.n)
        self.ep_len = np.zeros(self.n, np.int64)
        self.episodes: List[EpisodeRecord] = []
        self.vector_steps = 0
        self.reward_scale = float(cfg.env.reward_scale)
        self.last_component_usage: Optional[np.ndarray] = None
        self._saturation_sum = 0.0
        self._saturation_n = 0

    # ------------------------------------------------------------------ contexts
    def _make_ctx(self, obs: np.ndarray, ref: np.ndarray, phase: np.ndarray, prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {"obs_raw": obs.astype(np.float32), "reference": ref.astype(np.float32), "phase": phase.astype(np.float32),
                "ctrl_feat": prior["ctrl_feat"], "prior_mu": prior["prior_mu"], "prior_logstd": prior["prior_logstd"],
                "prior_valid": prior["prior_valid"]}

    def _gather_refs(self, idx=None):
        idx = range(self.n) if idx is None else idx
        ref = np.stack([self.pool.reference(i) for i in idx]) if self.spec.ref_dim else np.zeros((len(list(idx)), 0), np.float32)
        ph = np.stack([self.pool.phase(i) for i in idx]) if self.spec.phase_dim else np.zeros((len(list(idx)), 0), np.float32)
        return ref, ph

    def reset(self) -> None:
        obs = self.pool.reset_all()
        self.manager.reset_slots(np.ones(self.n, bool))
        ref, ph = self._gather_refs()
        prior = self.manager.predict(obs, ref)
        self.cur = self._make_ctx(obs, ref, ph, prior)
        self.ep_return[:] = 0
        self.ep_len[:] = 0

    # ------------------------------------------------------------------ one vector step
    def _batch(self, ctx: Dict[str, np.ndarray]) -> ContextBatch:
        return ContextBatch.from_numpy(ctx, device=self.device)

    def step(self, agent: Optional[BaseAgent], spec: SourceSpec, norm: NormSnapshot, env_transitions: int, warmup: bool) -> int:
        cfg = self.cfg
        n, d = self.n, self.d
        ctx_np = self.cur
        # ---- action
        if warmup and cfg.train.warmup_source == "uniform_action":
            a_pol = self.rng.numpy("behavior_source").uniform(-1, 1, size=(n, d)).astype(np.float32)
        elif warmup and cfg.train.warmup_source in ("prior", "mixed"):
            a_pol = self._prior_warmup_action(ctx_np, spec, mixed=cfg.train.warmup_source == "mixed")
        else:
            res = agent.act(self._batch(ctx_np), spec, norm, self.rng.torch("behavior_source"), K=cfg.algorithm.K_exec)
            a_pol = res.action_norm.cpu().numpy().astype(np.float32)
            if "component_usage" in res.info:
                self.last_component_usage = res.info["component_usage"]
            self._saturation_sum += float((np.abs(a_pol) > 0.99).mean())
            self._saturation_n += 1
        a_cmd = a_pol  # no safety filter in MuJoCo: policy action == command sent
        a_phys = self.spec.action.to_physical(a_cmd).astype(np.float32)

        # ---- environment step for every slot
        obs_cur, ref_cur = ctx_np["obs_raw"], ctx_np["reference"]
        outs = [self.pool.step_slot(i, a_phys[i]) for i in range(n)]
        next_obs = np.stack([o.next_obs for o in outs]).astype(np.float32)
        reward = np.asarray([o.reward for o in outs], np.float64)
        terminated = np.asarray([o.terminated for o in outs], bool)
        truncated = np.asarray([o.truncated for o in outs], bool)
        done = terminated | truncated

        # ---- controller advances exactly once, then TRUE next condition (before any reset)
        self.manager.advance(obs_cur, ref_cur, a_cmd)
        next_ref, next_ph = self._gather_refs()
        next_prior = self.manager.predict(next_obs, next_ref)
        next_ctx = self._make_ctx(next_obs, next_ref, next_ph, next_prior)

        # ---- store
        rec: Dict[str, np.ndarray] = {}
        for k in CONTEXT_FIELDS:
            rec[k] = ctx_np[k]
            rec["next_" + k] = next_ctx[k]
        rec.update({
            "action_policy_norm": a_pol, "action_command_norm": a_cmd,
            "reward": (reward * self.reward_scale).astype(np.float32),
            "terminated": terminated.astype(np.float32), "truncated": truncated.astype(np.float32),
            "bootstrap_mask": (1.0 - terminated.astype(np.float32)),
            "episode_id": self.pool.episode_id.copy(), "step_in_episode": self.pool.step_in_episode.copy(),
            "env_id": np.arange(n, dtype=np.int16), "source_version": np.full(n, spec.version, np.int32),
            "beta_behavior": np.full(n, spec.beta, np.float32), "sigma_behavior": np.full(n, spec.sigma_prior, np.float32),
            "controller_valid": ctx_np["prior_valid"].mean(-1) if ctx_np["prior_valid"].shape[1] else np.ones(n, np.float32),
            "fallback_reason": (ctx_np["prior_valid"].min(-1) < 0.5).astype(np.int8) if ctx_np["prior_valid"].shape[1] else np.zeros(n, np.int8),
        })
        self.replay.add_batch(rec)
        self.norm.update(obs_cur)  # training env only; evaluation never updates statistics

        self.ep_return += reward
        self.ep_len += 1
        new_cur = {k: v.copy() for k, v in next_ctx.items()}
        for i in np.nonzero(done)[0]:
            self.episodes.append(EpisodeRecord(float(self.ep_return[i]), int(self.ep_len[i]), bool(terminated[i]),
                                               bool(truncated[i]), env_transitions + n, int(i)))
            self.ep_return[i] = 0.0
            self.ep_len[i] = 0
            mask = np.zeros(n, bool)
            mask[i] = True
            self.manager.reset_slots(mask)
            o = self.pool.reset_slot(int(i))
            r_i, p_i = self._gather_refs([int(i)])
            pr = self.manager.predict(o[None], r_i, idx=np.array([i]))
            fresh = self._make_ctx(o[None], r_i, p_i, pr)
            for k in new_cur:
                new_cur[k][i] = fresh[k][0]
        self.cur = new_cur
        self.vector_steps += 1
        return n

    def _prior_warmup_action(self, ctx_np, spec: SourceSpec, mixed: bool) -> np.ndarray:
        n, d = self.n, self.d
        rs = self.rng.numpy("behavior_source")
        if not spec.uses_prior:
            return rs.uniform(-1, 1, size=(n, d)).astype(np.float32)
        src = build_source(spec, self._batch(ctx_np), d)
        a = torch.tanh(src.sample((), self.rng.torch("behavior_source"))).cpu().numpy().astype(np.float32)
        if mixed:
            use_uniform = rs.random(n) < 0.5
            a = np.where(use_uniform[:, None], rs.uniform(-1, 1, size=(n, d)).astype(np.float32), a)
        return a

    # ------------------------------------------------------------------ stats
    def drain_episode_stats(self) -> Dict[str, float]:
        eps, self.episodes = self.episodes, []
        out: Dict[str, float] = {}
        if eps:
            rets = np.asarray([e.ret for e in eps])
            lens = np.asarray([e.length for e in eps], float)
            out["charts/episode_return"] = float(rets.mean())  # per-episode weighted (not per-collector)
            out["charts/episode_length"] = float(lens.mean())
            out["charts/episodes"] = float(len(eps))
            out["charts/terminated_frac"] = float(np.mean([e.terminated for e in eps]))
        if self._saturation_n:
            out["actions/behavior_saturation_frac"] = self._saturation_sum / self._saturation_n
            self._saturation_sum, self._saturation_n = 0.0, 0
        return out

    # ------------------------------------------------------------------ checkpoint (episode-restart + optional exact)
    def state_dict(self) -> Dict[str, Any]:
        sd = {"vector_steps": self.vector_steps, "episode_id": self.pool.episode_id.copy(),
              "next_episode": self.pool._next_episode, "ep_return": self.ep_return.copy(), "ep_len": self.ep_len.copy(),
              "cur": {k: v.copy() for k, v in self.cur.items()}, "step_in_episode": self.pool.step_in_episode.copy()}
        if all(hasattr(e, "get_snapshot") for e in self.pool.envs):
            sd["env_snapshots"] = [e.get_snapshot() for e in self.pool.envs]
        return sd

    def load_state_dict(self, sd: Dict[str, Any], exact: bool) -> bool:
        """Returns True if the in-flight episodes were restored exactly, False if the caller must restart episodes."""
        self.vector_steps = sd["vector_steps"]
        self.pool._next_episode = sd["next_episode"]
        if exact and "env_snapshots" in sd:
            for e, s in zip(self.pool.envs, sd["env_snapshots"]):
                e.set_snapshot(s)
            self.pool.episode_id[:] = sd["episode_id"]
            self.pool.step_in_episode[:] = sd["step_in_episode"]
            self.ep_return, self.ep_len = sd["ep_return"].copy(), sd["ep_len"].copy()
            self.cur = {k: v.copy() for k, v in sd["cur"].items()}
            self.pool._seeded = [True] * self.n
            return True
        return False
