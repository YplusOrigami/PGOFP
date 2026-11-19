"""Sampling MPC (CEM) controller. First version: no cross-step warm start (pure query), fixed query randomness.

Planning uses a separate MjModel/MjData (envs.mujoco.planning); `model_access=observation_model` only. The
returned distribution center is the FIRST planned action; the source variance is an explicit latent sigma, NOT
an optimal-control posterior (elite variance, if ever used, would be a constructed first-step distribution).
"""
from __future__ import annotations

import hashlib
import time
from typing import Dict

import numpy as np

from ..core.exceptions import ConfigError, UnsupportedCombinationError
from ..core.registry import CONTROLLERS
from ..core.specs import CapabilitySpec
from ..core.types import ControllerPrediction
from .base import BaseController, State


class CEMMPC(BaseController):
    n_components = 1
    ctrl_dim = 0

    def __init__(self, cfg: Dict, env_spec, env_kwargs: dict | None = None):
        from ..envs.mujoco.planning import RolloutPlanner

        cfg = {**cfg, **dict(cfg.get("tasks", {}).get(env_spec.env_id, {}))}  # per-task overrides (horizon, cost, ...)

        access = cfg.get("model_access", "observation_model")
        if access == "privileged_sim_state":
            raise UnsupportedCombinationError(
                "privileged_sim_state MPC is not implemented: for InvertedPendulum/Reacher the public observation "
                "already determines the full MuJoCo state, so observation_model equals the oracle there")
        if access != "observation_model":
            raise ConfigError(f"unknown model_access {access!r}")
        self.capability = CapabilitySpec(deterministic=True, provides_std=False, stateful=False, model_access=access)
        self.horizon = int(cfg.get("horizon", 10))
        self.population = int(cfg.get("population", 64))
        self.n_elite = max(2, int(round(cfg.get("elite_fraction", 0.125) * self.population)))
        self.iterations = int(cfg.get("iterations", 3))
        self.init_std = float(cfg.get("init_std", 0.6))
        self.min_std = float(cfg.get("min_std", 0.05))
        self.seed = int(cfg.get("seed", 0))
        self.timeout_s = float(cfg.get("timeout_s", 0.5))
        self.cost_cfg = dict(cfg.get("cost", {}))
        self.version = str(cfg.get("version", "cem_mpc_v1"))
        self.action_dim = env_spec.action_dim
        # Parallelizes the population dimension of each CEM rollout across threads (see RolloutPlanner).
        # nthread=1 (default) reproduces the previous single-threaded behavior bit-for-bit.
        self.nthread = int(cfg.get("nthread", 1))
        self.planner = RolloutPlanner(env_spec.env_id, env_kwargs, nthread=self.nthread)
        self.model_rollouts = 0  # counted separately from environment interaction

    def _rng(self, obs_row: np.ndarray) -> np.random.Generator:
        h = hashlib.sha256(np.asarray(obs_row, np.float32).tobytes() + self.seed.to_bytes(8, "little", signed=True)).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def plan(self, obs_row: np.ndarray) -> np.ndarray:
        rng = self._rng(obs_row)
        init = self.planner.state_from_obs(obs_row)
        target = np.asarray(obs_row[4:6], np.float64) if self.planner.env_id == "Reacher-v5" else None
        H, d, P = self.horizon, self.action_dim, self.population
        mean, std = np.zeros((H, d)), np.full((H, d), self.init_std)
        for _ in range(self.iterations):
            seqs = np.clip(mean + std * rng.standard_normal((P, H, d)), -1.0, 1.0)
            seqs[0] = np.clip(mean, -1.0, 1.0)
            qpos, qvel = self.planner.rollout(init, seqs)
            self.model_rollouts += P * H
            cost = self.planner.stage_cost(qpos, qvel, seqs, target, self.cost_cfg)
            elite = seqs[np.argsort(cost)[: self.n_elite]]
            mean, std = elite.mean(0), np.maximum(elite.std(0), self.min_std)
        return np.clip(mean[0], -1.0, 1.0)

    def predict(self, obs, ref, state) -> ControllerPrediction:
        n = obs.shape[0]
        centers = np.zeros((n, 1, self.action_dim))
        valid = np.ones((n, 1), bool)
        t0 = time.perf_counter()
        for i in range(n):
            ti = time.perf_counter()
            try:
                a = self.plan(obs[i])
                ok = bool(np.isfinite(a).all()) and (time.perf_counter() - ti) <= self.timeout_s
            except Exception:
                a, ok = np.zeros(self.action_dim), False
            centers[i, 0] = a if ok else 0.0
            valid[i, 0] = ok
        return ControllerPrediction(center_norm=centers, valid=valid, latency_s=time.perf_counter() - t0, version=self.version,
                                    diagnostics={"model_rollout_steps": self.model_rollouts})


@CONTROLLERS.register("cem_mpc")
def _build_mpc(cfg, env_spec, env_kwargs=None, **kw):
    return CEMMPC(cfg, env_spec, env_kwargs)
