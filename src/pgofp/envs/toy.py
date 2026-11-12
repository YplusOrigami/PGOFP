"""Deterministic toy environments for tests / smoke (no MuJoCo needed).

ToyReach: obs s in R^2 evolves s' = 0.9 s + 0.2 a_phys[:2]-ish drift + noise; reward = -||a - 0.6 tanh(s)||^2.
Supports exact snapshots (state + RNG) for strict resume tests.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.specs import ActionSpec, EnvSpec


class ToyReach:
    def __init__(self, action_dim: int = 2, horizon: int = 20, ref_dim: int = 0, phase_dim: int = 0, seed: int = 0):
        self.d = action_dim
        self.horizon = horizon
        self.spec = EnvSpec(
            env_id="ToyReach-v0", obs_dim=action_dim, action=ActionSpec((-2.0,) * action_dim, (2.0,) * action_dim),
            ref_dim=ref_dim, phase_dim=phase_dim, max_episode_steps=horizon, control_dt=0.05,
        )
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros(action_dim, np.float32)
        self.t = 0

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = self.rng.normal(size=self.d).astype(np.float32)
        self.t = 0
        return self.s.copy(), {}

    def step(self, a_phys: np.ndarray):
        a_norm = np.asarray(a_phys, np.float32) / 2.0
        target = 0.6 * np.tanh(self.s)
        reward = float(-np.sum((a_norm - target) ** 2))
        self.s = (0.9 * self.s + 0.3 * a_norm + 0.1 * self.rng.normal(size=self.d)).astype(np.float32)
        self.t += 1
        terminated = bool(np.abs(self.s).max() > 6.0)
        truncated = (not terminated) and self.t >= self.horizon
        return self.s.copy(), reward, terminated, truncated, {}

    def reference(self) -> np.ndarray:
        return np.zeros(self.spec.ref_dim, np.float32)

    def phase(self, step_in_episode: int) -> np.ndarray:
        if self.spec.phase_dim == 0:
            return np.zeros(0, np.float32)
        return np.asarray([1.0 - step_in_episode / self.horizon], np.float32)

    # exact snapshots
    def get_snapshot(self):
        return {"s": self.s.copy(), "t": self.t, "rng": self.rng.bit_generator.state}

    def set_snapshot(self, snap) -> None:
        self.s, self.t = snap["s"].copy(), snap["t"]
        self.rng.bit_generator.state = snap["rng"]

    def close(self) -> None:
        pass
