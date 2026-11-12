"""Environment adapters: Gymnasium -> standard (obs, reward, terminated, truncated, info) + EnvSpec.

The adapter never changes reward / observation / termination of the standard task. terminated and truncated are
kept separate end-to-end; bootstrapping uses (1 - terminated) for external time limits.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.specs import ActionSpec, EnvSpec


class GymAdapter:
    def __init__(self, env_id: str, kwargs: Optional[dict] = None, ref_dim: int = 0, phase_dim: int = 0):
        import gymnasium as gym

        self.env = gym.make(env_id, **(kwargs or {}))
        box = self.env.action_space
        self.spec = EnvSpec(
            env_id=env_id,
            obs_dim=int(np.prod(self.env.observation_space.shape)),
            action=ActionSpec(tuple(float(x) for x in box.low), tuple(float(x) for x in box.high)),
            ref_dim=ref_dim,
            phase_dim=phase_dim,
            max_episode_steps=self.env.spec.max_episode_steps if self.env.spec is not None else None,
            control_dt=getattr(self.env.unwrapped, "dt", None),
        )
        self._reset_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_count += 1
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action_phys: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action_phys, dtype=np.float32))
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def reference(self) -> np.ndarray:
        return np.zeros(self.spec.ref_dim, dtype=np.float32)

    def phase(self, step_in_episode: int) -> np.ndarray:
        return np.zeros(self.spec.phase_dim, dtype=np.float32)

    def unwrapped(self):
        return self.env.unwrapped

    def render_rgb(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
