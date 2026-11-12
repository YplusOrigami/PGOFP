"""Synchronous environment pool with EXPLICIT per-slot resets (no autoreset ambiguity).

step() returns the TRUE next observation (pre-reset) for every slot; the caller decides which slots to reset.
A reset only touches the finished slot, so per-env controller state/reference progress stay independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from ..utils.seed import derive_seed


@dataclass
class StepOut:
    next_obs: np.ndarray  # true final observation of the step (never the post-reset one)
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class SyncEnvPool:
    def __init__(self, make_env: Callable[[int], object], n_envs: int, base_seed: int, tag: str = "train"):
        self.envs: List = [make_env(i) for i in range(n_envs)]
        self.n = n_envs
        self.base_seed = base_seed
        self.tag = tag
        self.spec = self.envs[0].spec
        self.episode_id = np.zeros(n_envs, np.int64)
        self.step_in_episode = np.zeros(n_envs, np.int32)
        self._seeded = [False] * n_envs
        self._next_episode = 0

    def reset_slot(self, i: int, seed: Optional[int] = None) -> np.ndarray:
        if seed is None and not self._seeded[i]:
            seed = derive_seed(self.base_seed, self.tag, "env", i)
        obs, _ = self.envs[i].reset(seed=seed)
        self._seeded[i] = True
        self.episode_id[i] = self._next_episode
        self._next_episode += 1
        self.step_in_episode[i] = 0
        return obs

    def reset_all(self) -> np.ndarray:
        return np.stack([self.reset_slot(i) for i in range(self.n)])

    def step_slot(self, i: int, action_phys: np.ndarray) -> StepOut:
        obs, r, term, trunc, info = self.envs[i].step(action_phys)
        self.step_in_episode[i] += 1
        return StepOut(obs, r, term, trunc, info)

    def reference(self, i: int) -> np.ndarray:
        return self.envs[i].reference()

    def phase(self, i: int) -> np.ndarray:
        return self.envs[i].phase(int(self.step_in_episode[i]))

    def close(self) -> None:
        for e in self.envs:
            e.close()
