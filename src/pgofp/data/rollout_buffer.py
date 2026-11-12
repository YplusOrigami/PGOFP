"""On-policy rollout storage + GAE (kept separate from the off-policy replay)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class RolloutBuffer:
    """[T, N] storage. `terminated` stops bootstrapping; `truncated` bootstraps from V(final observation)."""

    def __init__(self, steps: int, n_envs: int, obs_dim: int, action_dim: int):
        self.T, self.N = steps, n_envs
        z = lambda *s: np.zeros((steps, n_envs, *s), np.float32)
        self.obs, self.z_act, self.logp, self.value = z(obs_dim), z(action_dim), z(), z()
        self.reward, self.terminated, self.truncated, self.trunc_value = z(), z(), z(), z()
        self.pos = 0

    def add(self, obs, z_act, logp, value, reward, terminated, truncated, trunc_value) -> None:
        i = self.pos
        self.obs[i], self.z_act[i], self.logp[i], self.value[i] = obs, z_act, logp, value
        self.reward[i], self.terminated[i], self.truncated[i], self.trunc_value[i] = reward, terminated, truncated, trunc_value
        self.pos += 1

    def full(self) -> bool:
        return self.pos >= self.T

    def compute_gae(self, last_value: np.ndarray, gamma: float, lam: float) -> Dict[str, np.ndarray]:
        adv = np.zeros((self.T, self.N), np.float32)
        last = np.zeros(self.N, np.float32)
        for t in reversed(range(self.T)):
            nxt_v = last_value if t == self.T - 1 else self.value[t + 1]
            # on truncation the next-state value is V(final obs) (stored), not the reset observation's value
            nxt_v = np.where(self.truncated[t] > 0, self.trunc_value[t], nxt_v)
            cont = 1.0 - self.terminated[t]
            episode_end = np.maximum(self.terminated[t], self.truncated[t])
            delta = self.reward[t] + gamma * nxt_v * cont - self.value[t]
            last = delta + gamma * lam * (1.0 - episode_end) * last
            adv[t] = last
        ret = adv + self.value
        return {"adv": adv, "ret": ret}

    def flatten(self, adv, ret, device) -> Dict[str, torch.Tensor]:
        f = lambda a: torch.as_tensor(a.reshape(self.T * self.N, *a.shape[2:]), device=device)
        return {"obs": f(self.obs), "z": f(self.z_act), "logp": f(self.logp), "adv": f(adv), "ret": f(ret), "value": f(self.value)}

    def reset(self) -> None:
        self.pos = 0
