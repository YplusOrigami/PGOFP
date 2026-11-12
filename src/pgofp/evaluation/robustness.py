"""Predefined perturbations for robustness / held-out evaluation. They create NEW named conditions;
the standard task (and its results) is never overwritten, and controllers are not re-tuned on the test perturbation."""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict

import numpy as np

PERTURBATIONS: Dict[str, dict] = {
    "obs_noise_small": {"obs_noise_std": 0.01},
    "obs_noise_medium": {"obs_noise_std": 0.05},
    "action_delay_1": {"action_delay": 1},
    "action_delay_2": {"action_delay": 2},
    "mass_x1.2": {"mass_scale": 1.2},
    "mass_x0.8": {"mass_scale": 0.8},
}


class PerturbedEnv:
    """Wraps an adapter exposing reset/step/spec/reference/phase. Noise is applied to what the policy SEES (the observation the
    pool returns), never to the simulator state; delay queues commands; mass scaling edits the MuJoCo model once."""

    def __init__(self, env, name: str, seed: int = 0):
        p = PERTURBATIONS[name]
        self.env, self.name, self.spec = env, name, env.spec
        self.noise = float(p.get("obs_noise_std", 0.0))
        self.delay = int(p.get("action_delay", 0))
        self.rng = np.random.default_rng(seed)
        self.q: deque = deque()
        if "mass_scale" in p:
            m = env.unwrapped().model
            m.body_mass[:] = m.body_mass * p["mass_scale"]

    def _noisy(self, obs):
        return obs + self.noise * self.rng.standard_normal(obs.shape).astype(obs.dtype) if self.noise else obs

    def reset(self, seed=None, options=None):
        self.q.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy(obs), info

    def step(self, a_phys):
        if self.delay:
            self.q.append(np.asarray(a_phys))
            a = self.q.popleft() if len(self.q) > self.delay else np.zeros_like(a_phys)
        else:
            a = a_phys
        obs, r, te, tr, info = self.env.step(a)
        return self._noisy(obs), r, te, tr, info

    def reference(self):
        return self.env.reference()

    def phase(self, k):
        return self.env.phase(k)

    def close(self):
        self.env.close()
