"""Controller base: PURE predict + one explicit advance per environment step.

  predict(obs, ref, state)  -> ControllerPrediction     (no state mutation; callable any number of times)
  advance(state, obs, ref, applied_cmd_norm) -> new state  (called once per executed env step)
State is a dict of arrays with leading dim n_envs, owned by ControllerManager (one slot per env).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.specs import CapabilitySpec
from ..core.types import ControllerPrediction

State = Dict[str, np.ndarray]


class BaseController:
    capability = CapabilitySpec()
    n_components = 1
    ctrl_dim = 0  # width of the feature vector exposed to the networks (controller memory)
    version = "0"

    def initial_state(self, n_envs: int) -> State:
        return {}

    def reset_state(self, state: State, mask: np.ndarray) -> State:
        """Reset the masked env slots to the initial state; others untouched."""
        init = self.initial_state(len(mask))
        out = {k: v.copy() for k, v in state.items()}
        for k in out:
            out[k][mask] = init[k][mask]
        return out

    def predict(self, obs: np.ndarray, ref: np.ndarray, state: State) -> ControllerPrediction:
        raise NotImplementedError

    def advance(self, state: State, obs: np.ndarray, ref: np.ndarray, applied_cmd_norm: np.ndarray) -> State:
        return state

    def features(self, state: State) -> np.ndarray:
        n = next(iter(state.values())).shape[0] if state else 0
        return np.zeros((n, 0), np.float32)
