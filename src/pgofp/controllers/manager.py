"""Per-environment controller state + conversion to frozen latent prior arrays.

One ControllerManager serves ALL prior components (M >= 0).  Training and evaluation use separate instances, and
replay re-queries never touch the online states (replay stores frozen prior parameters instead).
Runtime controller failures (timeout/NaN/exception) become valid=0 for that component, which moves its source
weight onto the base Gaussian in BOTH sampling and density; configuration errors fail at construction.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.exceptions import PriorUnavailableError
from ..core.types import ControllerPrediction
from ..priors.controller_source import atanh_np
from .base import BaseController, State


class ControllerManager:
    def __init__(self, controllers: Sequence[BaseController], n_envs: int, action_dim: int, atanh_eps: float = 1e-3):
        self.controllers: List[BaseController] = list(controllers)
        self.n_envs = n_envs
        self.d = action_dim
        self.atanh_eps = atanh_eps
        self.n_prior = sum(c.n_components for c in self.controllers)
        self.ctrl_dim = sum(c.ctrl_dim for c in self.controllers)
        self.states: List[State] = [c.initial_state(n_envs) for c in self.controllers]
        self.fallback_count = 0
        self.query_count = 0
        self.latency_sum = 0.0
        self.versions = [c.version for c in self.controllers]
        self._atanh_clamp_sum = 0.0
        self._atanh_clamp_n = 0

    # ------------------------------------------------------------------ queries (pure)
    def predict(self, obs: np.ndarray, ref: np.ndarray, idx: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Frozen prior arrays for envs `idx` (default all) from the CURRENT states. Never mutates state."""
        n = obs.shape[0]
        sel = np.arange(self.n_envs) if idx is None else np.asarray(idx)
        if len(sel) != n:
            raise ValueError("obs rows must match idx length")
        mus, logstds, valids, feats = [], [], [], []
        t0 = time.perf_counter()
        for c, st in zip(self.controllers, self.states):
            sub = {k: v[sel] for k, v in st.items()}
            try:
                pred: ControllerPrediction = c.predict(obs, ref, sub)
                mu, ls, val = self._to_latent(pred, n, c.n_components)
            except Exception as e:  # runtime failure -> explicit fallback, never silent
                if isinstance(e, (PriorUnavailableError, TypeError, AttributeError, KeyError)):
                    raise
                mu = np.zeros((n, c.n_components, self.d), np.float32)
                ls = np.zeros_like(mu)
                val = np.zeros((n, c.n_components), np.float32)
            self.fallback_count += int((val < 0.5).sum())
            self.query_count += val.size
            mus.append(mu), logstds.append(ls), valids.append(val)
            feats.append(c.features(sub))
        self.latency_sum += time.perf_counter() - t0
        if not self.controllers:
            return {"prior_mu": np.zeros((n, 0, self.d), np.float32), "prior_logstd": np.zeros((n, 0, self.d), np.float32),
                    "prior_valid": np.zeros((n, 0), np.float32), "ctrl_feat": np.zeros((n, 0), np.float32)}
        return {
            "prior_mu": np.concatenate(mus, 1), "prior_logstd": np.concatenate(logstds, 1),
            "prior_valid": np.concatenate(valids, 1),
            "ctrl_feat": np.concatenate(feats, 1).astype(np.float32) if self.ctrl_dim else np.zeros((n, 0), np.float32),
        }

    def _to_latent(self, pred: ControllerPrediction, n: int, m: int):
        if pred.center_latent is not None:
            mu = np.asarray(pred.center_latent, np.float32)
        elif pred.center_norm is not None:
            z, ratio = atanh_np(np.asarray(pred.center_norm, np.float64), self.atanh_eps)
            mu = z.astype(np.float32)
            self._atanh_clamp_sum += ratio * mu.size
            self._atanh_clamp_n += mu.size
        else:
            raise PriorUnavailableError("controller returned neither center_norm nor center_latent")
        if mu.shape != (n, m, self.d):
            raise PriorUnavailableError(f"controller center shape {mu.shape} != {(n, m, self.d)}")
        valid = np.ones((n, m), np.float32) if pred.valid is None else np.asarray(pred.valid, np.float32).reshape(n, m)
        finite = np.isfinite(mu).all(-1)
        valid = valid * finite.astype(np.float32)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        ls = np.zeros_like(mu) if pred.log_std_latent is None else np.asarray(pred.log_std_latent, np.float32)
        ls = np.where(np.isfinite(ls), ls, 0.0)
        return mu, ls, valid

    # ------------------------------------------------------------------ state transitions
    def advance(self, obs: np.ndarray, ref: np.ndarray, applied_cmd_norm: np.ndarray) -> None:
        """Called exactly once per executed env step for ALL slots (obs = observation the command was chosen from)."""
        self.states = [c.advance(st, obs, ref, applied_cmd_norm) for c, st in zip(self.controllers, self.states)]

    def reset_slots(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, bool)
        self.states = [c.reset_state(st, mask) for c, st in zip(self.controllers, self.states)]

    # ------------------------------------------------------------------ stats / checkpointing
    def stats(self) -> Dict[str, float]:
        out = {
            "prior/fallback_rate": self.fallback_count / max(self.query_count, 1),
            "prior/controller_latency_ms": 1000.0 * self.latency_sum / max(self.query_count, 1) * max(self.n_prior, 1),
        }
        if self._atanh_clamp_n:
            out["prior/atanh_clamp_ratio"] = self._atanh_clamp_sum / self._atanh_clamp_n
        return out

    def reset_stats(self) -> None:
        self.fallback_count = self.query_count = 0
        self.latency_sum = 0.0
        self._atanh_clamp_sum = 0.0
        self._atanh_clamp_n = 0

    def state_dict(self) -> Dict:
        return {"states": [{k: v.copy() for k, v in s.items()} for s in self.states], "versions": self.versions}

    def load_state_dict(self, sd: Dict) -> None:
        if sd["versions"] != self.versions:
            raise PriorUnavailableError(f"controller versions changed: {sd['versions']} vs {self.versions}")
        self.states = [{k: v.copy() for k, v in s.items()} for s in sd["states"]]
