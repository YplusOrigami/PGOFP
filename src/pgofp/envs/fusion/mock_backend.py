"""Deterministic small-state-space stand-in for the fusion simulator (interface tests ONLY; not HFM).

State: 3 tracked quantities (Ip, R, Z). Dynamics x' = x + dt(-a x + B u_real), u_real = delayed + rate-limited 12-D
voltages. This backend OWNS delay/rate-limit (ownership=backend), so wrappers must not apply them again.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from .protocol import (ActuatorOwnership, BackendDescription, BackendStep, FieldSpec, ObservationSchema)

MOCK_FIELDS = ("Ip_error", "R_error", "Z_error", "Ip_rate", "R_rate", "Z_rate")


class MockFusionBackend:
    def __init__(self, horizon: int = 60, delay_steps: int = 2, rate_limit_frac: float = 0.16, terminal_on_horizon: bool = False,
                 dt: float = 0.01, seed: int = 0, error_limit: float = 3.0):
        self.horizon = horizon
        self.delay_steps = delay_steps
        self.rate = rate_limit_frac
        self.terminal_on_horizon = terminal_on_horizon
        self.dt = dt
        self.error_limit = error_limit
        g = np.random.default_rng(12345)  # fixed fixture matrix: the mock plant never depends on the run seed
        self.B = g.normal(size=(3, 12)) * 4.0
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.x = np.zeros(3)
        self.ref_traj = self._make_ref()
        self.buf = deque(maxlen=delay_steps + 1)
        self.prev_v = np.zeros(12)
        schema = ObservationSchema(tuple(FieldSpec(n, "normalized", i) for i, n in enumerate(MOCK_FIELDS)), 6, verified=False,
                                   source="mock_fixture")
        self._desc = BackendDescription(schema, 12, 7, dt, ActuatorOwnership("backend", "backend", "backend"),
                                        ("snapshot",), is_mock=True)

    def _make_ref(self) -> np.ndarray:
        s = np.linspace(0, 1, self.horizon + 1)[:, None]
        return np.concatenate([0.5 * s, 0.3 * np.sin(2 * np.pi * s), -0.2 * s], 1).astype(np.float32)

    def describe(self) -> BackendDescription:
        return self._desc

    def reset(self, shot_id: str = "mock0", reset_params: Optional[dict] = None, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scale = float((reset_params or {}).get("perturbation", 0.2))
        self.x = self.ref_traj[0].astype(np.float64) + scale * self.rng.normal(size=3)
        self.t = 0
        self.buf.clear()
        self.prev_v = np.zeros(12)
        return self._obs()[0], {"shot_id": shot_id}

    def _obs(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        e = self.x - self.ref_traj[min(self.t, self.horizon)]
        rate = -0.5 * e
        vals = dict(zip(MOCK_FIELDS, [*e, *rate]))
        vals = {k: np.asarray([v], np.float32) for k, v in vals.items()}
        return vals, {k: np.ones(1, np.float32) for k in vals}

    def advance(self, voltage_command: np.ndarray) -> BackendStep:
        v = np.asarray(voltage_command, np.float64)
        self.buf.append(v)
        v_d = self.buf[0] if len(self.buf) > self.delay_steps else np.zeros(12)
        lim = self.rate * 1499.0
        v_r = self.prev_v + np.clip(v_d - self.prev_v, -lim, lim)
        self.prev_v = v_r
        self.x = self.x + self.dt * (-1.0 * (self.x - self.ref_traj[min(self.t, self.horizon)]) + self.B @ (v_r / 1499.0))
        self.t += 1
        e = self.x - self.ref_traj[min(self.t, self.horizon)]
        terms = {"Ip": float(abs(e[0])), "R": float(abs(e[1])), "Z": float(abs(e[2]))}
        reward = -(terms["Ip"] + terms["R"] + terms["Z"])
        diverged = bool(np.abs(e).max() > self.error_limit)
        at_end = self.t >= self.horizon
        terminated = diverged or (at_end and self.terminal_on_horizon)
        truncated = at_end and not terminated
        if diverged:
            reward -= 10.0  # fixed penalty on disruption-like termination (mirrors the proposal's reward structure)
        vals, mask = self._obs()
        reason = "divergence" if diverged else ("horizon_terminal" if (at_end and self.terminal_on_horizon) else ("time_limit" if truncated else ""))
        return BackendStep(vals, mask, float(reward), terms, terminated, reason, truncated, v_r, {"t": self.t})

    def get_reference(self, time_index: int, horizon: int) -> np.ndarray:
        idx = np.minimum(np.arange(time_index, time_index + 1 + horizon), self.horizon)
        return self.ref_traj[idx].reshape(-1)

    def get_snapshot(self):
        return {"x": self.x.copy(), "t": self.t, "buf": list(self.buf), "prev_v": self.prev_v.copy(),
                "rng": self.rng.bit_generator.state}

    def set_snapshot(self, snap) -> None:
        self.x, self.t = snap["x"].copy(), snap["t"]
        self.buf = deque(snap["buf"], maxlen=self.delay_steps + 1)
        self.prev_v = snap["prev_v"].copy()
        self.rng.bit_generator.state = snap["rng"]
