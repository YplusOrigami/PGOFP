"""7 normalized commands -> 12 coil voltages, with explicit ownership of delay / rate-limit / calibration.

`CoilMap` refuses unverified matrices for real backends. The test matrix used with the mock backend is built in code
(`CoilMap.mock_symmetric`) and flagged `is_mock`; it is NOT a statement about the real device mapping.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ...core.exceptions import BackendUnavailableError, ConfigError
from .protocol import ActuatorOwnership


class CoilMap:
    def __init__(self, matrix: np.ndarray, voltage_scale: np.ndarray, verified: bool, is_mock: bool = False, source: str = ""):
        matrix = np.asarray(matrix, np.float64)
        if matrix.shape[0] != 12 or matrix.ndim != 2:
            raise ConfigError(f"coil matrix must be [12, n_cmd], got {matrix.shape}")
        self.M = matrix
        self.n_cmd = matrix.shape[1]
        self.voltage_scale = np.asarray(voltage_scale, np.float64)  # per-coil full scale [V]
        self.verified = verified
        self.is_mock = is_mock
        self.source = source

    @staticmethod
    def mock_symmetric() -> "CoilMap":
        """7 commands -> 12 coils: command 0 -> main coil, commands 1..4 each drive an up/down PAIR (identical voltage),
        commands 5..6 -> two further pairs. Test fixture only."""
        M = np.zeros((12, 7))
        M[0, 0] = 1.0
        pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 11)]
        for j, (a, b) in enumerate(pairs[:6], start=1):
            M[a, j] = 1.0
            M[b, j] = 1.0
        return CoilMap(M, np.full(12, 1499.0), verified=True, is_mock=True, source="mock_fixture")

    @staticmethod
    def from_yaml(path: str | Path) -> "CoilMap":
        d = yaml.safe_load(Path(path).read_text())
        if not d.get("verified", False) or d.get("matrix") is None:
            raise BackendUnavailableError(f"{path}: coil_map is not verified (verified:false / matrix missing); "
                                          "real runs require the device-document matrix")
        lim = d.get("voltage_limits_V", {})
        scale = np.full(12, float(next(iter(lim.values()))) if lim else 1499.0)
        return CoilMap(np.asarray(d["matrix"]), scale, True, False, d.get("source", ""))

    def expand(self, cmd_norm: np.ndarray) -> np.ndarray:
        """[n_cmd] in [-1, 1] -> [12] voltages (symmetry is enforced by the matrix itself)."""
        c = np.clip(np.asarray(cmd_norm, np.float64), -1.0, 1.0)
        return (self.M @ c) * self.voltage_scale


class ActuatorWrapper:
    """Applies delay / rate limit ONLY for stages it owns; stages owned by the backend are passed through untouched,
    so an SDK that already models the power supply is never double-applied."""

    def __init__(self, ownership: ActuatorOwnership, delay_steps: int = 0, rate_limit_frac: Optional[float] = None,
                 full_scale: Optional[np.ndarray] = None):
        ownership.validate()
        self.own = ownership
        self.delay_steps = delay_steps
        self.rate = rate_limit_frac
        self.full_scale = full_scale
        self.reset()

    def reset(self) -> None:
        self.buf = deque(maxlen=max(self.delay_steps, 0) + 1)
        self.prev = None

    def apply(self, v_set: np.ndarray) -> np.ndarray:
        v = np.asarray(v_set, np.float64)
        if self.own.delay == "wrapper" and self.delay_steps > 0:
            self.buf.append(v)
            v = self.buf[0] if len(self.buf) > self.delay_steps else np.zeros_like(v)
        if self.own.rate_limit == "wrapper" and self.rate is not None:
            lim = self.rate * (self.full_scale if self.full_scale is not None else np.full_like(v, 1499.0))
            if self.prev is not None:
                v = self.prev + np.clip(v - self.prev, -lim, lim)
        self.prev = v
        return v
