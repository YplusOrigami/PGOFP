"""Reference trajectories: loading, hashing, time indexing and look-ahead permissions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .protocol import ReferenceSpec


class ReferenceTrack:
    """A [T, d_r] reference with a content hash. `window(t, lookahead)` returns only what is permitted at deployment."""

    def __init__(self, data: np.ndarray, task_id: str, dt: float, lookahead: int = 0):
        self.data = np.asarray(data, np.float32)
        if self.data.ndim != 2:
            raise ValueError("reference must be [T, d_r]")
        self.task_id, self.dt, self.lookahead = task_id, dt, lookahead
        self.hash = hashlib.sha256(self.data.tobytes()).hexdigest()[:16]

    @staticmethod
    def from_npz(path: str | Path, task_id: str, key: str = "reference", dt: float = 0.01, lookahead: int = 0) -> "ReferenceTrack":
        return ReferenceTrack(np.load(path)[key], task_id, dt, lookahead)

    @property
    def spec(self) -> ReferenceSpec:
        return ReferenceSpec(self.task_id, self.hash, self.dt, self.data.shape[0],
                             self.data.shape[1] * (1 + self.lookahead), self.lookahead)

    def window(self, t: int) -> np.ndarray:
        """Reference at step t plus `lookahead` future steps (clamped at the end), flattened. Always for the QUERIED step:
        replay stores current and next references at collection time; it never re-reads 'now' for an old sample."""
        T = self.data.shape[0]
        idx = np.minimum(np.arange(t, t + 1 + self.lookahead), T - 1)
        return self.data[idx].reshape(-1)
