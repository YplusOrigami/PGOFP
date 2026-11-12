"""Optional safety layer. The filter never hides that the policy's action was modified.

Policy density (entropy/Q targets) describes the pre-filter policy; with the filter ON the pushforward density is
unknown, so density is reported as `density_before_filter` and filter on/off must be compared as a separate experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FilterResult:
    raw_command: np.ndarray
    filtered_command: np.ndarray
    intervention: bool
    reason: str


class SafetyFilter:
    """Clip to a symmetric box and limit per-step command change. Not a proven constrained-optimal safety guarantee."""

    def __init__(self, box: float = 1.0, max_delta: Optional[float] = None, enabled: bool = True):
        self.box, self.max_delta, self.enabled = box, max_delta, enabled
        self.prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, cmd: np.ndarray) -> FilterResult:
        raw = np.asarray(cmd, np.float64)
        if not self.enabled:
            self.prev = raw
            return FilterResult(raw, raw, False, "")
        out, reasons = np.clip(raw, -self.box, self.box), []
        if np.any(out != raw):
            reasons.append("box")
        if self.max_delta is not None and self.prev is not None:
            lim = np.clip(out, self.prev - self.max_delta, self.prev + self.max_delta)
            if np.any(lim != out):
                reasons.append("rate")
            out = lim
        self.prev = out
        return FilterResult(raw, out, bool(reasons), "+".join(reasons))
