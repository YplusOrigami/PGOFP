"""beta / sigma_prior schedules producing immutable SourceSpec snapshots.

Clock: env_transitions / schedule_total_steps (never optimizer steps or wall time).
One learner update uses ONE SourceSpec snapshot for Actor, Critic target, density and alpha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.exceptions import ConfigError
from ..core.specs import SourceSpec


@dataclass
class ScheduleConfig:
    source_kind: str = "gaussian"  # gaussian | prior_mixture
    kind: str = "constant"  # constant | linear | piecewise | cosine
    beta: float = 0.75
    sigma_prior: float = 0.3
    sigma_base: float = 1.0
    sigma_min: float = 0.05
    sigma_max: float = 2.0
    prior_std_mode: str = "fixed"
    std_multiplier: float = 1.0
    component_weights: Sequence[float] = (1.0,)
    total_steps: int = 1_000_000
    beta_start: float = 0.0
    beta_end: float = 0.75
    sigma_start: float = 1.0
    sigma_end: float = 0.3
    points: List[Sequence[float]] = field(default_factory=list)  # piecewise: [progress, beta, sigma]
    max_delta: float = 0.0  # max change of beta or sigma between consecutive snapshots (0 = unlimited)


class SourceScheduler:
    def __init__(self, cfg: ScheduleConfig):
        if cfg.kind not in ("constant", "linear", "piecewise", "cosine"):
            raise ConfigError(f"unknown schedule kind {cfg.kind!r}")
        if cfg.kind == "piecewise" and len(cfg.points) < 2:
            raise ConfigError("piecewise schedule needs >= 2 points")
        self.cfg = cfg
        self._last: Optional[SourceSpec] = None
        self._version = 0

    # ------------------------------------------------------------------ values
    def _values(self, progress: float):
        c = self.cfg
        if c.kind == "constant":
            return c.beta, c.sigma_prior
        if c.kind == "linear":
            p = progress
        elif c.kind == "cosine":
            p = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            pts = sorted(c.points, key=lambda x: x[0])
            if progress <= pts[0][0]:
                return pts[0][1], pts[0][2]
            for (p0, b0, s0), (p1, b1, s1) in zip(pts[:-1], pts[1:]):
                if progress <= p1:
                    a = (progress - p0) / max(p1 - p0, 1e-12)
                    return b0 + a * (b1 - b0), s0 + a * (s1 - s0)
            return pts[-1][1], pts[-1][2]
        return (c.beta_start + p * (c.beta_end - c.beta_start), c.sigma_start + p * (c.sigma_end - c.sigma_start))

    def spec_at(self, env_transitions: Optional[int]) -> SourceSpec:
        """Snapshot for the given clock value. `None` freezes at the last emitted spec (evaluation)."""
        c = self.cfg
        if env_transitions is None and self._last is not None:
            return self._last
        progress = min(1.0, max(0.0, (env_transitions or 0) / max(c.total_steps, 1)))
        beta, sigma = self._values(progress)
        if c.source_kind == "gaussian":
            beta = 0.0
        if c.max_delta > 0 and self._last is not None:
            beta = self._last.beta + max(-c.max_delta, min(c.max_delta, beta - self._last.beta))
            sigma = self._last.sigma_prior + max(-c.max_delta, min(c.max_delta, sigma - self._last.sigma_prior))
        sigma = min(max(sigma, c.sigma_min), c.sigma_max)
        if self._last is None or abs(beta - self._last.beta) > 1e-12 or abs(sigma - self._last.sigma_prior) > 1e-12:
            self._version += 1 if self._last is not None else 0
        spec = SourceSpec(
            kind=c.source_kind, beta=float(beta), sigma_base=c.sigma_base, sigma_prior=float(sigma),
            sigma_min=c.sigma_min, sigma_max=c.sigma_max, std_multiplier=c.std_multiplier,
            prior_std_mode=c.prior_std_mode, component_weights=tuple(c.component_weights), version=self._version,
        )
        self._last = spec
        return spec

    # ------------------------------------------------------------------ state
    def state_dict(self) -> Dict[str, Any]:
        return {"last": self._last, "version": self._version}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self._last = sd["last"]
        self._version = sd["version"]
