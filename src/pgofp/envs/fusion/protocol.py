"""FusionBackend protocol and schemas. Env / controller / algorithm only ever see these protocols, never HFM's private API.

The real SDK import is deferred to backend creation; without any SDK every MuJoCo/mock path still runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    name: str
    unit: str
    index: Optional[int] = None  # None = unknown (never guessed)
    count: int = 1


@dataclass(frozen=True)
class ObservationSchema:
    fields: Tuple[FieldSpec, ...]
    obs_dim: int
    verified: bool = False
    source: str = "unverified"


@dataclass(frozen=True)
class ReferenceSpec:
    task_id: str
    reference_hash: str
    dt: float
    n_steps: int
    ref_dim: int
    lookahead: int = 0


@dataclass(frozen=True)
class ActuatorOwnership:
    """Who applies transmission delay / rate limit / PSM calibration. Exactly one layer may own each stage."""

    delay: str = "backend"  # backend | wrapper
    rate_limit: str = "backend"
    calibration: str = "backend"

    def validate(self) -> None:
        for k, v in (("delay", self.delay), ("rate_limit", self.rate_limit), ("calibration", self.calibration)):
            if v not in ("backend", "wrapper"):
                raise ValueError(f"ownership of {k} must be backend|wrapper, got {v!r}")


@dataclass
class BackendStep:
    observation: Dict[str, np.ndarray]  # named fields (never positional)
    obs_mask: Dict[str, np.ndarray]  # 1 = measured, 0 = missing (NOT silently filled with 0)
    reward: float
    reward_terms: Dict[str, float]
    terminated: bool
    termination_reason: str
    truncated: bool
    applied_voltage: Optional[np.ndarray]  # actuator response actually applied (12-D), if reported
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendDescription:
    obs_schema: ObservationSchema
    n_coils: int
    n_commands: int
    control_period_s: float
    ownership: ActuatorOwnership
    capabilities: Tuple[str, ...] = ()
    is_mock: bool = False


@runtime_checkable
class FusionBackend(Protocol):
    def describe(self) -> BackendDescription: ...

    def reset(self, shot_id: str, reset_params: Optional[dict] = None, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], dict]: ...

    def advance(self, voltage_command: np.ndarray) -> BackendStep: ...

    def get_reference(self, time_index: int, horizon: int) -> np.ndarray: ...

    # optional
    def get_snapshot(self) -> Any: ...

    def set_snapshot(self, snap: Any) -> None: ...
