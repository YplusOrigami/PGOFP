"""Immutable specs with a schema version and a stable content hash."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 1


def stable_hash(obj: Any) -> str:
    """Short deterministic hash of a JSON-able object."""
    blob = json.dumps(obj, sort_keys=True, default=_json_default)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    return str(o)


@dataclass(frozen=True)
class ActionSpec:
    """Physical action box. Flow latent -> tanh -> affine to physical (see math.action_transform)."""

    low: Tuple[float, ...]
    high: Tuple[float, ...]

    def __post_init__(self):
        if len(self.low) != len(self.high):
            raise ValueError("low/high length mismatch")
        if any(h <= l for l, h in zip(self.low, self.high)):
            raise ValueError("high must be strictly greater than low")

    @property
    def dim(self) -> int:
        return len(self.low)

    @property
    def bias(self) -> np.ndarray:
        return (np.asarray(self.high) + np.asarray(self.low)) / 2.0

    @property
    def scale(self) -> np.ndarray:
        return (np.asarray(self.high) - np.asarray(self.low)) / 2.0

    def to_physical(self, a_norm: np.ndarray) -> np.ndarray:
        return self.bias + self.scale * a_norm

    def to_normalized(self, a_phys: np.ndarray) -> np.ndarray:
        return (a_phys - self.bias) / self.scale


@dataclass(frozen=True)
class CapabilitySpec:
    """What a controller/backend can and cannot do; used for pre-flight validation."""

    deterministic: bool = True
    provides_std: bool = False
    stateful: bool = False
    needs_reference: bool = False
    model_access: str = "none"  # none | observation_model | privileged_sim_state
    latent_coordinates: bool = False  # True if center/std are already pre-tanh


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    obs_dim: int
    action: ActionSpec
    ref_dim: int = 0
    phase_dim: int = 0
    max_episode_steps: Optional[int] = None
    control_dt: Optional[float] = None
    obs_names: Tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    @property
    def action_dim(self) -> int:
        return self.action.dim

    @property
    def hash(self) -> str:
        return stable_hash(dataclasses.asdict(self))


@dataclass(frozen=True)
class SourceSpec:
    """Immutable snapshot of the conditional source distribution parameters.

    kind="gaussian": p_src = N(0, sigma_base^2 I).
    kind="prior_mixture": p_src = (1-beta) N(0, sigma_base^2 I) + beta * sum_m w_m q_m(z|c)
        where q_m are the (pre-tanh) Gaussian prior components carried in the context.
        beta=1 is a pure prior source, beta=0 degenerates to gaussian.
    prior_std_mode: "fixed" uses sigma_prior for every channel (deterministic controllers);
        "native" uses the controller's own log-std times std_multiplier (frozen PPO).
    All stds are clamped to [sigma_min, sigma_max] (latent units) so densities are always defined.
    """

    kind: str = "gaussian"
    beta: float = 0.0
    sigma_base: float = 1.0
    sigma_prior: float = 0.3
    sigma_min: float = 0.05
    sigma_max: float = 2.0
    std_multiplier: float = 1.0
    prior_std_mode: str = "fixed"
    component_weights: Tuple[float, ...] = (1.0,)
    version: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.kind not in ("gaussian", "prior_mixture"):
            raise ValueError(f"unknown source kind {self.kind!r}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if self.prior_std_mode not in ("fixed", "native"):
            raise ValueError("prior_std_mode must be fixed|native")
        if not (0 < self.sigma_min <= self.sigma_max):
            raise ValueError("need 0 < sigma_min <= sigma_max")
        if self.sigma_base <= 0:
            raise ValueError("sigma_base must be positive")
        w = np.asarray(self.component_weights, dtype=np.float64)
        if w.size and (np.any(w < 0) or abs(w.sum() - 1.0) > 1e-6):
            raise ValueError("component_weights must be non-negative and sum to 1")

    @property
    def uses_prior(self) -> bool:
        return self.kind == "prior_mixture" and self.beta > 0.0

    @property
    def hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("version")
        return stable_hash(d)

    def replace(self, **kw) -> "SourceSpec":
        return dataclasses.replace(self, **kw)

    def feature_vector(self) -> np.ndarray:
        """Scalar schedule variables fed to networks when `condition.source_features` is enabled."""
        return np.asarray([self.beta, self.sigma_prior], dtype=np.float32)


@dataclass(frozen=True)
class ArtifactSpec:
    kind: str
    path: str
    sha256: str
    meta: Dict[str, Any] = field(default_factory=dict)
