"""Interface specs (typing.Protocol). Implementations are decoupled through these."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

from .specs import CapabilitySpec, EnvSpec, SourceSpec
from .types import (ActionResult, ContextBatch, ControllerPrediction, DensityResult, MetricRecord,
                    ReplayBatch, UpdateResult)


@runtime_checkable
class EnvAdapter(Protocol):
    spec: EnvSpec

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]: ...

    def step(self, action_command_phys: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]: ...

    def build_context(self) -> Dict[str, np.ndarray]: ...

    def close(self) -> None: ...


@runtime_checkable
class Controller(Protocol):
    capability: CapabilitySpec

    def initial_state(self, n_envs: int) -> Dict[str, np.ndarray]: ...

    def predict(self, obs_raw: np.ndarray, reference: np.ndarray, state: Dict[str, np.ndarray]) -> ControllerPrediction: ...

    def advance(self, state: Dict[str, np.ndarray], obs_raw: np.ndarray, reference: np.ndarray,
                applied_command_norm: np.ndarray) -> Dict[str, np.ndarray]: ...


@runtime_checkable
class SourceDistribution(Protocol):
    def sample(self, sample_shape: Tuple[int, ...], generator: Optional[torch.Generator]) -> torch.Tensor: ...

    def log_prob(self, z: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class DensityEstimator(Protocol):
    def estimate_at(self, z_a: torch.Tensor, cond: torch.Tensor, net, source: SourceDistribution,
                    generator: Optional[torch.Generator]) -> DensityResult: ...


@runtime_checkable
class Agent(Protocol):
    def act(self, ctx: ContextBatch, spec: SourceSpec, generator, deterministic: bool = False) -> ActionResult: ...

    def update(self, batch: ReplayBatch, update_ctx: Dict[str, Any]) -> UpdateResult: ...

    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, sd: Dict[str, Any]) -> None: ...


@runtime_checkable
class MetricLogger(Protocol):
    def write_scalars(self, record: MetricRecord) -> None: ...

    def write_artifact(self, name: str, path: str, kind: str, meta: Optional[dict] = None) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...
