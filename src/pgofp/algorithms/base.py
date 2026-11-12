from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch

from ..core.specs import SourceSpec
from ..core.types import ActionResult, ContextBatch, ReplayBatch, UpdateResult
from ..utils.normalization import NormSnapshot
from ..utils.seed import RngStreams


@dataclass
class UpdateContext:
    """Everything one learner update needs, frozen at the start of the update."""

    spec: SourceSpec  # one immutable SourceSpec snapshot for Actor / Critic target / density / alpha
    norm: NormSnapshot  # one normalizer snapshot
    rng: RngStreams
    env_transitions: int = 0
    do_actor: bool = True
    log_step: bool = False


class BaseAgent(ABC):
    off_policy = True

    @abstractmethod
    def act(self, ctx: ContextBatch, spec: SourceSpec, norm: NormSnapshot, generator, *, deterministic: bool = False,
            K: int = 1) -> ActionResult: ...

    @abstractmethod
    def update(self, batch: ReplayBatch, uctx: UpdateContext) -> UpdateResult: ...

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, sd: Dict[str, Any]) -> None: ...

    def train_mode(self) -> None: ...

    def eval_mode(self) -> None: ...
