"""Ring replay buffer storing raw conditions + frozen prior caches (see docs: ReplayTransition).

Stored per transition: current/next context (obs_raw, reference, ctrl_feat, phase, prior_mu, prior_logstd, prior_valid),
both the policy action and the command actually sent to the env (Critic input), reward, terminated/truncated,
bootstrap mask and audit fields.  No unbounded `info` dict and no torch graphs are stored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..core.exceptions import ConfigError
from ..core.types import CONTEXT_FIELDS, ContextBatch, ContextSpec, ReplayBatch

AUDIT_FIELDS: Dict[str, Tuple[Tuple[int, ...], Any]] = {
    "episode_id": ((), np.int64),
    "step_in_episode": ((), np.int32),
    "env_id": ((), np.int16),
    "source_version": ((), np.int32),
    "beta_behavior": ((), np.float32),
    "sigma_behavior": ((), np.float32),
    "controller_valid": ((), np.float32),
    "fallback_reason": ((), np.int8),
}


class ReplayBuffer:
    def __init__(self, capacity: int, ctx_spec: ContextSpec):
        self.capacity = int(capacity)
        self.ctx_spec = ctx_spec
        d = ctx_spec.action_dim
        self.schema: Dict[str, Tuple[Tuple[int, ...], Any]] = {}
        for k, shp in ctx_spec.shapes().items():
            self.schema[k] = (shp, np.float32)
            self.schema["next_" + k] = (shp, np.float32)
        self.schema.update({
            "action_policy_norm": ((d,), np.float32),
            "action_command_norm": ((d,), np.float32),
            "reward": ((), np.float32),
            "terminated": ((), np.float32),
            "truncated": ((), np.float32),
            "bootstrap_mask": ((), np.float32),
        })
        self.schema.update(AUDIT_FIELDS)
        self.data = {k: np.zeros((self.capacity, *shp), dtype=dt) for k, (shp, dt) in self.schema.items()}
        self.ptr = 0
        self.size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------ write
    def add_batch(self, rec: Dict[str, np.ndarray]) -> None:
        n = None
        for k, (shp, _) in self.schema.items():
            if k not in rec:
                if k in AUDIT_FIELDS:
                    continue
                raise ConfigError(f"replay add missing field {k!r}")
            a = np.asarray(rec[k])
            if a.shape[1:] != shp:
                raise ConfigError(f"replay field {k!r}: expected per-sample shape {shp}, got {a.shape[1:]}")
            n = a.shape[0] if n is None else n
            if a.shape[0] != n:
                raise ConfigError("inconsistent batch sizes in replay add")
        assert n is not None
        idx = (self.ptr + np.arange(n)) % self.capacity
        for k in self.schema:
            if k in rec:
                self.data[k][idx] = rec[k]
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))
        self.total_added += n

    # ------------------------------------------------------------------ read
    def sample(self, batch_size: int, rng: np.random.Generator, device="cpu") -> ReplayBatch:
        if self.size == 0:
            raise ConfigError("cannot sample from an empty replay buffer")
        idx = rng.integers(0, self.size, size=batch_size)
        return self._gather(idx, device)

    def _gather(self, idx: np.ndarray, device) -> ReplayBatch:
        def T(k):
            return torch.as_tensor(self.data[k][idx], dtype=torch.float32, device=device)

        ctx = ContextBatch(**{k: T(k) for k in CONTEXT_FIELDS})
        nctx = ContextBatch(**{k: T("next_" + k) for k in CONTEXT_FIELDS})
        return ReplayBatch(
            ctx=ctx, next_ctx=nctx, action_policy_norm=T("action_policy_norm"), action_command_norm=T("action_command_norm"),
            reward=T("reward"), terminated=T("terminated"), truncated=T("truncated"), bootstrap_mask=T("bootstrap_mask"),
            extras={"episode_id": torch.as_tensor(self.data["episode_id"][idx], device=device),
                    "step_in_episode": torch.as_tensor(self.data["step_in_episode"][idx], device=device)},
        )

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ptr=self.ptr, size=self.size, total_added=self.total_added, capacity=self.capacity,
                 **{k: v[: self.size] if self.size < self.capacity else v for k, v in self.data.items()})
        tmp.replace(path)

    def load(self, path: str | Path) -> None:
        z = np.load(path)
        if int(z["capacity"]) != self.capacity:
            raise ConfigError(f"replay capacity mismatch: {int(z['capacity'])} vs {self.capacity}")
        size = int(z["size"])
        for k in self.schema:
            if k in z.files:
                self.data[k][:size] = z[k][:size]
        self.ptr, self.size, self.total_added = int(z["ptr"]), size, int(z["total_added"])
