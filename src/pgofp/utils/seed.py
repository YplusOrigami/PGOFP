"""Seed derivation and named RNG streams (all states checkpointable)."""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict

import numpy as np
import torch

STREAM_NAMES = (
    "env_train",
    "behavior_source",
    "replay",
    "time",
    "candidates",
    "trace",
    "target_source",
    "alpha_source",
    "eval_env",
    "eval_policy",
    "prior_train",
    "controller",
    "init",
)


def derive_seed(base: int, *names: Any) -> int:
    """Order-independent, name-based seed derivation (never depends on call order)."""
    h = hashlib.sha256(("/".join([str(base), *map(str, names)])).encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


class RngStreams:
    """Separate generators for every consumer so evaluation never consumes training randomness."""

    def __init__(self, seed: int, device: torch.device | str = "cpu", names=STREAM_NAMES):
        self.seed = int(seed)
        self.device = torch.device(device)
        self._torch: Dict[str, torch.Generator] = {}
        self._numpy: Dict[str, np.random.Generator] = {}
        for n in names:
            g = torch.Generator(device=self.device)
            g.manual_seed(derive_seed(seed, n))
            self._torch[n] = g
            self._numpy[n] = np.random.default_rng(derive_seed(seed, n, "np"))

    def torch(self, name: str) -> torch.Generator:
        return self._torch[name]

    def numpy(self, name: str) -> np.random.Generator:
        return self._numpy[name]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "torch": {k: g.get_state() for k, g in self._torch.items()},
            "numpy": {k: g.bit_generator.state for k, g in self._numpy.items()},
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        for k, st in sd["torch"].items():
            if k in self._torch:
                self._torch[k].set_state(st.cpu() if hasattr(st, "cpu") else st)
        for k, st in sd["numpy"].items():
            if k in self._numpy:
                self._numpy[k].bit_generator.state = st
