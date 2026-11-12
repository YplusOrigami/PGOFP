"""Cache of FROZEN controller outputs keyed by the full input content (obs/reference/state/version).

Only deterministic, frozen controller parameters are cached; random source samples (eps) never are.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np


class PriorCache:
    def __init__(self, version: str, max_items: int = 100_000):
        self.version = version
        self.max_items = max_items
        self._d: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, obs: np.ndarray, ref: np.ndarray, state_vec: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(self.version.encode())
        for a in (obs, ref, state_vec):
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        return h.hexdigest()

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        v = self._d.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        self._d.move_to_end(key)
        return v

    def put(self, key: str, value: Dict[str, np.ndarray]) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)
