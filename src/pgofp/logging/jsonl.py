"""Append-only JSONL writers: metrics (scalars with their env-step) and events (fallbacks, NaN, resume, ...)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


class JsonlWriter:
    def __init__(self, path: str | Path, flush_every: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1 << 16)
        self._n = 0
        self.flush_every = flush_every

    def write(self, obj: Dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, default=_default, allow_nan=True) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.flush()
            os.fsync(self._f.fileno()) if hasattr(os, "fsync") else None
            self._f.close()


def _default(o):
    try:
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # truncated final line after a crash
    return out


def truncate_after_step(path: str | Path, max_step: int, discarded_path: str | Path) -> int:
    """Drop records with step > max_step (written by a crashed run past its last checkpoint); keep them aside."""
    p = Path(path)
    recs = read_jsonl(p)
    keep = [r for r in recs if r.get("step", 0) <= max_step]
    drop = [r for r in recs if r.get("step", 0) > max_step]
    if drop:
        with open(discarded_path, "a") as f:
            for r in drop:
                f.write(json.dumps(r, default=_default) + "\n")
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in keep:
                f.write(json.dumps(r, default=_default) + "\n")
        tmp.replace(p)
    return len(drop)
