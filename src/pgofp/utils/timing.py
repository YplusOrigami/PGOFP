from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict

import torch


def sync(device) -> None:
    """Block until queued device work completes so timings are not just kernel-launch times."""
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.synchronize(d)
    elif d.type == "mps":
        torch.mps.synchronize()


class StageTimer:
    """Accumulates wall time per named stage between flushes."""

    def __init__(self, device="cpu", sync_device: bool = False):
        self.device = device
        self.sync_device = sync_device
        self._acc: Dict[str, float] = defaultdict(float)
        self._cnt: Dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        if self.sync_device:
            sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_device:
                sync(self.device)
            self._acc[name] += time.perf_counter() - t0
            self._cnt[name] += 1

    def flush(self) -> Dict[str, float]:
        out = {f"timing/{k}_sec": v for k, v in self._acc.items()}
        out.update({f"timing/{k}_calls": float(c) for k, c in self._cnt.items()})
        self._acc.clear()
        self._cnt.clear()
        return out
