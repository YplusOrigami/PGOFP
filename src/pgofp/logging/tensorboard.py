"""Local scalar curves; WandB separately selects either TB sync or direct MetricRecord logging."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class TensorBoardWriter:
    def __init__(self, log_dir: str | Path, purge_step: Optional[int] = None):
        from torch.utils.tensorboard import SummaryWriter

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        kw = {} if purge_step is None else {"purge_step": purge_step}
        self.w = SummaryWriter(log_dir=str(log_dir), **kw)

    def scalars(self, step: int, values: Dict[str, float]) -> None:
        for k, v in values.items():
            if v == v and abs(v) != float("inf"):  # skip NaN/Inf: they would poison curves
                self.w.add_scalar(k, v, global_step=step)

    def histogram(self, tag: str, values, step: int) -> None:
        self.w.add_histogram(tag, values, global_step=step)

    def flush(self) -> None:
        self.w.flush()

    def close(self) -> None:
        self.w.flush()
        self.w.close()
