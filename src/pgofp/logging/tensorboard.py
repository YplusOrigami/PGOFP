"""Local scalar curves; WandB separately selects either TB sync or direct MetricRecord logging."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def use_tensorboard_stub() -> None:
    """Select TensorBoard's supported no-TensorFlow compatibility backend.

    TensorBoard otherwise prefers an installed TensorFlow, even in PyTorch-only
    jobs. A server's unrelated TF wheel may have an incompatible NumPy ABI.
    The `notf` marker is checked by tensorboard.compat before it imports TF:
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/__init__.py
    This affects TensorBoard's backend only; it does not replace tensorflow.
    """
    import sys
    from types import ModuleType

    sys.modules.setdefault("tensorboard.compat.notf", ModuleType("tensorboard.compat.notf"))


# Install the marker before W&B or PyTorch can resolve TensorBoard's lazy TF API.
use_tensorboard_stub()


class TensorBoardWriter:
    def __init__(self, log_dir: str | Path, purge_step: Optional[int] = None):
        from importlib.metadata import version

        from packaging.version import Version

        tb_version = version("tensorboard")
        if Version(tb_version) < Version("2.18"):
            raise RuntimeError(f"TensorBoard {tb_version} is too old for this project's NumPy 2-compatible backend; "
                               "install tensorboard>=2.18 with the Python environment running this job")
        use_tensorboard_stub()
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
