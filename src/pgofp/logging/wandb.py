"""WandB config/summary and one selected scalar route: TensorBoard sync or direct.

Disabled mode imports no SDK. Initialization errors are exposed to the caller,
which may require online logging instead of allowing local-only operation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from the repo-root .env into os.environ (existing variables win). Keys never go through configs/logs."""
    import os

    p = Path(path) if path else Path(__file__).resolve().parents[3] / ".env"
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class WandbSession:
    def __init__(self, mode: str, project: str, entity: Optional[str], group: Optional[str], tags, run_name: str,
                 config: Dict[str, Any], run_dir: str | Path, resume_id: Optional[str] = None,
                 tb_root: Optional[str | Path] = None, scalar_source: str = "tensorboard"):
        self.run = None
        self.error: Optional[str] = None
        self.mode = mode
        self.id: Optional[str] = None
        self.scalar_source = scalar_source
        if scalar_source not in ("tensorboard", "direct"):
            raise ValueError("scalar_source must be tensorboard or direct")
        if mode == "disabled":
            return
        try:
            load_dotenv()
            import wandb

            kw = dict(project=project, entity=entity, group=group, tags=list(tags), name=run_name, config=config,
                      dir=str(run_dir), mode=mode, sync_tensorboard=(scalar_source == "tensorboard"), reinit=True)
            if resume_id:
                kw["id"], kw["resume"] = resume_id, "allow"
            if tb_root is not None and scalar_source == "tensorboard":
                wandb.tensorboard.patch(root_logdir=str(tb_root))
            self.run = wandb.init(**kw)
            self.id = self.run.id if self.run is not None else None
            if self.run is not None and scalar_source == "direct":
                self.run.define_metric("charts/env_steps")
                self.run.define_metric("*", step_metric="charts/env_steps")
        except Exception as e:  # missing package, no network in online mode, auth failure ...
            self.error = f"{type(e).__name__}: {e}"
            self.run = None

    @property
    def active(self) -> bool:
        return self.run is not None

    def summary(self, values: Dict[str, Any]) -> None:
        if self.run is not None:
            for k, v in values.items():
                self.run.summary[k] = v

    def scalars(self, step: int, values: Dict[str, float]) -> None:
        """Direct mode uses an explicit environment-step axis; W&B's internal step remains monotone.

        Duplicate final/periodic eval steps are valid and do not get silently dropped.
        TensorBoard sync is disabled in this mode, so every scalar has one cloud producer.
        """
        if self.run is not None and self.scalar_source == "direct":
            import math

            finite = {k: v for k, v in values.items() if math.isfinite(v)}
            self.run.log({**finite, "charts/env_steps": step})

    def table(self, name: str, columns, rows) -> None:
        if self.run is not None:
            import wandb

            self.run.log({name: wandb.Table(columns=list(columns), data=[list(r) for r in rows])}, commit=True)

    def artifact(self, name: str, path: str | Path, kind: str, meta: Optional[dict] = None, aliases=None) -> None:
        if self.run is None:
            return
        import wandb

        art = wandb.Artifact(name, type=kind, metadata=meta or {})
        p = Path(path)
        (art.add_dir if p.is_dir() else art.add_file)(str(p))
        self.run.log_artifact(art, aliases=aliases)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception:
                pass
