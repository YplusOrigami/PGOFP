"""MetricLogger: training code only depends on this fan-out.

  MetricRecord -> metrics.jsonl (raw local truth, every record carries its env-step)
               -> TensorBoard SummaryWriter (global_step = env_transitions)
               -> WandB (direct records OR TensorBoard sync, never both)
  events.jsonl : fallbacks, NaN, resume, exceptions, wandb degradation notices.
Online-required experiments fail at initialization when the service is unavailable.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.types import MetricRecord
from .jsonl import JsonlWriter, truncate_after_step
from .tensorboard import TensorBoardWriter
from .wandb import WandbSession
from .progress import ProgressFile


class Logger:
    def __init__(self, run_dir: str | Path, cfg, config_dict: Dict[str, Any], run_name: str, *,
                 resume_step: Optional[int] = None, wandb_id: Optional[str] = None):
        lg = cfg.logging
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.last_step = -1
        self.progress_file = ProgressFile(self.run_dir / "progress.json", lg.console_progress)
        self.progress("initializing", 0, 1, unit="tasks", message=f"WandB={lg.wandb.mode}")
        self.events = JsonlWriter(self.run_dir / "events.jsonl", flush_every=1)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        if resume_step is not None:
            dropped = truncate_after_step(self.metrics_path, resume_step, self.run_dir / "metrics_discarded.jsonl")
            self.event("log_truncated_on_resume", resume_step=resume_step, dropped_records=dropped)
        self.metrics = JsonlWriter(self.metrics_path, flush_every=lg.flush_every) if lg.jsonl else None
        tb_dir = self.run_dir / "tensorboard"
        self.wandb = WandbSession(lg.wandb.mode, lg.wandb.project, lg.wandb.entity, lg.wandb.group,
                                  list(lg.wandb.tags) + list(cfg.tags), run_name, config_dict, self.run_dir,
                                  resume_id=wandb_id, tb_root=tb_dir if lg.wandb.mode != "disabled" else None,
                                  scalar_source=lg.wandb.scalar_source)
        if self.wandb.error:
            self.event("wandb_degraded", error=self.wandb.error, mode=lg.wandb.mode)
        actual_mode = getattr(getattr(self.wandb.run, "settings", None), "mode", lg.wandb.mode) if self.wandb.active else None
        if lg.wandb.require_online and (lg.wandb.mode != "online" or not self.wandb.active or actual_mode != "online"):
            from ..core.exceptions import ConfigError

            self.event("wandb_required_online_failed", mode=lg.wandb.mode)
            self.events.close()
            if self.metrics is not None:
                self.metrics.close()
            self.wandb.finish(1)
            raise ConfigError("WandB online logging is required but unavailable; check credentials/network. "
                              "No training was started. Local diagnostics are preserved.")
        if self.wandb.active:
            self.event("wandb_connected", run_id=self.wandb.id, url=self.wandb.run.url,
                       project=self.wandb.run.project, entity=self.wandb.run.entity, mode=lg.wandb.mode)
        self.tb = TensorBoardWriter(tb_dir, purge_step=None if resume_step is None else resume_step + 1) if lg.tensorboard else None
        self.upload_artifacts = lg.wandb.upload_artifacts

    # ------------------------------------------------------------------ writes
    def progress(self, stage: str, current: int, total: int, **kwargs) -> None:
        self.progress_file.update(stage, current, total, **kwargs)

    def write_scalars(self, record: MetricRecord) -> None:
        if record.step < self.last_step:  # env-step axis must be monotone
            self.event("non_monotone_step", step=record.step, last_step=self.last_step)
            return
        self.last_step = max(self.last_step, record.step)
        wall = record.wall_time or (time.time() - self.t0)
        if self.metrics is not None:
            self.metrics.write({"step": record.step, "wall_time": round(wall, 3), **record.scalars})
        if self.tb is not None:
            self.tb.scalars(record.step, record.scalars)
        self.wandb.scalars(record.step, record.scalars)

    def event(self, kind: str, **data) -> None:
        from ..utils.provenance import utc_now

        self.events.write({"utc": utc_now(), "kind": kind, **data})

    def write_artifact(self, name: str, path: str | Path, kind: str, meta: Optional[dict] = None, aliases=None) -> None:
        self.event("artifact", name=name, path=str(path), artifact_type=kind, meta=meta or {})
        if self.upload_artifacts:
            try:
                self.wandb.artifact(name, path, kind, meta, aliases)
            except Exception as e:
                self.event("artifact_upload_failed", name=name, error=str(e))

    def write_table(self, name: str, columns, rows) -> None:
        try:
            self.wandb.table(name, columns, rows)
        except Exception as e:
            self.event("table_upload_failed", name=name, error=str(e))

    def histogram(self, tag: str, values, step: int) -> None:
        if self.tb is not None:
            self.tb.histogram(tag, values, step)

    # ------------------------------------------------------------------ lifecycle
    def flush(self) -> None:
        for w in (self.metrics, self.events):
            if w is not None:
                w.flush()
        if self.tb is not None:
            self.tb.flush()

    def close(self, exit_code: int = 0) -> None:
        self.flush()
        if self.tb is not None:
            self.tb.close()
        if self.metrics is not None:
            self.metrics.close()
        self.events.close()
        self.wandb.finish(exit_code)

    @property
    def wandb_id(self) -> Optional[str]:
        return self.wandb.id
