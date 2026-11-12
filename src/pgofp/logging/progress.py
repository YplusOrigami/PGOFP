"""Small atomic progress snapshots for parent processes, terminals and saved logs."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

LABELS = {
    "initializing": "初始化日志", "prior_evaluation": "先验基线评估",
    "prior_evaluation_deterministic": "先验确定性评估", "warmup": "热身采样",
    "prior_evaluation_result": "先验基线结果",
    "flow_training": "Flow训练", "flow_evaluation": "Flow评估", "checkpoint": "保存检查点",
    "ppo_rollout": "PPO采样", "ppo_update": "PPO更新", "ppo_calibration": "PPO校准",
    "ppo_export": "导出PPO先验", "completed": "完成",
}


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)


def read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def bar(current: int, total: int, width: int = 18) -> str:
    fraction = min(1., max(0., current / max(total, 1)))
    n = int(width * fraction)
    return f"[{'#' * n}{'-' * (width - n)}] {current:,}/{total:,} ({fraction:.0%})"


def redact(message: str) -> str:
    key = os.environ.get("WANDB_API_KEY")
    return message.replace(key, "[REDACTED]") if key else message


def error_tail(path: Path, lines: int = 8) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            tail = f.read().decode(errors="replace").splitlines()[-lines:]
        return redact("\n".join(tail))
    except OSError:
        return "(日志尚未创建)"


class ProgressFile:
    def __init__(self, path: Path, enabled: bool, interval: float = 1.):
        self.path, self.enabled, self.interval = path, enabled, interval
        self.state: dict = {}
        self.last_write = 0.

    def update(self, stage: str, current: int, total: int, *, message: str = "", unit: str = "steps",
               status: str = "running", force: bool = False) -> None:
        changed = self.state.get("stage") != stage or self.state.get("status") != status
        reached_end = current >= total and self.state.get("current", -1) < total
        now = time.monotonic()
        self.state = {"stage": stage, "current": int(current), "total": int(total), "unit": unit,
                      "message": redact(message), "status": status, "updated_at": time.time()}
        if self.enabled and (force or changed or current >= total or now - self.last_write >= self.interval):
            atomic_json(self.path, self.state)
            self.last_write = now
            if changed or reached_end or status != "running":
                with self.path.with_name("progress_events.jsonl").open("a") as f:
                    f.write(json.dumps(self.state, ensure_ascii=False) + "\n")

    def fail(self, error: str) -> None:
        s = self.state
        self.update(s.get("stage", "initializing"), s.get("current", 0), s.get("total", 1),
                    message=error, unit=s.get("unit", "steps"), status="failed", force=True)


class ProgressDisplay:
    """Plain lines deliberately remain readable in tmux, redirected logs and schedulers."""
    def __init__(self, label: str, interval: float = 5.):
        self.label, self.interval = label, interval
        self.last_stage = None
        self.last_print = 0.
        self.event_offset = 0

    def poll(self, path: Path, *, force: bool = False) -> None:
        """Replay phase transitions, even when a short phase ends between polls."""
        journal = path.with_name("progress_events.jsonl")
        try:
            with journal.open() as f:
                f.seek(self.event_offset)
                for line in f:
                    try:
                        self.show(json.loads(line), force=True)
                    except ValueError:
                        continue
                self.event_offset = f.tell()
        except OSError:
            pass
        self.show(read_progress(path), force=force)

    def show(self, state: dict, *, force: bool = False) -> None:
        if not state:
            return
        now = time.monotonic()
        stage = state.get("stage", "initializing")
        if force or stage != self.last_stage or now - self.last_print >= self.interval:
            text = (f"{self.label} | {LABELS.get(stage, stage)} "
                    f"{bar(state.get('current', 0), state.get('total', 1))} "
                    f"{state.get('unit', '')} {state.get('message', '')}")
            print(redact(text), flush=True)
            self.last_print, self.last_stage = now, stage
