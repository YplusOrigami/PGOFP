"""Launch runs as INDEPENDENT processes with bounded concurrency, GPU assignment, state tracking and explicit retry rules.

--dry-run only writes the resolved manifest (no process is created). Numerical failures (NaN) are recorded as failed and
are NOT retried or deleted; only infrastructure-style exits (signals / OOM kill) are retried.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.provenance import utc_now
from ..logging.progress import ProgressDisplay, bar, error_tail, read_progress
from .manifest import MatrixManifest
from .matrix import RunPlan, expand_matrix, load_matrix

INFRA_EXIT_CODES = {-9, -15, 137, 143}  # SIGKILL/SIGTERM (OOM killer, scheduler preemption)


def build_manifest(matrix_name: str, matrix_dir: str | Path, **kw) -> MatrixManifest:
    spec = load_matrix(matrix_name)
    plans = expand_matrix(spec, **kw)
    man = MatrixManifest(matrix_dir, spec["name"], [p.to_record() for p in plans])
    return man


class Launcher:
    def __init__(self, manifest: MatrixManifest, repo_root: str | Path, python: str = sys.executable, max_parallel: int = 1,
                 gpus: Optional[List[int]] = None, max_retries: int = 1, script: str = "scripts/train.py",
                 show_progress: bool = False):
        self.man, self.root, self.python = manifest, Path(repo_root), python
        self.max_parallel, self.gpus, self.max_retries, self.script = max_parallel, gpus or [], max_retries, script
        self.running: Dict[str, Dict[str, Any]] = {}
        self.show_progress = show_progress

    def _cmd(self, rec: Dict[str, Any]) -> List[str]:
        runs_dir = self.man.dir / "runs"
        resume = [f"resume={rec['resume_checkpoint']}", "resume_mode=episode_restart"] if rec.get("resume_checkpoint") else []
        return [self.python, str(self.root / self.script), *rec["overrides"], *resume,
                f"run_name={rec['run_key']}", f"output_dir={runs_dir}"]

    def _start(self, rec: Dict[str, Any], slot: int) -> None:
        env = dict(os.environ)
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpus[slot % len(self.gpus)])
        (self.man.dir / "logs").mkdir(parents=True, exist_ok=True)
        log = open(self.man.dir / "logs" / f"{rec['run_key']}.log", "ab")
        proc = subprocess.Popen(self._cmd(rec), cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT)
        rec["status"], rec["started_utc"] = "running", utc_now()
        rec["pid"] = proc.pid
        print(f"[run] started {rec['run_key']} (pid={proc.pid})", flush=True)
        axes = rec.get("axes", {})
        label = f"[3/6 Flow] {axes.get('env', rec['run_key'])}/{axes.get('prior', rec['variant'])} β={axes.get('source.beta', '?')} seed={rec['seed']}"
        self.running[rec["run_key"]] = {"proc": proc, "log": log, "slot": slot,
                                         "display": ProgressDisplay(label), "last_poll": 0., "run_dir": None}
        if self.show_progress:
            print(f"{label} | 启动子进程；完整日志：{log.name}", flush=True)
        self.man.save()

    def _numerical_failure(self, rec) -> bool:
        runs = self.man.dir / "runs"
        for d in runs.glob(f"{rec['run_key']}__*"):
            m = d / "manifest.json"
            if m.is_file() and json.loads(m.read_text()).get("outcome") == "failed_numerical":
                return True
        return False

    def _run_outcome(self, rec):
        dirs = list((self.man.dir / "runs").glob(f"{rec['run_key']}__*"))
        for d in sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True):
            p = d / "manifest.json"
            if p.is_file():
                data = json.loads(p.read_text())
                rec["run_dir"] = str(d)
                return data.get("outcome")
        return None

    def _show_progress(self, rec, handle, force=False):
        now = time.monotonic()
        if not force and now - handle["last_poll"] < 1.:
            return
        handle["last_poll"] = now
        if handle["run_dir"] is None:
            dirs = list((self.man.dir / "runs").glob(f"{rec['run_key']}__*"))
            if dirs:
                handle["run_dir"] = max(dirs, key=lambda p: p.stat().st_mtime)
        if handle["run_dir"]:
            handle["display"].poll(handle["run_dir"] / "progress.json", force=force)

    def _overall_progress(self):
        c = self.man.counts()
        done = c["completed"] + c["failed"] + c["skipped"]
        print(f"[3/6 Flow总进度] {bar(done, len(self.man.runs))} "
              f"成功={c['completed']} 失败={c['failed']} 运行={c['running']} 待运行={c['planned']}", flush=True)

    def run(self, only_pending: bool = True, poll_s: float = 2.0) -> None:
        (self.man.dir / "logs").mkdir(parents=True, exist_ok=True)
        queue = [r for r in self.man.runs if r["status"] == "planned" or (not only_pending and r["status"] in ("failed", "interrupted"))]
        slot = 0
        if self.show_progress:
            self._overall_progress()
        try:
            while queue or self.running:
                while queue and len(self.running) < self.max_parallel:
                    self._start(queue.pop(0), slot)
                    slot += 1
                for key, h in list(self.running.items()):
                    if self.show_progress:
                        self._show_progress(self.man.get(key), h)
                    code = h["proc"].poll()
                    if code is None:
                        continue
                    h["log"].close()
                    rec = self.man.get(key)
                    rec["exit_code"], rec["finished_utc"] = code, utc_now()
                    del self.running[key]
                    outcome = self._run_outcome(rec)
                    if code == 0 and outcome == "interrupted":
                        rec["status"] = "interrupted"
                    elif code == 0:
                        rec["status"] = "completed"
                    elif code in INFRA_EXIT_CODES and rec["retries"] < self.max_retries and not self._numerical_failure(rec):
                        rec["retries"] += 1
                        rec["status"] = "planned"
                        queue.append(rec)
                    else:
                        rec["status"] = "failed"
                    print(f"[run] {rec['status']} {key} (exit={code})", flush=True)
                    if self.show_progress:
                        self._show_progress(rec, h, force=True)
                        if rec["status"] == "failed":
                            state = read_progress(h["run_dir"] / "progress.json") if h["run_dir"] else {}
                            rec["failure_stage"] = state.get("stage", "initializing")
                            rec["failure_message"] = state.get("message", "See process log")
                            print(f"[ERROR 3/6] {key} | 出错步骤={rec['failure_stage']} exit={code}\n"
                                  f"日志：{h['log'].name}\n{error_tail(Path(h['log'].name))}", flush=True)
                        self._overall_progress()
                    self.man.save()
                time.sleep(poll_s)
        except KeyboardInterrupt:
            for key, h in self.running.items():
                h["proc"].terminate()
                self.man.get(key)["status"] = "interrupted"
            for h in self.running.values():
                try:
                    h["proc"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    h["proc"].kill()
                    h["proc"].wait(timeout=5)
                finally:
                    h["log"].close()
            self.man.save()
            raise
        finally:
            self.man.write_failures()
            self.man.save()
