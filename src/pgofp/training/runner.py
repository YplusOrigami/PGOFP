"""Training lifecycle (docs section 8.1). The runner orchestrates; loss formulas live in algorithms/objectives.py."""
from __future__ import annotations

import csv
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from ..algorithms.base import UpdateContext
from ..algorithms.factory import build_agent
from ..config.compose import config_hash, save_resolved, to_container
from ..config.validate import validate_config
from ..controllers.factory import build_controllers
from ..controllers.manager import ControllerManager
from ..core.exceptions import ConfigError, NumericalError, ResumeIncompatibleError
from ..core.specs import SourceSpec
from ..core.types import ContextBatch, ContextSpec, MetricRecord
from ..data.replay_buffer import ReplayBuffer
from ..envs.factory import make_pool
from ..evaluation.evaluator import Evaluator, summarize
from ..logging.logger import Logger
from ..networks.conditioning import ConditionBuilder, ConditionConfig
from ..priors.schedules import ScheduleConfig, SourceScheduler
from ..utils.normalization import RunningNorm
from ..utils.provenance import collect_provenance, utc_now
from ..utils.seed import RngStreams, seed_everything
from ..utils.timing import StageTimer
from ..utils.torch_utils import select_device
from .checkpoint import CheckpointManager, check_resume_compat, compat_fingerprint
from .collector import Collector
from .hooks import IntervalTrigger
from .update_scheduler import UpdateScheduler


def derive_prior_std_mode(cfg) -> str:
    p = cfg.prior
    if p.type == "ppo":
        return "native" if p.distribution == "native" else "fixed"
    if p.type == "multi":
        modes = {("native" if c.get("type") == "ppo" and c.get("distribution", "native") == "native" else "fixed") for c in p.components}
        return modes.pop()
    return "fixed"


def make_run_name(cfg) -> str:
    beta = f"b{cfg.source.beta:g}" if cfg.source.kind != "gaussian" else "gauss"
    return f"{cfg.env.id}__{cfg.algorithm.name}__{cfg.prior.type}-{beta}__s{cfg.seed}__{config_hash(cfg)[:8]}"


class Runner:
    def __init__(self, cfg, run_dir: Optional[str | Path] = None):
        self.cfg = cfg
        self.warnings = validate_config(cfg)
        self.name = cfg.run_name or make_run_name(cfg)
        if run_dir is None:
            self.run_id = f"{self.name}__{uuid.uuid4().hex[:6]}"
            run_dir = Path(cfg.output_dir) / self.run_id
        else:
            self.run_id = Path(run_dir).name
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.parent_run_id: Optional[str] = None
        self.outcome = "created"
        self.env_transitions = 0
        self._resume_state: Optional[Dict[str, Any]] = None
        self._built = False

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, overrides: Optional[List[str]] = None, run_dir: Optional[str | Path] = None,
                        mode: str = "episode_restart") -> "Runner":
        """Rebuild a runner (config, networks, normalizer, frozen source schedule) from a checkpoint for evaluation / export."""
        import yaml
        from omegaconf import OmegaConf

        from ..config.compose import validate_schema

        state = CheckpointManager(Path(ckpt_path).parent.parent).load(ckpt_path)
        cfg = OmegaConf.create(yaml.safe_load(state["cfg_yaml"]))
        OmegaConf.set_struct(cfg, False)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        cfg = validate_schema(cfg)
        r = cls(cfg, run_dir=run_dir)
        r.build()
        r.load_state(state, mode)
        return r

    def evaluate_frozen(self, episodes: int, seed_base: int, *, deterministic: bool = False, K: Optional[int] = None,
                        max_steps: Optional[int] = None):
        """Evaluation with the checkpoint's FROZEN source parameters and normalizer (no schedule progress, no stat updates)."""
        spec = self.source_sched.spec_at(None) if self.source_sched._last is not None else self.source_sched.spec_at(self.env_transitions)
        snap = self.norm.snapshot(self.device)
        return self.evaluator.run(self.make_policy_fn(spec, snap, deterministic=deterministic, K=K), episodes, seed_base, max_steps), spec

    # ------------------------------------------------------------------ construction
    def build(self) -> "Runner":
        cfg = self.cfg
        seed_everything(cfg.seed)
        self.device = select_device(cfg.device)
        self.rng = RngStreams(cfg.seed, self.device)
        n_envs = cfg.train.num_envs
        self.pool = make_pool(cfg, n_envs, cfg.seed, "train")
        self.env_spec = self.pool.spec
        self.eval_pool = make_pool(cfg, 1, cfg.seed, "eval")
        ctrls, comp_w = build_controllers(cfg, self.env_spec, cfg.seed)
        ctrls_eval, _ = build_controllers(cfg, self.env_spec, cfg.seed)  # separate instances for evaluation
        self.manager = ControllerManager(ctrls, n_envs, self.env_spec.action_dim, cfg.algorithm.atanh_eps)
        self.eval_manager = ControllerManager(ctrls_eval, 1, self.env_spec.action_dim, cfg.algorithm.atanh_eps)
        if cfg.source.kind == "prior_mixture" and self.manager.n_prior == 0:
            raise ConfigError("prior source requested but no controller was built")
        self.ctx_spec = ContextSpec(obs_dim=self.env_spec.obs_dim, action_dim=self.env_spec.action_dim,
                                    ref_dim=self.env_spec.ref_dim, ctrl_dim=self.manager.ctrl_dim,
                                    phase_dim=self.env_spec.phase_dim, n_prior=self.manager.n_prior)
        s = cfg.source
        self.source_sched = SourceScheduler(ScheduleConfig(
            source_kind=s.kind, kind=s.schedule, beta=s.beta, sigma_prior=s.sigma_prior, sigma_base=s.sigma_base,
            sigma_min=s.sigma_min, sigma_max=s.sigma_max, prior_std_mode=derive_prior_std_mode(cfg),
            std_multiplier=s.std_multiplier, component_weights=comp_w or (1.0,), total_steps=s.total_steps,
            beta_start=s.beta_start, beta_end=s.beta_end, sigma_start=s.sigma_start, sigma_end=s.sigma_end,
            points=[list(p) for p in s.points], max_delta=s.max_delta))
        self.builder = ConditionBuilder(self.ctx_spec, ConditionConfig(**OmegaConf.to_container(cfg.condition)),
                                        schedule_is_constant=(s.schedule == "constant"))
        self.agent = build_agent(cfg, self.env_spec, self.builder, self.device)
        self.replay = ReplayBuffer(cfg.replay.capacity, self.ctx_spec)
        self.norm = RunningNorm(self.env_spec.obs_dim, cfg.env.obs_clip, enabled=cfg.env.normalize_obs)
        self.collector = Collector(cfg, self.pool, self.manager, self.env_spec, self.replay, self.norm, self.rng, self.device)
        self.evaluator = Evaluator(self.eval_pool, self.eval_manager, self.env_spec)
        utd = 0.0 if cfg.algorithm.name == "controller_only" else cfg.train.utd_ratio
        self.update_sched = UpdateScheduler(utd, cfg.train.learning_starts, cfg.train.actor_update_interval)
        self.triggers = {"log": IntervalTrigger(cfg.train.log_interval), "eval": IntervalTrigger(cfg.eval.interval),
                         "ckpt": IntervalTrigger(cfg.checkpoint.interval)}
        self.ckpt = CheckpointManager(self.run_dir)
        self.timer = StageTimer(self.device, cfg.train.sync_device_timing)
        self.best_eval = -float("inf")
        self.compat_extra = {"controllers": self.manager.versions}
        self._built = True
        return self

    # ------------------------------------------------------------------ state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent.state_dict(), "normalizer": self.norm.state_dict(),
            "source_sched": self.source_sched.state_dict(), "update_sched": self.update_sched.state_dict(),
            "rng": self.rng.state_dict(), "collector": self.collector.state_dict(), "manager": self.manager.state_dict(),
            "triggers": {k: t.state_dict() for k, t in self.triggers.items()},
            "counters": {"env_transitions": self.env_transitions, "vector_steps": self.collector.vector_steps,
                         "gradient_steps": self.update_sched.gradient_steps,
                         "actor_updates": getattr(self.agent, "actor_updates", 0),
                         "critic_updates": getattr(self.agent, "critic_updates", 0)},
            "best_eval": self.best_eval,
            "cfg_yaml": OmegaConf.to_yaml(self.cfg), "compat": compat_fingerprint(self.cfg, self.compat_extra),
            "run": {"run_id": self.run_id, "parent_run_id": self.parent_run_id, "wandb_id": self.logger.wandb_id,
                    "last_logged_step": self.logger.last_step},
            "controller_versions": self.manager.versions,
        }

    def load_state(self, state: Dict[str, Any], mode: str) -> None:
        check_resume_compat(state, self.cfg, self.compat_extra, mode)
        self.agent.load_state_dict(state["agent"])
        self.norm.load_state_dict(state["normalizer"])
        if mode == "weights_only":
            return
        self.source_sched.load_state_dict(state["source_sched"])
        self.update_sched.load_state_dict(state["update_sched"])
        self.rng.load_state_dict(state["rng"])
        for k, t in self.triggers.items():
            t.load_state_dict(state["triggers"][k])
        c = state["counters"]
        self.env_transitions = c["env_transitions"]
        self.best_eval = state.get("best_eval", -float("inf"))
        self.parent_run_id = state["run"]["run_id"] if state["run"]["run_id"] != self.run_id else state["run"].get("parent_run_id")
        self._resume_state = state

    # ------------------------------------------------------------------ main loop
    def train(self) -> Dict[str, Any]:
        if not self._built:
            self.build()
        cfg = self.cfg
        resume_step = self.env_transitions if self._resume_state is not None else None
        wandb_id = self._resume_state["run"].get("wandb_id") if self._resume_state else None
        self.logger = Logger(self.run_dir, cfg, to_container(cfg), self.name, resume_step=resume_step, wandb_id=wandb_id)
        save_resolved(cfg, self.run_dir / "config_resolved.yaml")
        self._write_manifest("running")
        for w in self.warnings:
            self.logger.event("config_warning", message=w)
        t_start = time.time()
        self._wall_offset = (self._resume_state or {}).get("wall_time", 0.0)
        self.outcome = "running"
        try:
            if cfg.eval.prior_before_training:
                from ..evaluation.prior_baseline import evaluate_prior

                evaluate_prior(cfg, self.run_dir, self.logger, self.env_transitions)
            self._start_collection()
            self._loop()
            self.outcome = "completed"
        except KeyboardInterrupt:
            self.outcome = "interrupted"
            self.logger.progress_file.fail("用户中断")
            self.logger.event("interrupted", env_transitions=self.env_transitions)
        except NumericalError as e:
            self.outcome = "failed_numerical"
            self.logger.progress_file.fail(str(e))
            self.logger.event("numerical_error", env_transitions=self.env_transitions, error=str(e))
            (self.run_dir / "numerical_failure.json").write_text(json.dumps({"step": self.env_transitions, "error": str(e),
                                                                           "alpha": getattr(getattr(self.agent, 'temp', None), 'alpha', None)}))
            self._finish(t_start)
            raise
        except Exception as e:
            self.outcome = "failed"
            self.logger.progress_file.fail(f"{type(e).__name__}: {e}")
            self.logger.event("exception", env_transitions=self.env_transitions, error=f"{type(e).__name__}: {e}")
            self._finish(t_start)
            raise
        return self._finish(t_start)

    def _start_collection(self) -> None:
        st = self._resume_state
        exact = self.cfg.resume_mode == "exact_resume" and st is not None
        restored = False
        if st is not None:
            rp = self.ckpt.replay_path("last")
            if rp is not None and self.cfg.resume_mode != "weights_only":
                self.replay.load(rp)
            if exact:
                self.manager.load_state_dict(st["manager"])
                restored = self.collector.load_state_dict(st["collector"], exact=True)
                if not restored:
                    raise ResumeIncompatibleError("exact_resume requires an environment with get/set_snapshot (not available here); "
                                                  "use resume_mode=episode_restart")
                self.logger.event("resume", mode="exact_resume", env_transitions=self.env_transitions)
            else:
                self.collector.vector_steps = st["collector"]["vector_steps"]
                self.pool._next_episode = st["collector"]["next_episode"]
                self.logger.event("resume", mode=self.cfg.resume_mode, env_transitions=self.env_transitions,
                                  note="episodes restarted; trajectory continuation is not bit-exact")
        if not restored:
            self.collector.reset()

    def _loop(self) -> None:
        cfg, tr = self.cfg, self.cfg.train
        acc: Dict[str, List[float]] = defaultdict(list)
        t_loop = time.time()
        steps_at_loop = self.env_transitions
        last_log_wall, last_log_steps = time.time(), self.env_transitions
        while self.env_transitions < tr.total_env_steps:
            self.logger.progress("warmup" if self.env_transitions < tr.learning_starts else "flow_training",
                                 self.env_transitions, tr.learning_starts if self.env_transitions < tr.learning_starts else tr.total_env_steps,
                                 message=f"gradient_steps={self.update_sched.gradient_steps}")
            spec = self.source_sched.spec_at(self.env_transitions)
            warm = self.env_transitions < tr.learning_starts
            with self.timer.stage("collect"):
                new = self.collector.step(self.agent, spec, self.norm.snapshot(self.device), self.env_transitions, warm)
            self.env_transitions += new
            n_up = self.update_sched.n_updates(new, self.env_transitions)
            first = True
            for _ in range(n_up):
                with self.timer.stage("update"):
                    spec_u = self.source_sched.spec_at(self.env_transitions)
                    batch = self.replay.sample(tr.batch_size, self.rng.numpy("replay"), self.device)
                    uctx = UpdateContext(spec_u, self.norm.snapshot(self.device), self.rng, self.env_transitions,
                                         do_actor=self.update_sched.do_actor(),
                                         log_step=first and self.triggers["log"].due(self.env_transitions))
                    res = self.agent.update(batch, uctx)
                    self.update_sched.tick()
                    first = False
                    for k, v in res.metrics.items():
                        acc[k].append(v)
            self.logger.progress("warmup" if warm else "flow_training", self.env_transitions,
                                 tr.learning_starts if warm else tr.total_env_steps,
                                 message=f"gradient_steps={self.update_sched.gradient_steps}")
            if self.triggers["log"].due(self.env_transitions):
                now = time.time()
                scalars = {k: float(np.mean(v)) for k, v in acc.items() if v}
                acc.clear()
                scalars.update(self.collector.drain_episode_stats())
                scalars.update(self._prior_stats(spec))
                scalars.update({
                    "charts/env_steps": float(self.env_transitions), "charts/gradient_steps": float(self.update_sched.gradient_steps),
                    "charts/vector_steps": float(self.collector.vector_steps), "charts/replay_size": float(len(self.replay)),
                    "charts/steps_per_second": (self.env_transitions - last_log_steps) / max(now - last_log_wall, 1e-9),
                    "charts/wall_time_sec": self._wall_offset + now - t_loop,
                    "charts/utd_credit": self.update_sched.credit,
                })
                scalars.update(self.timer.flush())
                last_log_wall, last_log_steps = now, self.env_transitions
                self.logger.write_scalars(MetricRecord(self.env_transitions, scalars, wall_time=scalars["charts/wall_time_sec"]))
                self.triggers["log"].fire(self.env_transitions)
            if self.triggers["eval"].due(self.env_transitions) and self.env_transitions >= tr.learning_starts:
                with self.timer.stage("eval"):
                    self._evaluate()
                self.triggers["eval"].fire(self.env_transitions)
            elif self.triggers["eval"].due(self.env_transitions):
                self.triggers["eval"].fire(self.env_transitions)  # nothing meaningful to evaluate during warmup
            if self.triggers["ckpt"].due(self.env_transitions):
                with self.timer.stage("checkpoint"):
                    self.save_checkpoint("last")
                self.triggers["ckpt"].fire(self.env_transitions)

    def _prior_stats(self, spec: SourceSpec) -> Dict[str, float]:
        out = {"prior/beta": spec.beta, "prior/sigma_prior": spec.sigma_prior, "prior/source_version": float(spec.version)}
        out.update(self.manager.stats())
        self.manager.reset_stats()
        u = self.collector.last_component_usage
        if u is not None:
            for i, x in enumerate(np.asarray(u).ravel()):
                out[f"prior/component_usage_{i}"] = float(x)
        return out

    # ------------------------------------------------------------------ evaluation
    def make_policy_fn(self, spec: SourceSpec, norm_snap, *, deterministic: bool = False, K: Optional[int] = None):
        K = self.cfg.algorithm.K_eval if K is None else K

        def policy(ctx_np):
            res = self.agent.act(ContextBatch.from_numpy(ctx_np, device=self.device), spec, norm_snap,
                                 self.rng.torch("eval_policy"), deterministic=deterministic, K=K)
            return res.action_norm.cpu().numpy()

        return policy

    def _evaluate(self) -> Dict[str, float]:
        ecfg = self.cfg.eval
        spec = self.source_sched.spec_at(None) if self.source_sched._last is not None else self.source_sched.spec_at(self.env_transitions)
        snap = self.norm.snapshot(self.device)  # frozen for the whole evaluation
        def progress(done, total, episode_step, cap):
            self.logger.progress("flow_evaluation", done, total, unit="episodes",
                                 message=f"env_steps={self.env_transitions}; episode_step={episode_step}/{cap}")

        self.logger.progress("flow_evaluation", 0, ecfg.episodes, unit="episodes")
        results = self.evaluator.run(self.make_policy_fn(spec, snap), ecfg.episodes, ecfg.seed_base, ecfg.max_steps, progress=progress)
        scalars = summarize(results, "eval")
        if ecfg.representative_action:
            rep = self.evaluator.run(self.make_policy_fn(spec, snap, deterministic=True), min(ecfg.episodes, 5), ecfg.seed_base, ecfg.max_steps,
                                     progress=progress)
            scalars.update(summarize(rep, "eval_representative"))
        self._append_episodes(results)
        self.logger.write_scalars(MetricRecord(self.env_transitions, scalars))
        score = scalars.get("eval/return_mean", -float("inf"))
        if score > self.best_eval:
            self.best_eval = score
            self.save_checkpoint("best", with_replay=False)
        return scalars

    def _append_episodes(self, results) -> None:
        p = self.run_dir / "evaluation" / "episodes.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.is_file()
        with open(p, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["checkpoint_step", "eval_seed", "return", "length", "terminated", "truncated"])
            for r in results:
                w.writerow([self.env_transitions, r.seed, r.ret, r.length, int(r.terminated), int(r.truncated)])

    # ------------------------------------------------------------------ persistence
    def save_checkpoint(self, name: str, with_replay: Optional[bool] = None) -> Path:
        self.logger.progress("checkpoint", 0, 1, unit="files", message=name)
        with_replay = self.cfg.checkpoint.save_replay if with_replay is None else with_replay
        state = self.state_dict()
        state["wall_time"] = self._wall_offset + 0.0
        path = self.ckpt.save(name, state, replay=self.replay if (with_replay and name == "last") else None)
        self.logger.event("checkpoint", name=name, env_transitions=self.env_transitions, path=str(path))
        self.logger.progress("checkpoint", 1, 1, unit="files", message=name)
        return path

    def _write_manifest(self, outcome: str, extra: Optional[Dict[str, Any]] = None) -> None:
        m = {"run_id": self.run_id, "parent_run_id": self.parent_run_id, "outcome": outcome, "config_hash": config_hash(self.cfg),
             "provenance": collect_provenance(), "env_spec_hash": self.env_spec.hash if self._built else None,
             "controller_versions": self.manager.versions if self._built else None, "smoke": bool(self.cfg.smoke),
             "updated_utc": utc_now(), **(extra or {})}
        wb = getattr(getattr(self, "logger", None), "wandb", None)
        if wb is not None and wb.active:
            m["wandb"] = {"run_id": wb.id, "url": wb.run.url, "project": wb.run.project,
                          "entity": wb.run.entity, "mode": wb.mode, "scalar_source": wb.scalar_source}
        (self.run_dir / "manifest.json").write_text(json.dumps(m, indent=2, default=str))

    def _finish(self, t_start: float) -> Dict[str, Any]:
        summary = {"outcome": self.outcome, "env_transitions": self.env_transitions,
                   "gradient_steps": self.update_sched.gradient_steps, "wall_time_sec": time.time() - t_start,
                   "best_eval_return": None if self.best_eval == -float("inf") else self.best_eval}
        if self.outcome not in ("completed", "interrupted"):
            summary["failure_stage"] = self.logger.progress_file.state.get("stage", "initializing")
        try:
            if self.outcome in ("completed", "interrupted"):
                self.save_checkpoint("last")
                if self.outcome == "completed":
                    self.save_checkpoint("final", with_replay=False)
                    final = self._evaluate() if self.env_transitions >= self.cfg.train.learning_starts else {}
                    summary["final_eval"] = final
                    (self.run_dir / "evaluation").mkdir(exist_ok=True)
                    (self.run_dir / "evaluation" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
                    self.logger.wandb.summary({k: v for k, v in summary.items() if k != "final_eval"})
        except Exception as e:
            self.outcome = "failed_numerical" if isinstance(e, NumericalError) else "failed"
            self.logger.progress_file.fail(f"{type(e).__name__}: {e}")
            summary.update(outcome=self.outcome, failure_stage=self.logger.progress_file.state.get("stage"))
            self.logger.event("finalization_failed", stage=summary["failure_stage"], error=str(e))
            raise
        finally:
            if self.outcome == "completed":
                self.logger.progress("completed", self.env_transitions, self.cfg.train.total_env_steps,
                                     status="completed", force=True)
            self._write_manifest(self.outcome, {"summary": summary})
            self.logger.close(0 if self.outcome in ("completed", "interrupted") else 1)
            self.pool.close()
            self.eval_pool.close()
        return summary
