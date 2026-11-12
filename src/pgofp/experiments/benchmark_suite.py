"""Portable preparation, validation and execution of the requested 150-run suite.

Formal PPO artifacts are frozen and shared across Flow seeds. Validation artifacts
have their own directories and can never be selected by the formal plan.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import yaml

from ..config.compose import compose, config_hash, save_resolved
from ..config.validate import validate_config
from ..controllers.artifacts import load_prior_artifact
from ..core.exceptions import ConfigError
from ..logging.jsonl import read_jsonl
from ..logging.progress import ProgressDisplay, atomic_json, bar, error_tail, read_progress
from ..logging.wandb import load_dotenv
from ..utils.provenance import utc_now
from ..utils.torch_utils import select_device
from .aggregate import aggregate
from .launcher import Launcher
from .manifest import MatrixManifest, write_json_atomic
from .matrix import RunPlan
from .record_audit import audit_run

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "configs/experiment/mujoco_prior_beta_30k.yaml"


def load_suite(path: Path = SPEC) -> dict:
    s = yaml.safe_load(path.read_text())["suite"]
    if s["seeds"] != [0, 1, 2] or s["betas"] != [0, 0.25, 0.5, 0.75, 1]:
        raise ConfigError("This suite requires seeds=0/1/2 and beta=0/0.25/0.5/0.75/1")
    if len(task_pairs(s)) != 10:
        raise ConfigError("Expected six PPO, two PID and two MPC task/prior pairs")
    return s


def task_pairs(s: dict) -> list[tuple[str, str]]:
    return [(e, p) for p in ("ppo", "pid", "mpc") for e in s[f"{p}_envs"]]


def env_id(env: str) -> str:
    return compose([f"env={env}", "prior=gaussian"]).env.id


def artifact_for(s: dict, env: str, artifact_root: Path) -> Path:
    return artifact_root / f"{env}_ppo_seed{s['prior_seed']}" / f"stage_{s['prior_steps']}"


def default_prior_paths(s: dict, root: Path = ROOT) -> dict[str, Path]:
    paths = {}
    for env in s["ppo_envs"]:
        historical = artifact_for(s, env, root / "artifacts/priors")
        paths[env] = historical if historical.is_dir() else artifact_for(s, env, root / "artifacts/priors" / s["name"])
    return paths


def logging_overrides(env: str, prior: str, beta, *, online: bool, smoke: bool, entity: str | None,
                      suite_name: str) -> list[str]:
    project = "pgofp-smoke" if smoke else f"pgofp-{env}-30k"
    group = f"{suite_name}__{env}__{prior}__beta{beta:g}" if beta is not None else f"{suite_name}__{env}__ppo_pretrain"
    return ["logging=wandb_online" if online else "logging=local", f"logging.wandb.project={project}",
            f"logging.wandb.group={group}", f"logging.wandb.require_online={str(online).lower()}",
            "logging.wandb.upload_artifacts=false",  # curves/config/numeric data only; weights stay local
            "logging.console_progress=true",
            "logging.wandb.scalar_source=direct",  # avoids AppleDouble ._events poisoning TB sync on macOS exFAT
            *([f"logging.wandb.entity={entity}"] if entity else [])]


def build_plans(s: dict, prior_paths: dict[str, Path], *, device: str = "auto", online: bool | None = None,
                entity: str | None = None, smoke: bool = False) -> list[RunPlan]:
    online = bool(s.get("wandb_online", False)) if online is None else online
    plans = []
    betas = [0.0, 0.5, 1.0] if smoke else s["betas"]
    seeds = [0] if smoke else s["seeds"]
    name = s["name"] + ("_validation" if smoke else "")
    for env, prior in task_pairs(s):
        for beta in betas:
            for seed in seeds:
                b = f"{beta:g}"
                ov = list(s["flow_overrides"]) + [f"env={env}", f"prior={prior}", f"source.beta={b}",
                                                        f"seed={seed}", f"device={device}"]
                if prior == "ppo":
                    ov += ["prior.distribution=native", f"prior.artifact_path={prior_paths[env]}"]
                elif prior == "mpc":
                    ov += [f"prior.controller.seed={s['mpc_seed']}"]
                    if env == "inverted_double_pendulum":
                        ov += ["prior.controller_config=inverted_double_pendulum_mpc"]
                ov += logging_overrides(env, prior, beta, online=online, smoke=smoke, entity=entity, suite_name=s["name"])
                ov += [f"tags=[{name},budget30k,{prior},beta_sweep]", f"smoke={str(smoke).lower()}"]
                if smoke:
                    # Preserve B=256, network sizes, candidate budget and density settings.
                    # Only the interaction budget, warmup and evaluation horizon are shortened.
                    ov += ["train.total_env_steps=24", "train.learning_starts=16", "train.log_interval=8",
                           "replay.capacity=64", "eval.interval=24", "eval.episodes=1", "eval.max_steps=8",
                           "checkpoint.interval=24"]
                cfg = compose(ov)
                validate_config(cfg, check_artifacts=False)
                key = f"{name}__{env}__{prior}__beta{b}__s{seed}"
                plans.append(RunPlan(key, ov, seed, f"{prior}_beta{b}",
                                     {"env": env, "prior": prior, "source.beta": beta}, config_hash(cfg)))
    if len(plans) != (30 if smoke else 150):
        raise ConfigError("Unexpected run count")
    return plans


def write_plan(s: dict, plans: list[RunPlan], out: Path, prior_paths: dict[str, Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"[计划生成] {bar(0, len(plans))} 仅准备配置，不启动训练", flush=True)
    for index, p in enumerate(plans, 1):
        save_resolved(compose(p.overrides), out / "resolved_configs" / f"{p.run_key}.yaml")
        if index % 25 == 0 or index == len(plans):
            print(f"[计划生成] {bar(index, len(plans))}", flush=True)
    write_json_atomic(out / "plan.json", {"created_utc": utc_now(), "suite": s, "flow_run_count": len(plans),
                      "flow_steps_each": 30000, "total_flow_steps": len(plans) * 30000,
                      "prior_steps_requested_each": s["prior_steps"], "prior_seed": s["prior_seed"],
                      "prior_paths": {k: str(v) for k, v in prior_paths.items()},
                      "note": "Plan only. Missing priors are prepared before formal execution; no combinations are skipped.",
                      "runs": [p.to_record() for p in plans]})


def check_artifact(path: Path, env: str, *, min_steps: int, prior_seed: int, smoke: bool = False):
    art = load_prior_artifact(path, expect_env=env_id(env))
    if art.manifest.get("distribution") != "tanh_gaussian":
        raise ConfigError(f"{env}: expected native tanh-Gaussian PPO")
    if art.manifest.get("prior_seed") != prior_seed:
        raise ConfigError(f"{env}: prior seed does not match {prior_seed}")
    if int(art.budget.get("prior_pretrain_env_steps", 0)) < min_steps:
        raise ConfigError(f"{env}: prior has fewer than {min_steps} preparation steps")
    if not smoke and art.manifest.get("smoke", False):
        raise ConfigError("Validation priors cannot enter the formal matrix")
    return art


def safe_error(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    key = os.environ.get("WANDB_API_KEY")
    return msg.replace(key, "[REDACTED]") if key else msg


@contextmanager
def phase(out: Path, number: int, label: str):
    state = {"stage": number, "label": label, "status": "running", "updated_utc": utc_now()}
    atomic_json(out / "suite_status.json", state)
    print(f"[{number}/6 {label}] {bar(number - 1, 6)} 开始", flush=True)
    try:
        yield
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=safe_error(exc), updated_utc=utc_now())
        atomic_json(out / "suite_status.json", state)
        print(f"[ERROR {number}/6 {label}] {safe_error(exc)}", file=sys.stderr, flush=True)
        raise
    else:
        state.update(status="completed", updated_utc=utc_now())
        atomic_json(out / "suite_status.json", state)
        print(f"[{number}/6 {label}] {bar(number, 6)} 完成", flush=True)


def wandb_preflight() -> dict:
    load_dotenv()
    import wandb

    # SDK login verifies the existing credential without putting it in CLI arguments.
    api = wandb.Api(timeout=20)
    viewer = api.viewer
    username = viewer.get("username") if isinstance(viewer, dict) else getattr(viewer, "username", None)
    return {"authenticated": True, "username": username, "default_entity": api.default_entity,
            "sdk_version": wandb.__version__}


@contextmanager
def suite_lock(out: Path):
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    with (out / ".suite.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ConfigError(f"Another launcher already holds {out}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def child_env() -> dict:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_logged(cmd: list[str], log_path: Path, *, progress_path: Path | None = None, label: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=log, stderr=subprocess.STDOUT)
        try:
            display = ProgressDisplay(label)
            while proc.poll() is None:
                if progress_path:
                    display.poll(progress_path)
                time.sleep(.5)
            code = proc.returncode
            if progress_path:
                display.poll(progress_path, force=True)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        state = read_progress(progress_path) if progress_path else {}
        raise ConfigError(f"{label}: 出错步骤={state.get('stage', 'initializing')}，退出码={code}\n"
                          f"日志：{log_path}\n{error_tail(log_path)}")


def prepare_priors(s: dict, paths: dict[str, Path], out: Path, *, device: str, online: bool,
                   entity: str | None, smoke: bool) -> dict[str, Path]:
    registry_path = out / "prior_registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    paths = dict(paths)
    for index, env in enumerate(s["ppo_envs"], 1):
        print(f"[2/6 PPO先验准备] {bar(index - 1, len(s['ppo_envs']))} 当前任务={env}", flush=True)
        if env in registry and registry[env].get("status") == "ready":
            paths[env] = Path(registry[env]["path"])
        path = paths[env]
        if (path / "manifest.json").is_file():
            art = check_artifact(path, env, min_steps=s["prior_steps"], prior_seed=s["prior_seed"], smoke=smoke)
            print(f"[prior] reuse {env}: {path}", flush=True)
        else:
            # Never overwrite an interrupted preparation's checkpoints/artifacts.
            if path.parent.exists():
                base = path.parents[1] / "attempts" / f"{env}_{uuid.uuid4().hex[:8]}"
                path = artifact_for(s, env, base)
                paths[env] = path
            registry[env] = {"path": str(path), "status": "preparing", "updated_utc": utc_now()}
            write_json_atomic(registry_path, registry)
            output = out / "prior_training" / f"{env}_{uuid.uuid4().hex[:8]}"
            ov = [f"env={env}", f"seed={s['prior_seed']}", f"device={device}",
                  f"algorithm.params.total_steps={s['prior_steps']}", f"artifact_dir={path.parents[1]}",
                  f"output_dir={output}", *s["prior_overrides"]]
            ov += logging_overrides(env, "ppo", None, online=online, smoke=smoke, entity=entity, suite_name=s["name"])
            ov += [f"smoke={str(smoke).lower()}", f"tags=[{s['name']},ppo_preparation]"]
            if smoke:
                ov += ["algorithm.params.rollout_steps=64", "algorithm.params.epochs=1",
                       "algorithm.params.minibatch_size=64", "checkpoint_fractions=[1.0]", "calibration.episodes=1"]
            print(f"[prior] prepare {env}, requested steps={s['prior_steps']}; log={out / 'logs' / ('prior_' + env + '.log')}", flush=True)
            run_logged([sys.executable, str(ROOT / "scripts/train_prior.py"), *ov], out / "logs" / f"prior_{env}.log",
                       progress_path=output / f"{env}_ppo_seed{s['prior_seed']}" / "progress.json",
                       label=f"[2/6 PPO先验准备] {env} ({index}/{len(s['ppo_envs'])})")
            art = check_artifact(path, env, min_steps=s["prior_steps"], prior_seed=s["prior_seed"], smoke=smoke)
        registry[env] = {"path": str(path), "status": "ready", "content_hash": art.content_hash,
                         "actual_env_steps": art.budget["prior_pretrain_env_steps"], "prior_seed": s["prior_seed"],
                         "validation_only": smoke, "updated_utc": utc_now()}
        write_json_atomic(registry_path, registry)
        print(f"[2/6 PPO先验准备] {bar(index, len(s['ppo_envs']))} {env} 已就绪，实际步数={registry[env]['actual_env_steps']:,}", flush=True)
    return paths


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def recover_manifest(man: MatrixManifest, plans: list[RunPlan]) -> None:
    expected = {p.run_key: p for p in plans}
    if set(expected) != {r["run_key"] for r in man.runs}:
        raise ConfigError("Existing matrix has different tasks; select a fresh --output-dir")
    for rec in man.runs:
        if rec["config_hash"] != expected[rec["run_key"]].config_hash:
            raise ConfigError(f"Config changed for {rec['run_key']}; use a new output directory")
        # Include logging requirements in compatibility even though numeric config_hash excludes logging.
        old = compose(rec["overrides"])
        new = compose(expected[rec["run_key"]].overrides)
        if rec["status"] == "running" and _process_alive(rec.get("pid")):
            raise ConfigError(f"A previous training child is still alive (PID {rec['pid']}); stop it before relaunch")
        disabling_cloud = old.logging.wandb.mode != "disabled" and new.logging.wandb.mode == "disabled"
        if old.logging.wandb.mode != new.logging.wandb.mode and not disabling_cloud:
            raise ConfigError("Enabling cloud for an existing local-only matrix requires a fresh output directory")
        if not disabling_cloud and old.logging.wandb.scalar_source != new.logging.wandb.scalar_source:
            raise ConfigError("Cloud scalar route changed; use a fresh output directory")
        if disabling_cloud:
            rec.setdefault("logging_history", []).append({"updated_utc": utc_now(),
                "previous_mode": old.logging.wandb.mode, "new_mode": "disabled",
                "previous_overrides": list(rec["overrides"])})
            rec["overrides"] = list(expected[rec["run_key"]].overrides)
            # Retry only an authenticated-startup failure with zero collected data.
            # Numerical/training failures and completed results remain untouched.
            if rec["status"] == "failed":
                dirs = sorted((man.dir / "runs").glob(f"{rec['run_key']}__*"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
                if dirs:
                    latest = dirs[0]
                    events = read_jsonl(latest / "events.jsonl") if (latest / "events.jsonl").is_file() else []
                    metrics = read_jsonl(latest / "metrics.jsonl") if (latest / "metrics.jsonl").is_file() else []
                    auth_failed = any(e.get("kind") == "wandb_required_online_failed" for e in events)
                    has_checkpoint = any(p.is_file() and not p.name.startswith("._")
                                         for p in (latest / "checkpoints").rglob("*"))
                    if (auth_failed and not metrics and not has_checkpoint
                            and not (latest / "numerical_failure.json").exists()):
                        rec["status"] = "planned"
                        rec["retry_note"] = "WandB startup failed before collecting data; retry with cloud disabled"
            print(f"[日志配置迁移] {rec['run_key']}: {old.logging.wandb.mode} -> disabled；状态={rec['status']}", flush=True)
        if rec["status"] not in ("running", "interrupted"):
            continue
        dirs = sorted((man.dir / "runs").glob(f"{rec['run_key']}__*"), key=lambda p: p.stat().st_mtime, reverse=True)
        latest = dirs[0] if dirs else None
        meta = latest / "manifest.json" if latest else None
        outcome = json.loads(meta.read_text()).get("outcome") if meta and meta.is_file() else None
        if outcome in ("completed", "failed_numerical", "failed"):
            rec["status"] = "completed" if outcome == "completed" else "failed"
        else:
            rec["status"] = "planned"
            checkpoint = latest / "checkpoints/last.pt" if latest else None
            if checkpoint and checkpoint.is_file():
                rec["resume_checkpoint"] = str(checkpoint)
                rec["resume_note"] = "episode_restart: replay/model/counters restored, environment episodes restart"
        rec["run_dir"] = str(latest) if latest else None
    man.save()


def verify_local(man: MatrixManifest, expected_steps: int) -> dict:
    checks = []
    for index, rec in enumerate(man.runs, 1):
        dirs = sorted((man.dir / "runs").glob(f"{rec['run_key']}__*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not dirs:
            checks.append({"run_key": rec["run_key"], "ok": False, "reason": "no run directory"})
            continue
        d = dirs[0]
        meta = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").is_file() else {}
        records = read_jsonl(d / "metrics.jsonl") if (d / "metrics.jsonl").is_file() else []
        ev = [r for r in records if "eval/return_mean" in r]
        losses = [r for r in records if "loss/actor_qrmf" in r]
        cfg = compose(rec["overrides"])
        audit = audit_run(d, cfg, records)
        ok = (meta.get("outcome") == "completed" and meta.get("summary", {}).get("env_transitions") == expected_steps
              and bool(ev) and bool(losses) and audit["ok"]
              and bool(list((d / "tensorboard").glob("events.out.tfevents.*"))))
        checks.append({"run_key": rec["run_key"], "ok": ok, "run_dir": str(d), "eval_records": len(ev),
                       "update_log_records": len(losses), "wandb": meta.get("wandb"), "record_audit": audit,
                       "prior_baseline_required": cfg.eval.prior_before_training,
                       "representative_eval_required": cfg.eval.representative_action})
        print(f"[5/6 本地记录核验] {bar(index, len(man.runs))} {'通过' if ok else 'FAILED'} {rec['run_key']}", flush=True)
    return {"all_passed": all(r["ok"] for r in checks), "checks": checks}


def verify_cloud(local_report: dict) -> dict:
    import wandb

    api = wandb.Api(timeout=20)
    checks = []
    for index, rec in enumerate(local_report["checks"], 1):
        meta = rec.get("wandb")
        if not meta:
            checks.append({"run_key": rec["run_key"], "ok": False, "reason": "no online run recorded"})
            continue
        path = f"{meta['entity']}/{meta['project']}/{meta['run_id']}"
        ok = False
        for attempt in range(3):
            run = api.run(path)
            # Query sparse evaluations separately: a sample across all history
            # can miss four eval records among hundreds of training records.
            visible = []
            required = ["eval/return_mean", "loss/actor_qrmf"]
            if rec.get("prior_baseline_required"):
                required.append("prior_eval/return_mean")
            if rec.get("representative_eval_required"):
                required.append("eval_representative/return_mean")
            for key in required:
                rows = run.history(keys=[key, "charts/env_steps"], samples=100, pandas=False)
                visible.append(any(row.get(key) is not None and row.get("charts/env_steps") is not None for row in rows))
            ok = all(visible) and run.state == "finished"
            if ok:
                break
            if attempt < 2:
                time.sleep(3)
        checks.append({"run_key": rec["run_key"], "ok": ok, "url": meta["url"], "state": run.state,
                       "eval_and_loss_curves_visible": ok})
        print(f"[6/6 云端记录核验] {bar(index, len(local_report['checks']))} {'通过' if ok else 'FAILED'} {rec['run_key']}", flush=True)
    return {"all_passed": all(r["ok"] for r in checks), "checks": checks}


def execute(s: dict, out: Path, *, device: str, online: bool, entity: str | None,
            smoke: bool, max_parallel: int) -> int:
    with suite_lock(out):
        print(f"模式={'短验证（不是正式训练）' if smoke else '正式训练'} | 输出目录={out}", flush=True)
        with phase(out, 1, "设备与WandB检查" if online else "设备与本地日志检查"):
            selected = select_device(device)
            if selected.type == "cuda":
                import torch
                if not torch.cuda.is_available():
                    raise ConfigError("device=cuda requested but CUDA is unavailable; refusing a silent CPU fallback")
            preflight = {"created_utc": utc_now(), "device_requested": device, "device_selected": str(selected),
                         "validation_only": smoke, "wandb_online_required": online}
            if online:
                preflight["wandb"] = wandb_preflight()
                if entity is None:
                    entity = preflight["wandb"]["default_entity"]
            write_json_atomic(out / "preflight.json", preflight)
            print(f"[1/6] device={selected}, online={online}", flush=True)
        if smoke:
            s = dict(s, prior_steps=64)
            paths = {e: artifact_for(s, e, out / "validation_priors") for e in s["ppo_envs"]}
        else:
            paths = default_prior_paths(s)
        with phase(out, 2, "PPO先验准备"):
            paths = prepare_priors(s, paths, out, device=str(selected), online=online, entity=entity, smoke=smoke)
        man = None
        try:
            with phase(out, 3, "逐run先验评估与Flow训练"):
                plans = build_plans(s, paths, device=device, online=online, entity=entity, smoke=smoke)
                # No skipped combinations after preparation: load and hash-check all six priors.
                for env, path in paths.items():
                    check_artifact(path, env, min_steps=s["prior_steps"], prior_seed=s["prior_seed"], smoke=smoke)
                for p in plans:
                    validate_config(compose(p.overrides), check_artifacts=True)
                    save_resolved(compose(p.overrides), out / "resolved_configs" / f"{p.run_key}.yaml")
                if (out / "manifest.json").is_file():
                    man = MatrixManifest.load(out / "manifest.json")
                    recover_manifest(man, plans)
                else:
                    man = MatrixManifest(out, s["name"] + ("_validation" if smoke else ""), [p.to_record() for p in plans])
                    man.save()
                print(f"[3/6] {len(plans)} runs; 待运行={man.counts()['planned']}; 并发={max_parallel}", flush=True)
                Launcher(man, ROOT, python=sys.executable, max_parallel=max_parallel, max_retries=0,
                         show_progress=True).run(poll_s=0.5)
        finally:
            if man is not None:
                previous = read_progress(out / "suite_status.json")
                with phase(out, 4, "汇总实验数据"):
                    display = ProgressDisplay("[4/6 日志读取与seed汇总]")
                    aggregate(man, progress=lambda done, total: display.show(
                        {"stage": "aggregate", "current": done, "total": total, "unit": "runs"}, force=done >= total))
                if previous.get("status") in ("failed", "interrupted"):
                    atomic_json(out / "suite_status.json", previous)
        with phase(out, 5, "本地记录核验"):
            local = verify_local(man, 24 if smoke else 30000)
        report = {"created_utc": utc_now(), "validation_only": smoke, "counts": man.counts(), "local": local,
                  "wandb_mode": "online" if online else "disabled"}
        write_json_atomic(out / "verification.json", report)
        if online:
            with phase(out, 6, "WandB曲线回读"):
                report["cloud"] = verify_cloud(local)
            write_json_atomic(out / "verification.json", report)
        else:
            report["cloud"] = {"skipped": True, "reason": "WandB is disabled; all metrics remain local"}
            write_json_atomic(out / "verification.json", report)
            print("[6/6 WandB曲线回读] WandB已禁用，跳过登录、上传与云端核验；本地记录正常保存。", flush=True)
        passed = local["all_passed"] and (not online or report["cloud"]["all_passed"])
        atomic_json(out / "suite_status.json", {"status": "completed" if passed else "failed", "stage": 6,
                    "counts": man.counts(), "verification": str(out / "verification.json"), "updated_utc": utc_now()})
        print(json.dumps({"output_dir": str(out), "counts": man.counts(), "local_verified": local["all_passed"],
                          "cloud_verified": report.get("cloud", {}).get("all_passed")}, indent=2), flush=True)
        return 0 if passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true", help="Write all 150 configs without training or network access")
    action.add_argument("--validate", action="store_true", help="Six tiny PPO preparations + 30 short Flow runs; never formal results")
    action.add_argument("--run", action="store_true", help="Prepare/reuse million-step PPO priors then run 150 x 30k Flow jobs")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0; scheduler visibility is preserved")
    parser.add_argument("--max-parallel", type=int, default=1, help="Independent runs; start with 1 on the A100 20GB vGPU")
    parser.add_argument("--entity", default=None, help="WandB entity, otherwise existing environment/account default")
    parser.add_argument("--output-dir", type=Path)
    logging_mode = parser.add_mutually_exclusive_group()
    logging_mode.add_argument("--local-only", action="store_true", help="Disable W&B for validation or formal training")
    logging_mode.add_argument("--wandb-online", action="store_true", help="Explicitly enable W&B login/upload (default comes from suite config)")
    args = parser.parse_args(argv)
    try:
        if args.max_parallel < 1:
            raise ConfigError("--max-parallel must be positive")
        load_dotenv()
        s = load_suite()
        online = args.wandb_online or (bool(s.get("wandb_online", False)) and not args.local_only)
        out = (args.output_dir or (ROOT / "outputs/experiments" / s["name"])).resolve()
        if args.validate and args.output_dir is None:
            out = ROOT / "outputs/validation" / f"{s['name']}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if args.status:
            if (out / "manifest.json").is_file():
                counts = MatrixManifest.load(out / "manifest.json").counts()
            elif (out / "plan.json").is_file():
                plan = json.loads((out / "plan.json").read_text())
                counts = {"planned": len(plan["runs"]), "started": False,
                          "note": "Prepared only; --run first prepares missing PPO priors"}
            else:
                raise ConfigError("No plan or experiment manifest; run --plan first")
            print(json.dumps(counts, indent=2))
            return 0
        if args.plan:
            paths = default_prior_paths(s)
            plans = build_plans(s, paths, device=args.device, entity=args.entity, online=online)
            write_plan(s, plans, out, paths)
            print(json.dumps({"plan": str(out / "plan.json"), "runs": len(plans), "flow_env_steps": 4500000,
                              "task_prior_pairs": {k: v for k, v in Counter(p.axes["prior"] for p in plans).items()},
                              "missing_ppo_priors": [e for e, p in paths.items() if not (p / "manifest.json").is_file()]}, indent=2))
            return 0
        return execute(s, out, device=args.device, online=online, entity=args.entity or os.getenv("WANDB_ENTITY"),
                       smoke=args.validate, max_parallel=args.max_parallel)
    except KeyboardInterrupt:
        print("Interrupted. Completed results/checkpoints are preserved; rerun the same command to continue.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 -- CLI boundary: redact credentials before reporting SDK/network errors
        print(safe_error(e), file=sys.stderr)
        return 1
