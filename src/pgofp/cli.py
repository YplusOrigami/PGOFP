"""Command-line implementations (scripts/*.py are thin wrappers). All commands accept Hydra-style `key=value` overrides."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def _split_flags(argv: List[str]):
    flags = [a for a in argv if a.startswith("--")]
    rest = [a for a in argv if not a.startswith("--")]
    return flags, rest


def _parse_kv(argv: List[str]) -> Dict[str, str]:
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.lstrip("+")] = v
    return out


# ------------------------------------------------------------------------------------------------ train
def train(argv: List[str]) -> int:
    from .config.compose import compose
    from .core.exceptions import PGOFPError

    try:
        cfg = compose(argv)
    except PGOFPError as e:
        print(f"[config error] {e}", file=sys.stderr)
        return EXIT_USAGE
    algo = cfg.algorithm.name
    try:
        if algo == "ppo":
            return _train_ppo_baseline(cfg)
        from .training.runner import Runner

        resume = cfg.resume
        if resume:
            rp = Path(resume)
            ckpt = rp if rp.suffix == ".pt" else rp / "checkpoints" / "last.pt"
            run_dir = None if cfg.resume_mode == "weights_only" else ckpt.parent.parent
            runner = Runner(cfg, run_dir=run_dir).build()
            from .training.checkpoint import CheckpointManager

            state = CheckpointManager(ckpt.parent.parent).load(ckpt)
            runner.load_state(state, cfg.resume_mode)
        else:
            runner = Runner(cfg).build()
        summary = runner.train()
        print(json.dumps({k: v for k, v in summary.items() if k != "final_eval"}, indent=2, default=str))
        print(f"run dir: {runner.run_dir}")
        return EXIT_OK if summary["outcome"] in ("completed", "interrupted") else EXIT_FAIL
    except PGOFPError as e:
        print(f"[{type(e).__name__}] {e}", file=sys.stderr)
        return EXIT_FAIL


def _train_ppo_baseline(cfg) -> int:
    """algorithm=ppo through the same Logger / env-step axis (comparison baseline; no artifact export)."""
    import uuid

    from omegaconf import OmegaConf

    from .config.compose import config_hash, save_resolved, to_container
    from .logging.logger import Logger
    from .training.prior_trainer import PPOTrainer
    from .utils.torch_utils import select_device

    name = cfg.run_name or f"{cfg.env.id}__ppo__s{cfg.seed}__{config_hash(cfg)[:8]}"
    run_dir = Path(cfg.output_dir) / f"{name}__{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved(cfg, run_dir / "config_resolved.yaml")
    logger = Logger(run_dir, cfg, to_container(cfg), name)
    tr = PPOTrainer(cfg, cfg.seed, select_device(cfg.device), n_envs=int(cfg.train.num_envs))
    total = int(dict(OmegaConf.to_container(cfg.algorithm.params)).get("total_steps", cfg.train.total_env_steps))
    total = min(total, cfg.train.total_env_steps) if cfg.smoke else cfg.train.total_env_steps
    code = 0
    try:
        tr.run(total, log_fn=logger.write_scalars, eval_interval=cfg.eval.interval, eval_episodes=cfg.eval.episodes, eval_seed_base=cfg.eval.seed_base)
    except Exception as e:
        logger.event("exception", error=f"{type(e).__name__}: {e}")
        code = EXIT_FAIL
        raise
    finally:
        tr.close()
        logger.close(code)
    (run_dir / "manifest.json").write_text(json.dumps({"outcome": "completed" if code == 0 else "failed", "env_steps": tr.env_steps}))
    print(f"run dir: {run_dir}")
    return code


# ------------------------------------------------------------------------------------------------ prior training
def train_prior(argv: List[str]) -> int:
    from omegaconf import OmegaConf

    from .config.compose import compose
    from .logging.logger import Logger
    from .training.prior_trainer import PPOTrainer, assign_tiers
    from .utils.torch_utils import select_device

    from .config.compose import validate_schema

    raw = compose(argv, config_name="prior_train", validate=False)
    extra = {k: OmegaConf.to_container(raw[k], resolve=True) if OmegaConf.is_config(raw[k]) else raw[k]
             for k in ("artifact_dir", "checkpoint_fractions", "calibration") if k in raw}
    body = OmegaConf.create({k: v for k, v in OmegaConf.to_container(raw, resolve=True).items() if k not in extra})
    cfg = validate_schema(body)
    seed = int(cfg.seed)
    total = int(cfg.algorithm.params.get("total_steps", 1_000_000))
    fracs = list(extra.get("checkpoint_fractions", [0.1, 0.25, 0.5, 1.0]))
    prior_id = f"{cfg.env.name or cfg.env.id}_ppo_seed{seed}"
    art_root = Path(extra.get("artifact_dir", "artifacts/priors")) / prior_id
    art_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg.output_dir) / prior_id
    run_dir.mkdir(parents=True, exist_ok=True)
    from .config.compose import save_resolved

    save_resolved(raw, run_dir / "config_resolved.yaml")
    logger = Logger(run_dir, cfg, OmegaConf.to_container(cfg, resolve=True), prior_id)
    tr = PPOTrainer(cfg, seed, select_device(cfg.device), n_envs=int(OmegaConf.select(cfg, "train.num_envs", default=1)))
    calib = extra.get("calibration", {"seeds_base": 50000, "episodes": 10})
    calib_base, calib_eps = int(calib["seeds_base"]), int(calib["episodes"])
    stage_calib: Dict[str, Dict[str, float]] = {}
    t_start = time.time()

    def on_stage(step: int, trainer: PPOTrainer) -> None:
        stochastic = trainer.evaluate(calib_eps, calib_base, deterministic=False, progress_fn=logger.progress)
        deterministic = trainer.evaluate(calib_eps, calib_base, deterministic=True, progress_fn=logger.progress)
        name = f"stage_{step}"
        calib = {"stochastic": stochastic, "deterministic": deterministic, "random_action_return": random_ret, "calibration_seeds": [calib_base, calib_base + calib_eps - 1]}
        budget = {"prior_pretrain_env_steps": trainer.env_steps, "prior_seed": seed, "wall_time_sec": time.time() - t_start}
        logger.progress("ppo_export", 0, 1, unit="artifacts", message=name)
        trainer.export(art_root / name, calib, budget,
                       extra={"smoke": bool(cfg.smoke), "requested_prior_steps": total})
        stage_calib[name] = stochastic
        logger.event("prior_stage_exported", name=name, step=step, calibration=stochastic)
        logger.progress("ppo_export", 1, 1, unit="artifacts", message=name)

    stage_steps = sorted({max(1, int(f * total)) for f in fracs})
    code = EXIT_FAIL
    try:
        logger.progress("ppo_calibration", 0, calib_eps, unit="episodes", message="random-action reference")
        random_ret = tr.evaluate(calib_eps, calib_base, deterministic=False, random_actions=True,
                                 progress_fn=logger.progress)["return_mean"]
        tr.run(total, log_fn=logger.write_scalars, eval_interval=max(total // 50, 2048), eval_episodes=5, stage_steps=stage_steps,
               on_stage=on_stage, progress_fn=logger.progress)
        tiers = assign_tiers(stage_calib, random_ret)
        (art_root / "tiers.json").write_text(json.dumps({"tiers": tiers, "calibration": stage_calib, "random_action_return": random_ret,
                                                          "note": "tiers are derived from independent calibration seeds only"}, indent=2))
        logger.wandb.summary({"prior_pretrain_env_steps": tr.env_steps, "prior_seed": seed, "outcome": "completed"})
        code = EXIT_OK
        logger.progress("completed", tr.env_steps, tr.env_steps, status="completed", force=True)
    except Exception as e:
        logger.progress_file.fail(f"{type(e).__name__}: {e}")
        raise
    finally:
        tr.close()
        logger.close(code)
    print(json.dumps({"artifact_root": str(art_root), "tiers": tiers}, indent=2))
    return EXIT_OK


# ------------------------------------------------------------------------------------------------ evaluate
def evaluate(argv: List[str]) -> int:
    from omegaconf import OmegaConf

    from .config.compose import default_config_dir
    from .training.runner import Runner

    kv = _parse_kv(argv)
    ev = OmegaConf.load(default_config_dir() / "evaluate.yaml")
    ev = OmegaConf.merge(ev, OmegaConf.from_dotlist([a for a in argv if "=" in a]))
    if not ev.checkpoint:
        print("usage: evaluate.py checkpoint=/abs/path/to/checkpoint.pt [episodes=20] [prior.artifact_path=...]", file=sys.stderr)
        return EXIT_USAGE
    reserved = {"checkpoint", "eval_seeds_base", "episodes", "K_eval", "representative_action", "condition_bank_size", "condition_samples", "video"}
    overrides = [a for a in argv if "=" in a and a.split("=", 1)[0] not in reserved]
    out_dir = Path(kv.get("output_dir", "outputs/evals")) / f"{Path(ev.checkpoint).parent.parent.name}__{int(time.time())}"
    runner = Runner.from_checkpoint(ev.checkpoint, overrides, run_dir=out_dir)
    results, spec = runner.evaluate_frozen(int(ev.episodes), int(ev.eval_seeds_base), K=int(ev.K_eval))
    from .evaluation.evaluator import summarize

    summary = summarize(results, "eval")
    rows = [{"seed": r.seed, "return": r.ret, "length": r.length, "terminated": r.terminated, "truncated": r.truncated, **r.extras} for r in results]
    (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    import csv

    with open(out_dir / "evaluation" / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    payload = {"checkpoint": str(ev.checkpoint), "env_transitions": runner.env_transitions, "frozen_source_spec": spec.__dict__,
               "K_eval": int(ev.K_eval), "summary": summary}
    if ev.representative_action:
        rep, _ = runner.evaluate_frozen(min(int(ev.episodes), 10), int(ev.eval_seeds_base), deterministic=True)
        payload["representative_action_summary"] = summarize(rep, "eval_representative")
    (out_dir / "evaluation" / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload["summary"], indent=2))
    print(f"episode table: {out_dir / 'evaluation' / 'episodes.csv'}")
    return EXIT_OK


# ------------------------------------------------------------------------------------------------ experiments
def launch(argv: List[str]) -> int:
    from .experiments.aggregate import aggregate
    from .experiments.launcher import Launcher, build_manifest
    from .experiments.manifest import MatrixManifest

    flags, rest = _split_flags(argv)
    kv = _parse_kv(rest)
    if "experiment" not in kv:
        print("usage: launch_experiments.py experiment=<matrix name> (--dry-run | --execute) [--set k=v ...] [matrix_dir=...]", file=sys.stderr)
        return EXIT_USAGE
    if ("--dry-run" in flags) == ("--execute" in flags):
        print("specify exactly one of --dry-run / --execute", file=sys.stderr)
        return EXIT_USAGE
    variables = {k[4:]: v for k, v in kv.items() if k.startswith("var.")}
    extra = [f"{k}={v}" for k, v in kv.items() if k not in ("experiment", "matrix_dir", "max_parallel", "gpus") and not k.startswith("var.")]
    matrix_dir = Path(kv.get("matrix_dir", f"outputs/experiments/{kv['experiment']}"))
    manifest_path = matrix_dir / "manifest.json"
    if manifest_path.is_file() and "--execute" in flags:
        man = MatrixManifest.load(manifest_path)  # continue an existing matrix
    else:
        man = build_manifest(kv["experiment"], matrix_dir, extra_overrides=extra, variables=variables)
        man.save()
    counts = man.counts()
    print(json.dumps({"matrix_dir": str(matrix_dir), "counts": counts}, indent=2))
    for r in man.runs:
        if r["status"] == "skipped":
            print(f"  skipped {r['run_key']}: {r['reason'][:140]}")
    if "--dry-run" in flags:
        print(f"dry-run only: manifest written to {manifest_path} (no process started)")
        return EXIT_OK
    root = Path(__file__).resolve().parents[2]
    gpus = [int(x) for x in kv["gpus"].split(",")] if "gpus" in kv else None
    Launcher(man, root, max_parallel=int(kv.get("max_parallel", 1)), gpus=gpus).run()
    aggregate(man)
    print(json.dumps(man.counts(), indent=2))
    return EXIT_OK


def analyze(argv: List[str]) -> int:
    from .experiments.aggregate import aggregate
    from .experiments.manifest import MatrixManifest

    kv = _parse_kv(argv)
    if "matrix" not in kv:
        print("usage: analyze_results.py matrix=/abs/path/to/manifest.json [thresholds=env:val,env:val]", file=sys.stderr)
        return EXIT_USAGE
    man = MatrixManifest.load(kv["matrix"])
    thr = {}
    for part in kv.get("thresholds", "").split(","):
        if ":" in part:
            k, v = part.split(":")
            thr[k] = float(v)
    res = aggregate(man, thr)
    try:
        _plot_curves(man)
    except Exception as e:  # plotting is optional
        print(f"[plot skipped] {e}")
    print(f"wrote {man.dir/'summary.csv'} ({len(res['combo_rows'])} combinations, {len(res['seed_rows'])} runs)")
    return EXIT_OK


def _plot_curves(man) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .experiments.aggregate import _eval_curve, _find_run_dir

    groups: Dict[str, list] = {}
    for r in man.runs:
        if r["status"] != "completed":
            continue
        d = _find_run_dir(man, r["run_key"])
        if d is None:
            continue
        env = next((o.split("=", 1)[1] for o in r["overrides"] if o.startswith("env=")), "env")
        key = f"{env}|{r['variant']}|" + "|".join(f"{k}={v}" for k, v in r["axes"].items() if k != "env")
        groups.setdefault(key, []).append(_eval_curve(d))
    envs = sorted({k.split("|")[0] for k in groups})
    (man.dir / "figures").mkdir(exist_ok=True)
    for env in envs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, curves in sorted(groups.items()):
            if not k.startswith(env + "|"):
                continue
            xs = curves[0][0]
            ys = np.array([np.interp(xs, c[0], c[1]) for c in curves])
            ax.plot(xs, ys.mean(0), label=k.split("|", 1)[1] or "run")
            if len(curves) > 1:
                ax.fill_between(xs, ys.min(0), ys.max(0), alpha=0.15)
        ax.set_xlabel("env steps"); ax.set_ylabel("eval return (raw)"); ax.set_title(f"{env} (band = min/max over seeds)")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(man.dir / "figures" / f"eval_return__{env}.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------------------ controller calibration
def calibrate(argv: List[str]) -> int:
    from .controllers.calibration import run_calibration

    return run_calibration(argv)


# ------------------------------------------------------------------------------------------------ misc tools
def benchmark_latency(argv: List[str]) -> int:
    import torch

    from .evaluation.latency import benchmark_policy
    from .training.runner import Runner

    kv = _parse_kv(argv)
    if "checkpoint" not in kv:
        print("usage: benchmark_latency.py checkpoint=/abs/checkpoint.pt [device=cpu] [iters=2000]", file=sys.stderr)
        return EXIT_USAGE
    runner = Runner.from_checkpoint(kv["checkpoint"], [f"device={kv.get('device', 'cpu')}"] + [a for a in argv if "=" in a and not a.startswith(("checkpoint=", "device=", "iters="))],
                                    run_dir=Path(kv.get("output_dir", "outputs/latency")) / f"{int(time.time())}")
    runner.collector.reset()
    ctx = {k: v[:1] for k, v in runner.collector.cur.items()}
    spec = runner.source_sched.spec_at(None) if runner.source_sched._last is not None else runner.source_sched.spec_at(runner.env_transitions)
    obs, ref = ctx["obs_raw"], ctx["reference"]
    q = (lambda: runner.eval_manager.predict(obs, ref)) if runner.eval_manager.n_prior else None
    res = benchmark_policy(runner.agent, ctx, spec, runner.norm.snapshot(runner.device), runner.device, q, int(kv.get("iters", 2000)))
    out = Path(runner.run_dir) / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if k != "meta":
            print(f"{k:28s} p50={v['p50_ms']:.3f} p95={v['p95_ms']:.3f} p99={v['p99_ms']:.3f} max={v['max_ms']:.3f} ms  miss@10ms={v['deadline_miss_rate']:.4f}")
    print(f"meta: {res['meta']}  -> {out}")
    return EXIT_OK


def inspect_policy(argv: List[str]) -> int:
    from .evaluation.distributions import (condition_bank_from_replay, sample_policy_distribution, save_condition_bank,
                                           summarize_distribution)
    from .training.runner import Runner

    kv = _parse_kv(argv)
    if "checkpoint" not in kv:
        print("usage: inspect_policy.py checkpoint=/abs/checkpoint.pt [n_conditions=64] [n_samples=2048]", file=sys.stderr)
        return EXIT_USAGE
    out_dir = Path(kv.get("output_dir", "outputs/inspect")) / f"{int(time.time())}"
    runner = Runner.from_checkpoint(kv["checkpoint"], [a for a in argv if "=" in a and not a.startswith(("checkpoint=", "n_conditions=", "n_samples=", "output_dir="))], run_dir=out_dir)
    ck = Path(kv["checkpoint"]).parent / "replay" / "replay.npz"
    if ck.is_file():
        runner.replay.load(ck)
    if len(runner.replay) == 0:
        runner.collector.reset()
        for _ in range(500):
            runner.collector.step(runner.agent, runner.source_sched.spec_at(None), runner.norm.snapshot(runner.device), runner.env_transitions, False)
    n_c, n_s = int(kv.get("n_conditions", 64)), int(kv.get("n_samples", 2048))
    bank = condition_bank_from_replay(runner.replay, n_c, np.random.default_rng(0))
    spec = runner.source_sched.spec_at(None) if runner.source_sched._last is not None else runner.source_sched.spec_at(runner.env_transitions)
    samples = sample_policy_distribution(runner.agent, bank, spec, runner.norm.snapshot(runner.device), n_s, runner.rng.torch("eval_policy"), runner.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_condition_bank(out_dir / "condition_bank.npz", bank, {"checkpoint": kv["checkpoint"]})
    np.savez(out_dir / "action_samples.npz", **samples)
    stats = summarize_distribution(samples)
    (out_dir / "distribution_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved: {out_dir}")
    return EXIT_OK


def export_policy(argv: List[str]) -> int:
    from .training.bundle import export_bundle
    from .training.runner import Runner

    kv = _parse_kv(argv)
    if "checkpoint" not in kv:
        print("usage: export_policy.py checkpoint=/abs/checkpoint.pt [out=/abs/policy_bundle]", file=sys.stderr)
        return EXIT_USAGE
    out = Path(kv.get("out", Path(kv["checkpoint"]).parent.parent / "exports" / "policy_bundle"))
    runner = Runner.from_checkpoint(kv["checkpoint"], [a for a in argv if "=" in a and not a.startswith(("checkpoint=", "out="))],
                                    run_dir=Path(kv.get("output_dir", "outputs/exports_tmp")) / f"{int(time.time())}")
    export_bundle(runner, out)
    print(f"policy bundle written to {out}")
    return EXIT_OK


def validate_install(argv: List[str]) -> int:
    import platform

    ok = True

    def check(name: str, fn) -> None:
        nonlocal ok
        try:
            msg = fn()
            print(f"[ OK ] {name}: {msg}")
        except Exception as e:
            ok = False
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    import torch

    check("python/platform", lambda: f"{sys.version.split()[0]} on {platform.platform()}")
    check("torch", lambda: f"{torch.__version__}, cuda={torch.cuda.is_available()}")

    def jvp():
        from torch.func import jvp as f

        _, t = f(lambda x, t: torch.sin(x) * t, (torch.ones(3), torch.tensor(2.0)), (torch.ones(3), torch.tensor(1.0)))
        assert torch.allclose(t, torch.sin(torch.ones(3)) + 2 * torch.cos(torch.ones(3)))
        return "torch.func.jvp matches analytic derivative (cpu)"

    check("jvp", jvp)

    def mj():
        import gymnasium as gym
        import mujoco

        e = gym.make("InvertedPendulum-v5")
        o, _ = e.reset(seed=0)
        e.step(e.action_space.sample())
        return f"gymnasium {gym.__version__}, mujoco {mujoco.__version__}, reset/step ok"

    check("mujoco", mj)
    check("tensorboard writer", lambda: __import__("tensorboard").__version__)
    check("omegaconf", lambda: __import__("omegaconf").__version__)

    def wb():
        try:
            import wandb

            return f"wandb {wandb.__version__} (network NOT required: use logging=wandb_offline)"
        except ImportError:
            return "wandb not installed (optional; local logging works)"

    check("wandb (optional)", wb)
    print("\nOverall:", "OK" if ok else "FAILED")
    return EXIT_OK if ok else EXIT_FAIL


COMMANDS = {"train": train, "train_prior": train_prior, "evaluate": evaluate, "launch": launch, "analyze": analyze,
            "calibrate": calibrate, "benchmark_latency": benchmark_latency, "inspect": inspect_policy, "export": export_policy,
            "validate_install": validate_install}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: pgofp <" + "|".join(COMMANDS) + "> [key=value ...]", file=sys.stderr)
        return EXIT_USAGE
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
