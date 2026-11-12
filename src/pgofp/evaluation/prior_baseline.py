"""Evaluate the actual frozen control prior before each Flow run, independently of beta."""
from __future__ import annotations

import csv
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import torch
from omegaconf import OmegaConf

from ..controllers.factory import build_controllers
from ..controllers.manager import ControllerManager
from ..core.exceptions import ConfigError, PriorUnavailableError
from ..core.types import MetricRecord
from ..envs.factory import make_pool
from ..logging.progress import atomic_json
from ..utils.seed import derive_seed
from .evaluator import Evaluator, summarize


@contextmanager
def preserve_global_rng():
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def evaluate_prior(cfg, run_dir, logger, step: int) -> dict:
    if cfg.prior.type not in ("ppo", "pid", "cem_mpc"):
        raise ConfigError("Prior baseline requires a single PPO/PID/MPC control prior")
    ecfg = cfg.eval
    seed = int(derive_seed(ecfg.seed_base, "prior_baseline") % (2**32))
    report = {"env_id": cfg.env.id, "prior_type": cfg.prior.type, "beta": cfg.source.beta,
              "flow_seed": cfg.seed, "checkpoint_step": step, "eval_seed_base": ecfg.seed_base,
              "policy_rng_seed": seed, "episodes": ecfg.episodes, "max_steps": ecfg.max_steps,
              "smoke": bool(cfg.smoke), "prior_config": OmegaConf.to_container(cfg.prior, resolve=True),
              "primary_policy": "native_stochastic" if cfg.prior.type == "ppo" else "deterministic_controller"}
    output = run_dir / "evaluation" / "prior_baseline" / f"attempt_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    all_scalars, rows = {}, []
    t0 = time.monotonic()
    with preserve_global_rng():
        modes = [False, True] if cfg.prior.type == "ppo" else [True]
        for mode_index, deterministic in enumerate(modes):
            prefix = "prior_eval" if mode_index == 0 else "prior_eval_deterministic"
            stage = "prior_evaluation" if mode_index == 0 else "prior_evaluation_deterministic"
            logger.progress(stage, 0, ecfg.episodes, unit="episodes", message=report["primary_policy"])
            pool = make_pool(cfg, 1, cfg.seed, "prior_baseline")
            try:
                controllers, _ = build_controllers(cfg, pool.spec, cfg.seed)
                manager = ControllerManager(controllers, 1, pool.spec.action_dim, cfg.algorithm.atanh_eps)
                report["controller_versions"] = manager.versions
                rng = np.random.default_rng(seed)

                def policy(ctx, deterministic=deterministic, rng=rng):
                    if not np.all(ctx["prior_valid"][:, 0] > .5):
                        raise PriorUnavailableError("Control prior was invalid during baseline evaluation")
                    z = ctx["prior_mu"][:, 0].copy()
                    if not deterministic:
                        z += np.exp(ctx["prior_logstd"][:, 0]) * rng.standard_normal(z.shape)
                    return np.tanh(z).astype(np.float32)

                def progress(done, total, episode_step, cap, stage=stage):
                    logger.progress(stage, done, total, unit="episodes",
                                    message=f"episode_step={episode_step}/{cap}")

                results = Evaluator(pool, manager, pool.spec).run(policy, ecfg.episodes, ecfg.seed_base,
                                                                 ecfg.max_steps, progress=progress)
                scalars = summarize(results, prefix)
                all_scalars.update(scalars)
                for r in results:
                    rows.append({"mode": prefix, **asdict(r)})
                report[prefix] = {"summary": scalars, "episodes": [asdict(r) for r in results]}
            finally:
                pool.close()
    report["wall_time_sec"] = time.monotonic() - t0
    report["attempt_dir"] = str(output)
    with (output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "seed", "ret", "length", "terminated", "truncated", "extras"])
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / "summary.json", report)
    atomic_json(run_dir / "evaluation" / "prior_baseline.json", report)
    logger.write_scalars(MetricRecord(step, all_scalars))
    logger.wandb.summary(all_scalars)
    logger.event("prior_evaluated_before_flow", summary=all_scalars, path=str(output), training_env_steps=step)
    logger.progress("prior_evaluation_result", ecfg.episodes, ecfg.episodes, unit="episodes",
                    message=f"mean_return={all_scalars['prior_eval/return_mean']:.4f}, "
                            f"std={all_scalars['prior_eval/return_std']:.4f}; {report['primary_policy']}")
    return report
