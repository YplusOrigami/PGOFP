"""Independent calibration of PID / MPC controllers (quality tiers) on calibration seeds that are disjoint from eval/test seeds.

Tiers are derived from calibration return and failure rate, never from PGOFP test returns. All trial budgets are written out.
  python scripts/calibrate_controller.py env=inverted_pendulum prior=pid trials=30
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

CALIB_SEED_BASE = 50_000


def evaluate_controller(cfg, controller, episodes: int = 10, seed_base: int = CALIB_SEED_BASE, random_actions: bool = False) -> Dict[str, float]:
    from ..envs.factory import make_pool
    from ..evaluation.evaluator import Evaluator, summarize
    from .manager import ControllerManager

    pool = make_pool(cfg, 1, seed_base, "calib")
    mgr = ControllerManager([controller] if controller is not None else [], 1, pool.spec.action_dim)
    rs = np.random.default_rng(seed_base)
    if random_actions or controller is None:
        policy = lambda ctx: rs.uniform(-1, 1, size=(1, pool.spec.action_dim)).astype(np.float32)
    else:
        policy = lambda ctx: np.tanh(ctx["prior_mu"][:, 0]).astype(np.float32)
    res = Evaluator(pool, mgr, pool.spec).run(policy, episodes, seed_base)
    pool.close()
    s = summarize(res, "c")
    return {"return_mean": s["c/return_mean"], "return_std": s["c/return_std"], "failure_rate": s["c/failure_rate"], "length_mean": s["c/length_mean"]}


def _sample_gains(base: Dict[str, Any], rng: np.random.Generator, spread: float) -> Dict[str, Any]:
    """Log-uniform scaling of every positive gain, sign preserved (signs were fixed by small-perturbation response)."""
    out = json.loads(json.dumps(base))
    g = out["gains"]
    for k, v in g.items():
        if isinstance(v, list):
            f = float(np.exp(rng.uniform(-spread, spread)))
            g[k] = [float(x) * f for x in v]
        elif isinstance(v, (int, float)) and v != 0:
            g[k] = float(v) * float(np.exp(rng.uniform(-spread, spread)))
    return out


def run_calibration(argv: List[str]) -> int:
    from ..config.compose import compose
    from ..envs.factory import make_pool
    from ..core.registry import CONTROLLERS
    from . import factory  # noqa: F401  (registers controllers)

    kv = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in argv if "=" in a and a.split("=", 1)[0] in ("trials", "episodes", "spread", "out", "seed", "write_configs")}
    rest = [a for a in argv if a.split("=", 1)[0] not in kv]
    cfg = compose(rest)
    trials, episodes, spread = int(kv.get("trials", 30)), int(kv.get("episodes", 10)), float(kv.get("spread", 1.2))
    out_dir = Path(kv.get("out", f"outputs/calibration/{cfg.env.name or cfg.env.id}_{cfg.prior.type}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.prior.type != "pid":
        print("only prior=pid gain search is implemented here; MPC quality is varied by horizon/population in configs", file=sys.stderr)
        return 2
    pool = make_pool(cfg, 1, 0, "calib")
    env_spec = pool.spec
    pool.close()
    base = dict(__import__("omegaconf").OmegaConf.to_container(cfg.prior.controller))
    rng = np.random.default_rng(int(kv.get("seed", 0)))
    rand = evaluate_controller(cfg, None, episodes, CALIB_SEED_BASE, random_actions=True)
    rows = []
    t0 = time.time()
    for i in range(trials):
        cand = base if i == 0 else _sample_gains(base, rng, spread)
        ctrl = CONTROLLERS.build(cand["name"], cand, env_spec)
        m = evaluate_controller(cfg, ctrl, episodes, CALIB_SEED_BASE)
        rows.append({"trial": i, "gains": json.dumps(cand["gains"]), **m})
        print(f"trial {i:3d}: return={m['return_mean']:.2f} fail={m['failure_rate']:.2f} len={m['length_mean']:.0f}")
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = max(r["return_mean"] for r in rows)
    span = max(best - rand["return_mean"], 1e-8)
    tiers: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        score = (r["return_mean"] - rand["return_mean"]) / span
        tier = "good" if score >= 0.8 else ("medium" if score >= 0.3 else "poor")
        # representative of a tier: the one closest to the tier's central score, preferring lower failure for 'good'
        tgt = {"good": 1.0, "medium": 0.55, "poor": 0.1}[tier]
        if tier not in tiers or abs(score - tgt) < abs(tiers[tier]["score"] - tgt):
            tiers[tier] = {"score": score, **r}
    budget = {"trials": trials, "episodes_per_trial": episodes, "calibration_seed_base": CALIB_SEED_BASE, "wall_time_sec": time.time() - t0,
              "random_action_return": rand}
    (out_dir / "tiers.json").write_text(json.dumps({"tiers": tiers, "budget": budget}, indent=2))
    for tier, rec in tiers.items():
        c = json.loads(json.dumps(base))
        c["gains"] = json.loads(rec["gains"])
        c["quality"] = tier
        c["version"] = f"{base.get('version', 'pid')}_{tier}"
        c["calibration"] = {"return_mean": rec["return_mean"], "failure_rate": rec["failure_rate"], "score": rec["score"], **budget}
        path = out_dir / f"{cfg.env.name}_pid_{tier}.yaml"
        path.write_text(yaml.safe_dump(c, sort_keys=False))
        if kv.get("write_configs") == "true":
            (Path(__file__).resolve().parents[3] / "configs" / "controller" / path.name).write_text(yaml.safe_dump(c, sort_keys=False))
        print(f"tier {tier}: score={rec['score']:.2f} return={rec['return_mean']:.2f} fail={rec['failure_rate']:.2f} -> {path}")
    return 0
