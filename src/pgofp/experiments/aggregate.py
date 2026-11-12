"""Aggregate run directories into seed-level and combination-level tables (seed = the independent statistical unit)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..evaluation.metrics import bootstrap_ci, max_drawdown, return_auc, steps_to_threshold, tail_mean
from ..logging.jsonl import read_jsonl
from .manifest import MatrixManifest, write_json_atomic


def _eval_curve(run_dir: Path, metric: str = "eval/return_mean") -> Tuple[List[float], List[float]]:
    recs = [r for r in read_jsonl(run_dir / "metrics.jsonl") if metric in r]
    # Periodic and final evaluation may share a step. Preserve raw records while
    # using the last measurement per step for seed-level statistics.
    by_step = {r["step"]: r[metric] for r in recs}
    steps = sorted(by_step)
    return steps, [by_step[step] for step in steps]


def _find_run_dir(man: MatrixManifest, key: str) -> Optional[Path]:
    cands = sorted((man.dir / "runs").glob(f"{key}__*"))
    return cands[-1] if cands else None


def aggregate(man: MatrixManifest, thresholds: Optional[Dict[str, float]] = None, tail_frac: float = 0.2,
              progress=None) -> Dict[str, Any]:
    thresholds = thresholds or {}
    seed_rows: List[Dict[str, Any]] = []
    for index, rec in enumerate(man.runs, 1):
        row = {"run_key": rec["run_key"], "status": rec["status"], "variant": rec["variant"], "seed": rec["seed"], **{f"axis:{k}": v for k, v in rec["axes"].items()}}
        d = _find_run_dir(man, rec["run_key"])
        if d is not None and rec["status"] in ("completed", "failed", "interrupted"):
            steps, rets = _eval_curve(d)
            lens = _eval_curve(d, "eval/length_mean")[1]
            fail = _eval_curve(d, "eval/failure_rate")[1]
            env = next((o.split("=", 1)[1] for o in rec["overrides"] if o.startswith("env=")), "")
            thr = thresholds.get(env)
            row.update({
                "n_evals": len(rets), "final_tail_return": tail_mean(rets, tail_frac) if rets else float("nan"),
                "auc_return": return_auc(steps, rets) if rets else float("nan"), "max_drawdown": max_drawdown(rets) if rets else float("nan"),
                "tail_return_std": float(np.std(rets[-max(2, int(len(rets) * tail_frac)):])) if len(rets) > 1 else float("nan"),
                "final_failure_rate": tail_mean(fail, tail_frac) if fail else float("nan"),
                "final_length": tail_mean(lens, tail_frac) if lens else float("nan"),
                "steps_to_threshold": ("" if thr is None else (steps_to_threshold(steps, rets, thr) if steps_to_threshold(steps, rets, thr) is not None else "censored")),
                "run_dir": str(d),
            })
            baseline = d / "evaluation/prior_baseline.json"
            if baseline.is_file():
                prior = json.loads(baseline.read_text())["prior_eval"]["summary"]
                row["prior_return_mean"] = prior["prior_eval/return_mean"]
                row["prior_return_std"] = prior["prior_eval/return_std"]
                row["return_gain_over_prior"] = row["final_tail_return"] - row["prior_return_mean"]
        seed_rows.append(row)
        if progress:
            progress(index, len(man.runs))

    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in seed_rows:
        gk = (r["variant"], tuple(sorted((k, str(v)) for k, v in r.items() if k.startswith("axis:"))))
        groups[gk].append(r)
    combo_rows = []
    for (variant, axes), rows in groups.items():
        done = [r for r in rows if r["status"] == "completed" and np.isfinite(r.get("final_tail_return", np.nan))]
        c = {"variant": variant, **dict(axes), "n_seeds_planned": len(rows), "n_completed": len(done),
             "n_failed": sum(r["status"] == "failed" for r in rows), "n_skipped": sum(r["status"] == "skipped" for r in rows)}
        for m in ("final_tail_return", "auc_return", "max_drawdown", "tail_return_std", "final_failure_rate",
                  "prior_return_mean", "return_gain_over_prior"):
            values = [r[m] for r in done if np.isfinite(r.get(m, np.nan))]
            mean, lo, hi = bootstrap_ci(values) if values else (float("nan"),) * 3
            c[f"{m}_mean"], c[f"{m}_ci_lo"], c[f"{m}_ci_hi"] = mean, lo, hi
        c["note"] = "CI over training seeds; unstable for < 5 seeds" if len(done) < 5 else ""
        combo_rows.append(c)

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        keys: List[str] = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(man.dir / "summary_seed_level.csv", seed_rows)
    write_csv(man.dir / "summary.csv", combo_rows)
    man.write_failures()
    write_json_atomic(man.dir / "aggregate_counts.json", man.counts())
    return {"seed_rows": seed_rows, "combo_rows": combo_rows}
