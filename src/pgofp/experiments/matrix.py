"""Expand an experiment matrix into concrete run plans (task x variant x axes x seed), validating every combination.

Unsupported combinations are written as `skipped` with the reason (they are never silently dropped).
Matrix file format (configs/experiment/<name>.yaml):
  matrix:
    name: ...
    seeds: [0,1,2]
    base_overrides: [k=v, ...]            # applied to every run
    axes:     {env: [a,b], source.beta: [0, 0.5]}   # Cartesian product (group keys select config groups)
    variants: [{name: x, overrides: [...]}, ...]    # crossed with axes
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config.compose import compose, config_hash, default_config_dir
from ..config.validate import validate_config
from ..core.exceptions import (BackendUnavailableError, ConfigError, PGOFPError, PriorUnavailableError,
                               UnsupportedCombinationError)


@dataclass
class RunPlan:
    run_key: str
    overrides: List[str]
    seed: int
    variant: str
    axes: Dict[str, Any]
    config_hash: str = ""
    status: str = "planned"
    reason: str = ""

    def to_record(self) -> Dict[str, Any]:
        return {"run_key": self.run_key, "overrides": self.overrides, "seed": self.seed, "variant": self.variant, "axes": self.axes,
                "config_hash": self.config_hash, "status": self.status, "reason": self.reason, "run_dir": None, "exit_code": None,
                "started_utc": None, "finished_utc": None, "retries": 0}


def load_matrix(name_or_path: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(name_or_path)
    if not p.is_file():
        p = (config_dir or default_config_dir()) / "experiment" / f"{name_or_path}.yaml"
    d = yaml.safe_load(p.read_text())
    if "matrix" not in d:
        raise ConfigError(f"{p} is not a matrix experiment (no top-level `matrix:` key); smoke-style experiments are used as experiment=<name>")
    return d["matrix"]


def _wandb_routing(cfg, overrides: List[str], matrix: str, variant: str) -> List[str]:
    """One WandB project per environment (returns of different tasks must not share a chart), variant -> group (seed-aggregated curves).
    Smoke runs go to a separate project. Explicit user overrides win. Routing never enters the config hash (see config_hash)."""
    have = {o.split("=", 1)[0] for o in overrides}
    out: List[str] = []
    if "logging.wandb.project" not in have:
        out.append("logging.wandb.project=" + ("pgofp-smoke" if cfg.smoke else f"pgofp-{cfg.env.name or _slug(cfg.env.id).lower()}"))
    if "logging.wandb.group" not in have:
        out.append(f"logging.wandb.group={_slug(variant)}")
    return out


def _subst(s: str, variables: Dict[str, str]) -> str:
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: variables.get(m.group(1), m.group(0)), s)


def _slug(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


def expand_matrix(spec: Dict[str, Any], *, extra_overrides: Optional[List[str]] = None, variables: Optional[Dict[str, str]] = None,
                  seeds: Optional[List[int]] = None, config_dir: Optional[Path] = None) -> List[RunPlan]:
    variables = variables or {}
    base = [_subst(o, variables) for o in spec.get("base_overrides", [])]
    axes: Dict[str, List[Any]] = spec.get("axes", {}) or {}
    variants = spec.get("variants") or [{"name": "default", "overrides": []}]
    seeds = seeds if seeds is not None else spec.get("seeds", [0])
    keys = list(axes)
    plans: List[RunPlan] = []
    for var in variants:
        for combo in itertools.product(*[axes[k] for k in keys]) if keys else [()]:
            axis_over = [f"{k}={_subst(str(v), variables)}" for k, v in zip(keys, combo)]
            for seed in seeds:
                ov = base + [_subst(o, variables) for o in var.get("overrides", [])] + axis_over + list(extra_overrides or []) + [f"seed={seed}"]
                key = "__".join([spec["name"], _slug(var["name"]), *[f"{_slug(k)}-{_slug(v)}" for k, v in zip(keys, combo)], f"s{seed}"])
                plan = RunPlan(key, ov, seed, var["name"], dict(zip(keys, combo)))
                try:
                    cfg = compose(ov, config_dir=config_dir)
                    validate_config(cfg)
                    plan.config_hash = config_hash(cfg)
                    plan.overrides = ov + _wandb_routing(cfg, ov, spec["name"], var["name"])
                except (UnsupportedCombinationError, BackendUnavailableError, PriorUnavailableError) as e:
                    plan.status, plan.reason = "skipped", f"{type(e).__name__}: {e}"
                except (ConfigError, PGOFPError, Exception) as e:
                    plan.status, plan.reason = "skipped", f"invalid: {type(e).__name__}: {e}"
                plans.append(plan)
    return plans
