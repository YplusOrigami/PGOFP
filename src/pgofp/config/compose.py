"""Hydra-style composition without Hydra: defaults list -> groups -> experiment -> CLI overrides -> forced values.

  python scripts/train.py env=reacher algorithm=prior_flame_m prior=pid train.total_env_steps=100000

Group selection tokens (`env=reacher`) choose configs/<group>/<name>.yaml; dotted tokens override leaf values.
The composed config is validated against the structured schema (unknown keys raise) and saved fully resolved.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from ..core.exceptions import ConfigError
from .schema import Config

GROUPS = ("algorithm", "env", "prior", "source", "logging", "experiment")


def default_config_dir() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parents[3] / "configs", Path.cwd() / "configs"):
        if p.is_dir():
            return p
    raise ConfigError("cannot locate configs/ directory; pass config_dir explicitly")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    d = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    return d or {}


def _load_group(cfg_dir: Path, group: str, name: str, _seen=()) -> Dict[str, Any]:
    if name in _seen:
        raise ConfigError(f"cyclic inherits in {group}/{name}")
    d = _load_yaml(cfg_dir / group / f"{name}.yaml")
    parent = d.get("inherits")
    if parent:
        base = _load_group(cfg_dir, group, parent, _seen + (name,))
        d = _deep_merge(base, {k: v for k, v in d.items() if k != "inherits"})
    d.pop("inherits", None)
    return d


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _get_dotted(d: Dict[str, Any], path: str, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def compose(overrides: Optional[List[str]] = None, config_name: str = "train", config_dir: Optional[str | Path] = None,
            validate: bool = True) -> DictConfig:
    cfg_dir = Path(config_dir) if config_dir else default_config_dir()
    overrides = list(overrides or [])
    base = _load_yaml(cfg_dir / f"{config_name}.yaml")
    defaults = base.pop("defaults", [])
    selected: Dict[str, Optional[str]] = {}
    for item in defaults:
        if isinstance(item, dict):
            (g, n), = item.items()
            selected[g] = n
    dotlist: List[str] = []
    for tok in overrides:
        if "=" not in tok:
            raise ConfigError(f"override must be key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        k = k.lstrip("+")
        if k in GROUPS:
            if k not in selected and k != "experiment":
                raise ConfigError(f"unknown config group {k!r}")
            selected[k] = None if v in ("null", "None", "") else v
        else:
            dotlist.append(f"{k}={v}")

    merged: Dict[str, Any] = {}
    forces: Dict[str, Any] = {}
    for g in GROUPS:
        name = selected.get(g)
        if not name:
            continue
        gd = _load_group(cfg_dir, g, name)
        if g == "experiment":
            merged = _deep_merge(merged, gd)
        else:
            f = gd.get("forces") if g == "algorithm" else None
            if f:
                forces = dict(f)
            merged = _deep_merge(merged, {g: gd})
    merged = _deep_merge(base, merged)  # train.yaml body < groups < experiment (GROUPS order puts experiment last)

    cfg = OmegaConf.create(merged)
    if dotlist:
        try:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
        except Exception as e:  # pragma: no cover - omegaconf raises many types
            raise ConfigError(f"bad override: {e}") from e

    # forced values (e.g. gaussian_flame_m forces source.kind=gaussian); a conflicting explicit override is an error
    if forces:
        for path, value in forces.items():
            explicit = [t for t in dotlist if t.startswith(path + "=")]
            if explicit and str(OmegaConf.select(cfg, path)) != str(value):
                raise ConfigError(f"{cfg.algorithm.name} forces {path}={value}; override {explicit[0]!r} is not allowed")
            OmegaConf.update(cfg, path, value, merge=True)

    cfg = _resolve_controller_refs(cfg, cfg_dir)
    if validate:
        cfg = validate_schema(cfg)
    return cfg


def validate_schema(cfg: DictConfig) -> DictConfig:
    schema = OmegaConf.structured(Config)
    try:
        out = OmegaConf.merge(schema, cfg)
    except (ConfigKeyError, ValidationError) as e:
        raise ConfigError(f"invalid configuration: {e}") from e
    OmegaConf.set_struct(out, True)
    return out


def _resolve_controller_refs(cfg: DictConfig, cfg_dir: Path) -> DictConfig:
    """Inline configs/controller/<name>.yaml into prior.controller so the resolved config is self-contained."""
    prior = cfg.get("prior")
    if prior is None:
        return cfg
    name = prior.get("controller_config")
    if name == "auto":
        envname = cfg.get("env", {}).get("name", "")
        name = f"{envname}_pid" if prior.get("type") == "pid" else "cem_mpc"
        if not (cfg_dir / "controller" / f"{name}.yaml").is_file():
            # no controller config for this env: leave unresolved so validation reports the unsupported task/prior pair
            OmegaConf.update(cfg, "prior.controller_config", None, merge=False)
            return cfg
        OmegaConf.update(cfg, "prior.controller_config", name, merge=False)
    if name:
        cur = prior.get("controller")
        inline = OmegaConf.to_container(cur, resolve=False) if OmegaConf.is_config(cur) else (cur or {})
        ref = _load_yaml(cfg_dir / "controller" / f"{name}.yaml")
        OmegaConf.update(cfg, "prior.controller", _deep_merge(ref, inline), merge=False)
    comps = prior.get("components") or []
    new_comps = []
    for c in comps:
        c = OmegaConf.to_container(c, resolve=False)
        if c.get("controller_config"):
            ref = _load_yaml(cfg_dir / "controller" / f"{c['controller_config']}.yaml")
            c["controller"] = _deep_merge(ref, c.get("controller", {}))
        new_comps.append(c)
    if comps:
        OmegaConf.update(cfg, "prior.components", new_comps, merge=False)
    return cfg


def to_container(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def save_resolved(cfg: DictConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)


def config_hash(cfg: DictConfig) -> str:
    from ..core.specs import stable_hash

    d = to_container(cfg)
    for k in ("run_name", "output_dir", "resume", "logging", "tags", "notes"):  # identity excludes logging backend routing
        d.pop(k, None)
    return stable_hash(d)


def default_config(**overrides) -> DictConfig:
    """Schema defaults with dotted overrides (for tests and programmatic use): default_config(**{"train.batch_size": 64})."""
    cfg = validate_schema(OmegaConf.create({}))
    OmegaConf.set_struct(cfg, False)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=True)
    OmegaConf.set_struct(cfg, True)
    return validate_schema(cfg)
