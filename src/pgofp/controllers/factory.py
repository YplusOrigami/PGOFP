"""Build controllers (prior providers) from the resolved config. Train and eval get SEPARATE instances."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf

from ..core.exceptions import ConfigError
from ..core.registry import CONTROLLERS
from ..core.specs import EnvSpec
from . import mpc, pid, ppo  # noqa: F401  (registration side effects)
from .base import BaseController


def _build_one(ptype: str, cdict: Dict[str, Any], env_spec: EnvSpec, env_kwargs: dict, artifact_path, distribution: str,
               seed: int) -> BaseController:
    if ptype in ("pid", "cem_mpc"):
        name = cdict.get("name") or ("cem_mpc" if ptype == "cem_mpc" else None)
        if not name:
            raise ConfigError("prior.controller.name is required (e.g. inverted_pendulum_pid)")
        cdict = dict(cdict)
        cdict.setdefault("seed", seed)
        return CONTROLLERS.build(name, cdict, env_spec, env_kwargs=env_kwargs)
    if ptype == "ppo":
        return CONTROLLERS.build("ppo", cdict, env_spec, distribution=distribution, artifact_path=artifact_path)
    raise ConfigError(f"unknown prior type {ptype!r}")


def build_controllers(cfg, env_spec: EnvSpec, seed: int = 0) -> Tuple[List[BaseController], Tuple[float, ...]]:
    """Returns (controllers, component_weights). component_weights has one entry per prior component M."""
    p = cfg.prior
    kw = dict(OmegaConf.to_container(cfg.env.kwargs, resolve=True))
    if p.type == "none":
        return [], ()
    if p.type == "multi":
        if not p.components:
            raise ConfigError("prior.type=multi needs prior.components")
        ctrls, w = [], []
        for i, c in enumerate(OmegaConf.to_container(p.components, resolve=True)):
            ctrls.append(_build_one(c["type"], c.get("controller", {}), env_spec, kw, c.get("artifact_path"),
                                    c.get("distribution", "native"), seed + i))
            w.append(float(c.get("weight", 1.0)))
        tot = sum(w)
        return ctrls, tuple(x / tot for x in w)
    ctrl = _build_one(p.type, OmegaConf.to_container(p.controller, resolve=True), env_spec, kw, p.artifact_path,
                      p.distribution, seed)
    return [ctrl], (1.0,)
