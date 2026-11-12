"""Environment construction from config (train / eval / planning are independent instances)."""
from __future__ import annotations

from typing import Callable

from omegaconf import OmegaConf

from ..core.exceptions import ConfigError
from .vector import SyncEnvPool


def make_env_fn(cfg) -> Callable[[int], object]:
    backend = cfg.env.backend
    kwargs = dict(OmegaConf.to_container(cfg.env.kwargs, resolve=True))
    if backend == "mujoco":
        from .adapters import GymAdapter
        from .mujoco.tasks import TASKS

        if cfg.env.id not in TASKS:
            raise ConfigError(f"MuJoCo task {cfg.env.id!r} is not in the supported task table {sorted(TASKS)}")
        return lambda i: GymAdapter(cfg.env.id, kwargs)
    if backend == "toy":
        from .toy import ToyReach

        return lambda i: ToyReach(action_dim=int(kwargs.get("action_dim", 2)), horizon=int(kwargs.get("horizon", 20)),
                                  ref_dim=int(kwargs.get("ref_dim", 0)), phase_dim=int(kwargs.get("phase_dim", 0)), seed=i)
    if backend in ("fusion_mock", "fusion"):
        from .fusion.adapter import make_fusion_env_fn

        return make_fusion_env_fn(cfg, kwargs)
    raise ConfigError(f"unknown env backend {backend!r}")


def make_pool(cfg, n_envs: int, base_seed: int, tag: str) -> SyncEnvPool:
    return SyncEnvPool(make_env_fn(cfg), n_envs, base_seed, tag)
