from __future__ import annotations

from ..core.exceptions import ConfigError
from ..core.specs import EnvSpec
from ..networks.conditioning import ConditionBuilder
from .base import BaseAgent


def build_agent(cfg, env_spec: EnvSpec, builder: ConditionBuilder, device) -> BaseAgent:
    name = cfg.algorithm.name
    if name in ("prior_flame_m", "gaussian_flame_m"):
        from .prior_flame import PriorFlameAgent

        return PriorFlameAgent(cfg, env_spec, builder, device)
    if name == "online_ofql":
        from .online_ofql import OnlineOFQLAgent

        return OnlineOFQLAgent(cfg, env_spec, builder, device)
    if name == "sac":
        from .baselines import SACAgent

        return SACAgent(cfg, env_spec, builder, device)
    if name == "controller_only":
        from .baselines import ControllerOnlyAgent

        return ControllerOnlyAgent(cfg, env_spec, builder, device)
    if name == "flame_reference":
        from .flame_reference import FlameReferenceAgent

        return FlameReferenceAgent(cfg, env_spec, builder, device)
    raise ConfigError(f"algorithm {name!r} is not handled by the off-policy runner (ppo uses scripts/train_prior.py / PPOTrainer)")
