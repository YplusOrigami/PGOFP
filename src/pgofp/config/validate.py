"""Pre-flight validation: fail BEFORE any environment / training process is created."""
from __future__ import annotations

from pathlib import Path
from typing import List

from omegaconf import DictConfig

from ..core.exceptions import ConfigError, UnsupportedCombinationError

FLOW_ALGOS = ("prior_flame_m", "gaussian_flame_m")
OFF_POLICY_ALGOS = FLOW_ALGOS + ("online_ofql", "sac", "flame_reference")


def validate_config(cfg: DictConfig, check_artifacts: bool = True) -> List[str]:
    """Raises on hard errors; returns a list of warnings."""
    warns: List[str] = []
    a, s, p, e, t = cfg.algorithm, cfg.source, cfg.prior, cfg.env, cfg.train
    name = a.name
    if name == "gaussian_flame_m" and s.kind != "gaussian":
        raise ConfigError("gaussian_flame_m must use source.kind=gaussian (same code path as prior_flame_m)")
    if name == "prior_flame_m" and s.kind == "gaussian" and p.type != "none":
        warns.append("prior_flame_m with source.kind=gaussian: controller used only as a feature provider (ablation A13)")
    if s.kind == "prior_mixture":
        if p.type == "none":
            raise ConfigError("source.kind=prior_mixture needs a prior (prior.type != none)")
        if not (0.0 <= s.beta <= 1.0 and 0.0 <= s.beta_start <= 1.0 and 0.0 <= s.beta_end <= 1.0):
            raise ConfigError("beta values must lie in [0,1]")
    if not (0 < s.sigma_min <= s.sigma_max):
        raise ConfigError("need 0 < source.sigma_min <= source.sigma_max")
    if s.sigma_prior < s.sigma_min or s.sigma_prior > s.sigma_max:
        raise ConfigError("source.sigma_prior must lie inside [sigma_min, sigma_max]")
    if cfg.condition.mode == "prior_features" and p.type == "none":
        raise ConfigError("condition.mode=prior_features needs a prior")
    if s.schedule != "constant" and s.max_delta < 0:
        raise ConfigError("source.max_delta must be >= 0")
    if a.flow.flow_ratio < 0 or a.flow.flow_ratio > 1:
        raise ConfigError("algorithm.flow.flow_ratio must be in [0,1]")
    if a.flow.flow_ratio in (0.0, 1.0):
        warns.append(f"flow_ratio={a.flow.flow_ratio} is a boundary ablation (no diagonal / no average-velocity supervision)")
    if name in FLOW_ALGOS:
        if a.K_train < 1 or a.K_exec < 1:
            raise ConfigError("K_train and K_exec must be >= 1")
        if a.K_train == 1 and a.actor_objective == "reverse_qrmf" and a.K_local == 0:
            warns.append("K_train=1: no candidate-level Q preference (weights are trivially 1)")
        if a.K_exec > 1:
            warns.append("K_exec>1 is the best-of-K behavior variant, NOT the single-sample maximum-entropy policy")
        if a.use_entropy is False and a.temperature.mode == "auto":
            warns.append("use_entropy=false with temperature.mode=auto: alpha only acts as the Q-reweighting temperature")
    if t.batch_size > t.learning_starts and t.learning_starts > 0:
        warns.append("batch_size > learning_starts: early updates sample with heavy repetition")
    if t.utd_ratio < 0 or t.num_envs < 1:
        raise ConfigError("train.utd_ratio must be >= 0 and num_envs >= 1")
    if t.warmup_source not in ("uniform_action", "prior", "mixed"):
        raise ConfigError("train.warmup_source must be uniform_action|prior|mixed")
    if t.warmup_source in ("prior", "mixed") and p.type == "none":
        raise ConfigError("warmup_source prior/mixed needs a prior")
    if e.backend == "mujoco":
        from ..envs.mujoco.tasks import check_prior_supported

        check_prior_supported(e.id, p.type)
    if p.type == "ppo" and check_artifacts:
        if not p.artifact_path:
            raise ConfigError("prior.type=ppo requires prior.artifact_path (see scripts/train_prior.py)")
        if not Path(p.artifact_path).is_dir():
            raise ConfigError(f"prior.artifact_path {p.artifact_path!r} does not exist")
    if p.type in ("pid", "cem_mpc") and not p.controller:
        raise ConfigError(f"prior.type={p.type} needs prior.controller_config or inline prior.controller")
    if p.type == "multi":
        if not p.components:
            raise ConfigError("prior.type=multi needs prior.components")
        modes = {("native" if c.get("type") == "ppo" and c.get("distribution", "native") == "native" else "fixed")
                 for c in p.components}
        if len(modes) > 1:
            raise ConfigError("multi-controller sources must not mix native-std PPO and fixed-sigma components "
                              "(one SourceSpec std mode); use distribution=center_gaussian for the PPO component")
    if e.backend in ("fusion", "fusion_mock"):
        from ..envs.fusion.adapter import validate_fusion_config

        warns.extend(validate_fusion_config(cfg))
    if cfg.resume_mode not in ("exact_resume", "episode_restart", "weights_only"):
        raise ConfigError("resume_mode must be exact_resume|episode_restart|weights_only")
    if name not in OFF_POLICY_ALGOS and name not in ("ppo", "controller_only"):
        raise UnsupportedCombinationError(f"unknown algorithm {name!r}")
    if cfg.smoke:
        warns.append("smoke=true: this run must not enter formal statistics")
    return warns
