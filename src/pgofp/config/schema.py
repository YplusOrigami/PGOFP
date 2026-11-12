"""Structured configuration schema. Unknown keys / wrong types are rejected at compose time."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainCfg:
    num_envs: int = 1
    total_env_steps: int = 1_000_000
    learning_starts: int = 5_000
    utd_ratio: float = 1.0
    batch_size: int = 256
    warmup_source: str = "uniform_action"  # uniform_action | prior | mixed
    max_grad_norm: float = 10.0
    actor_update_interval: int = 1
    log_interval: int = 1_000
    sync_device_timing: bool = False


@dataclass
class ReplayCfg:
    capacity: int = 1_000_000


@dataclass
class NetworkCfg:
    actor_hidden: List[int] = field(default_factory=lambda: [256, 256, 256])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256, 256])
    time_embed_dim: int = 32
    time_max_freq: float = 4.0
    actor_activation: str = "silu"
    critic_activation: str = "relu"
    layer_norm: bool = False


@dataclass
class OptimCfg:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4


@dataclass
class FlowCfg:
    time_delta: float = 0.02
    flow_ratio: float = 0.5
    time_sampling: str = "uniform"  # uniform | logit_normal
    logit_mean: float = -0.4
    logit_std: float = 1.0


@dataclass
class TemperatureCfg:
    mode: str = "fixed"  # fixed | auto
    alpha: float = 0.1
    target_entropy: Any = "auto"  # "auto" => -action_dim (SAC-style start; ablate)
    alpha_min: float = 1e-3
    alpha_max: float = 10.0


@dataclass
class DensityCfg:
    method: str = "continuous_inverse_proxy"
    steps: int = 5
    solver: str = "heun"
    trace: str = "hutchinson"
    n_trace_samples: int = 1
    chunk_size: int = 8192
    grid: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    logdet: str = "full"


@dataclass
class AlgorithmCfg:
    name: str = "prior_flame_m"
    gamma: float = 0.99
    target_tau: float = 0.005
    K_train: int = 32
    K_local: int = 16  # defensive local proposal around current-policy samples (MIS-corrected); 0 = pure reverse kernel (spec-literal)
    sigma_local: float = 0.3
    K_exec: int = 1
    K_eval: int = 1
    actor_objective: str = "reverse_qrmf"  # reverse_qrmf | endpoint_is | reverse_no_q | reverse_uniform
    use_entropy: bool = True
    zt_proposal: str = "standard_normal"  # standard_normal | replay_action
    source_scope: List[str] = field(default_factory=lambda: ["behavior", "actor", "target"])
    atanh_eps: float = 1e-3
    flow: FlowCfg = field(default_factory=FlowCfg)
    temperature: TemperatureCfg = field(default_factory=TemperatureCfg)
    density: DensityCfg = field(default_factory=DensityCfg)
    params: Dict[str, Any] = field(default_factory=dict)  # baseline-specific knobs (ofql / sac / ppo / ...)
    forces: Dict[str, Any] = field(default_factory=dict)  # dotted-path -> forced value (e.g. source.kind: gaussian)
    inherits: Optional[str] = None


@dataclass
class EnvCfg:
    name: str = ""
    id: str = "InvertedPendulum-v5"
    backend: str = "mujoco"  # mujoco | toy | fusion_mock | fusion
    kwargs: Dict[str, Any] = field(default_factory=dict)
    normalize_obs: bool = True
    obs_clip: float = 10.0
    reward_scale: float = 1.0
    controllers: List[str] = field(default_factory=list)  # controller keys this task supports
    notes: str = ""


@dataclass
class PriorCfg:
    type: str = "none"  # none | pid | cem_mpc | ppo | multi
    controller_config: Optional[str] = None  # configs/controller/<name>.yaml
    controller: Dict[str, Any] = field(default_factory=dict)  # resolved inline params
    artifact_path: Optional[str] = None  # PriorArtifact dir (ppo)
    distribution: str = "native"  # ppo: native | center_gaussian
    components: List[Dict[str, Any]] = field(default_factory=list)  # multi-controller components
    fallback: str = "gaussian"
    model_access: str = "observation_model"
    cache_in_replay: bool = True


@dataclass
class SourceCfg:
    kind: str = "gaussian"  # gaussian | prior_mixture
    schedule: str = "constant"  # constant | linear | piecewise | cosine
    beta: float = 0.75
    sigma_prior: float = 0.3
    sigma_base: float = 1.0
    sigma_min: float = 0.05
    sigma_max: float = 2.0
    std_multiplier: float = 1.0
    total_steps: int = 1_000_000
    beta_start: float = 0.0
    beta_end: float = 0.75
    sigma_start: float = 1.0
    sigma_end: float = 0.3
    points: List[List[float]] = field(default_factory=list)
    max_delta: float = 0.0


@dataclass
class ConditionCfg:
    mode: str = "public_state"  # public_state | prior_features
    use_reference: bool = True
    use_phase: bool = True
    use_controller_features: bool = False
    source_features: str = "auto"
    reference_scale: float = 1.0


@dataclass
class WandbCfg:
    mode: str = "disabled"  # disabled | offline | online
    project: str = "pgofp"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    upload_artifacts: bool = False
    require_online: bool = False  # opt-in: fail before training instead of silently losing cloud logging
    scalar_source: str = "tensorboard"  # tensorboard sync OR direct MetricRecord logging, never both


@dataclass
class LoggingCfg:
    jsonl: bool = True
    tensorboard: bool = True
    flush_every: int = 20
    wandb: WandbCfg = field(default_factory=WandbCfg)
    histogram_interval: int = 0
    video_interval: int = 0
    console_progress: bool = False


@dataclass
class EvalCfg:
    interval: int = 10_000
    episodes: int = 10
    seed_base: int = 10_000
    max_steps: Optional[int] = None
    condition_bank_size: int = 0
    representative_action: bool = True
    prior_before_training: bool = False


@dataclass
class CheckpointCfg:
    interval: int = 50_000
    save_replay: bool = True
    best_metric: str = "eval/return_mean"


@dataclass
class Config:
    seed: int = 0
    run_name: Optional[str] = None
    output_dir: str = "outputs/runs"
    device: str = "auto"
    resume: Optional[str] = None
    resume_mode: str = "episode_restart"  # exact_resume | episode_restart | weights_only
    train: TrainCfg = field(default_factory=TrainCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    network: NetworkCfg = field(default_factory=NetworkCfg)
    optimizer: OptimCfg = field(default_factory=OptimCfg)
    algorithm: AlgorithmCfg = field(default_factory=AlgorithmCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    prior: PriorCfg = field(default_factory=PriorCfg)
    source: SourceCfg = field(default_factory=SourceCfg)
    condition: ConditionCfg = field(default_factory=ConditionCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    smoke: bool = False  # smoke runs must never enter formal statistics
