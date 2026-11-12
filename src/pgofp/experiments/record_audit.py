"""Success-run logging contract, shared by local and cloud acceptance checks."""
from __future__ import annotations

import math
from pathlib import Path

CORE_METRICS = {
    "loss/actor_qrmf", "loss/meanflow_diag", "loss/meanflow_offdiag", "critic/loss",
    "critic/q1_mean", "critic/q2_mean", "critic/target_mean", "critic/td_error",
    "actor/grad_norm", "critic/grad_norm", "weights/ess_mean", "weights/finite_ratio",
    "weights/max_weight_mean", "flow/diag_fraction", "flow/jvp_norm", "flow/one_step_gap_action",
    "entropy/proxy_mean", "entropy/inverse_residual", "alpha", "prior/beta", "prior/sigma_prior",
    "prior/fallback_rate", "prior/controller_latency_ms", "charts/env_steps", "charts/gradient_steps",
    "charts/replay_size", "charts/steps_per_second", "charts/wall_time_sec", "actions/saturation_frac",
    "timing/collect_sec", "timing/update_sec", "eval/return_mean", "eval/return_std",
    "eval/length_mean", "eval/failure_rate", "eval/episodes", "eval/action_saturation", "eval/action_smoothness",
}
REQUIRED_FILES = (
    "config_resolved.yaml", "manifest.json", "metrics.jsonl", "events.jsonl",
    "evaluation/episodes.csv", "evaluation/summary.json", "checkpoints/best.pt", "checkpoints/best.json",
    "checkpoints/last.pt", "checkpoints/last.json", "checkpoints/final.pt", "checkpoints/final.json",
)


def audit_run(directory: Path, cfg, records: list[dict]) -> dict:
    keys = {k for r in records for k, value in r.items()
            if isinstance(value, (float, int)) and math.isfinite(value)}
    required = set(CORE_METRICS)
    if cfg.eval.representative_action:
        required |= {"eval_representative/return_mean", "eval_representative/return_std"}
    if cfg.eval.prior_before_training:
        required |= {"prior_eval/return_mean", "prior_eval/return_std", "prior_eval/episodes"}
        if cfg.prior.type == "ppo":
            required.add("prior_eval_deterministic/return_mean")
    files = list(REQUIRED_FILES)
    if cfg.checkpoint.save_replay:
        files.append("checkpoints/replay/replay.npz")
    if cfg.eval.prior_before_training:
        files.append("evaluation/prior_baseline.json")
    missing_files = [name for name in files if not (directory / name).is_file()]
    missing_metrics = sorted(required - keys)
    return {"ok": not missing_files and not missing_metrics, "missing_files": missing_files,
            "missing_metrics": missing_metrics, "finite_metric_count": len(keys),
            "conditional_metrics": {
                "training_episode_returns": "Emitted when a training episode completes; no synthetic zero is written",
                "discarded_metrics": "metrics_discarded.jsonl exists only after truncating a resumed run",
            }}
