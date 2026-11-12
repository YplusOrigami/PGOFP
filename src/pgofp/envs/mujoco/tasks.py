"""Supported MuJoCo task table: exact env ids, make kwargs, which controller priors are valid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...core.exceptions import UnsupportedCombinationError


@dataclass(frozen=True)
class TaskInfo:
    env_id: str
    controllers: Tuple[str, ...]  # prior types that are meaningful for this task
    obs_names: Tuple[str, ...] = ()
    notes: str = ""
    default_kwargs: Dict = field(default_factory=dict)


TASKS: Dict[str, TaskInfo] = {
    "InvertedDoublePendulum-v5": TaskInfo(
        "InvertedDoublePendulum-v5", ("cem_mpc", "ppo"),
        ("x", "sin_q1", "sin_q2", "cos_q1", "cos_q2", "xdot", "q1dot", "q2dot", "constraint_force"),
        "Two-link upright balancing; MPC reconstructs public state with clipped velocities. No PID implemented."),
    "InvertedPendulum-v5": TaskInfo(
        "InvertedPendulum-v5", ("pid", "cem_mpc", "ppo"), ("x", "theta", "x_dot", "theta_dot"),
        "Reward saturates at 1/step; use failure/episode length and noisy-source-only controls, not return alone."),
    "Reacher-v5": TaskInfo(
        "Reacher-v5", ("pid", "cem_mpc", "ppo"),
        ("cos_q1", "cos_q2", "sin_q1", "sin_q2", "target_x", "target_y", "qd1", "qd2", "tip_dx", "tip_dy"),
        "Joint PD around a fixed-elbow analytic IK reference derived from the public target coordinates."),
    "HalfCheetah-v5": TaskInfo("HalfCheetah-v5", ("ppo",), notes="No useful per-joint PID prior is assumed."),
    "Hopper-v5": TaskInfo("Hopper-v5", ("ppo",), notes="Healthy termination + time limit are kept separate."),
    "Walker2d-v5": TaskInfo("Walker2d-v5", ("ppo",), notes="Extension task; not in the default main matrix."),
}


def check_prior_supported(env_id: str, prior_type: str) -> None:
    if prior_type in ("none", "multi"):
        return
    info = TASKS.get(env_id)
    if info is None:
        raise UnsupportedCombinationError(f"task {env_id!r} not in the MuJoCo task table")
    if prior_type not in info.controllers:
        raise UnsupportedCombinationError(
            f"prior {prior_type!r} is not supported for {env_id} (supported: {info.controllers}); "
            "mark as skipped in the experiment manifest")
