"""Named, public-observation feature extraction for MuJoCo tasks (no hidden-state access)."""
from __future__ import annotations

from typing import Dict

import numpy as np

# Reacher-v5 observation layout (verified against env.unwrapped):
#   0 cos q1, 1 cos q2, 2 sin q1, 3 sin q2, 4 target x, 5 target y, 6 qvel1, 7 qvel2, 8 (fingertip-target) x, 9 ... y
REACHER_L1, REACHER_L2 = 0.1, 0.11  # link lengths of the standard model (link0 / link1 body offsets)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def inverted_pendulum_features(obs: np.ndarray) -> Dict[str, np.ndarray]:
    obs = np.asarray(obs, np.float64)
    return {"x": obs[..., 0], "theta": obs[..., 1], "x_dot": obs[..., 2], "theta_dot": obs[..., 3]}


def reacher_features(obs: np.ndarray) -> Dict[str, np.ndarray]:
    obs = np.asarray(obs, np.float64)
    q1 = np.arctan2(obs[..., 2], obs[..., 0])
    q2 = np.arctan2(obs[..., 3], obs[..., 1])
    return {
        "q": np.stack([q1, q2], -1),
        "qdot": obs[..., 6:8],
        "target": obs[..., 4:6],
        "fingertip_minus_target": obs[..., 8:10],
    }


def reacher_ik(target_xy: np.ndarray, elbow: int = +1, l1: float = REACHER_L1, l2: float = REACHER_L2) -> np.ndarray:
    """Two-link analytic IK with a FIXED elbow branch (elbow=+1: q2>=0). Returns [..., 2] joint angles."""
    t = np.asarray(target_xy, np.float64)
    r2 = (t**2).sum(-1)
    c2 = np.clip((r2 - l1**2 - l2**2) / (2 * l1 * l2), -1.0, 1.0)
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(t[..., 1], t[..., 0]) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.stack([q1, q2], -1)


FEATURES = {
    "InvertedPendulum-v5": inverted_pendulum_features,
    "Reacher-v5": reacher_features,
}
