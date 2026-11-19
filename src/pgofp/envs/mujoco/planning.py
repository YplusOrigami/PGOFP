"""Planning models for sampling MPC, independent from the online environment.

capability `observation_model`: the model is a separate MjModel instance and the initial state is reconstructed
ONLY from the public observation (never from env.unwrapped.data).  For InvertedPendulum and Reacher the public
observation determines the full MuJoCo state (qpos, qvel), so this equals an oracle there; `privileged_sim_state`
is therefore rejected explicitly instead of silently aliasing it.
"""
from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np

from ...core.exceptions import ConfigError, UnsupportedCombinationError
from .features import REACHER_L1, REACHER_L2


class RolloutPlanner:
    def __init__(self, env_id: str, kwargs: dict | None = None, nthread: int = 1):
        import gymnasium as gym
        import mujoco
        from mujoco import rollout

        if env_id not in ("InvertedPendulum-v5", "Reacher-v5", "InvertedDoublePendulum-v5"):
            raise UnsupportedCombinationError(f"no observation->state map for planning on {env_id}")
        self.env_id = env_id
        self._mujoco, self._rollout = mujoco, rollout
        env = gym.make(env_id, **(kwargs or {}))
        u = env.unwrapped
        self.model = copy.deepcopy(u.model)  # independent copy: planning never touches the online env
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(u.frame_skip)
        self.nu = int(self.model.nu)
        self.nstate = int(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.nq, self.nv = int(self.model.nq), int(self.model.nv)
        env.close()
        # `mujoco.rollout.rollout` only parallelizes across the batch (population) dimension when
        # `data` is a sequence of >1 MjData instances; nthread is inferred as len(data). A persistent
        # thread pool avoids re-spawning threads on every `plan()` call (one call per CEM iteration,
        # every env step). nthread=1 reproduces the prior single-threaded behavior exactly (same
        # outputs, since rollout results are a pure function of model/init/control regardless of how
        # the batch is partitioned across threads).
        self.nthread = max(1, int(nthread))
        self._rollout_data = [mujoco.MjData(self.model) for _ in range(self.nthread)] if self.nthread > 1 else self.data

    # ------------------------------------------------------------------ obs -> state
    def state_from_obs(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, np.float64)
        s = np.zeros(self.nstate)
        if self.env_id == "InvertedPendulum-v5":
            qpos, qvel = o[0:2], o[2:4]
        elif self.env_id == "InvertedDoublePendulum-v5":
            # Public observations clip velocities to [-10,10]; this is an
            # observation-model approximation if a velocity reaches that bound.
            qpos = np.array([o[0], np.arctan2(o[1], o[3]), np.arctan2(o[2], o[4])])
            qvel = o[5:8]
        else:  # Reacher: q1, q2 from cos/sin; target position is qpos[2:4]; target velocities are zero
            q = np.array([np.arctan2(o[2], o[0]), np.arctan2(o[3], o[1])])
            qpos, qvel = np.concatenate([q, o[4:6]]), np.concatenate([o[6:8], [0.0, 0.0]])
        s[1:1 + self.nq] = qpos
        s[1 + self.nq:1 + self.nq + self.nv] = qvel
        return s

    # ------------------------------------------------------------------ rollouts
    def rollout(self, init_state: np.ndarray, ctrl_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ctrl_norm [P,H,nu] in [-1,1] -> (qpos [P,H,nq], qvel [P,H,nv]) sampled at control instants."""
        P, H, _ = ctrl_norm.shape
        bias, scale = (self.ctrl_high + self.ctrl_low) / 2, (self.ctrl_high - self.ctrl_low) / 2
        phys = bias + scale * np.clip(ctrl_norm, -1.0, 1.0)
        ctrl = np.repeat(phys, self.frame_skip, axis=1)  # hold each command for frame_skip physics steps
        init = np.broadcast_to(init_state, (P, self.nstate)).copy()
        states, _ = self._rollout.rollout(self.model, self._rollout_data, init, ctrl, persistent_pool=self.nthread > 1)
        idx = np.arange(1, H + 1) * self.frame_skip - 1
        st = states[:, idx, :]
        return st[..., 1:1 + self.nq], st[..., 1 + self.nq:1 + self.nq + self.nv]

    # ------------------------------------------------------------------ costs (computable from the public model/obs)
    def stage_cost(self, qpos, qvel, ctrl_norm, target_xy, cfg: Dict) -> np.ndarray:
        """Returns per-candidate total cost [P]."""
        if self.env_id == "InvertedDoublePendulum-v5":
            tip_x, height = double_pendulum_tip(qpos)
            penalty = (0.01 * tip_x**2 + (height - 2.0)**2
                       + 1e-3 * qvel[..., 1]**2 + 5e-3 * qvel[..., 2]**2)
            # A planned fall is absorbing; the independent simulator otherwise
            # continues after the environment would have terminated.
            fallen = np.maximum.accumulate(height <= 1.0, axis=1)
            return (penalty + cfg.get("healthy_reward", 10.0) * fallen).sum(1)
        if self.env_id == "InvertedPendulum-v5":
            x, th = qpos[..., 0], qpos[..., 1]
            xd, thd = qvel[..., 0], qvel[..., 1]
            c = (cfg.get("w_theta", 10.0) * th**2 + cfg.get("w_x", 0.5) * x**2 + cfg.get("w_xdot", 0.05) * xd**2
                 + cfg.get("w_thdot", 0.1) * thd**2 + cfg.get("w_u", 0.01) * (ctrl_norm[..., 0] ** 2))
            fail = (np.abs(th) > 0.2).any(1)
            return c.sum(1) + cfg.get("fail_penalty", 50.0) * fail
        q1, q2 = qpos[..., 0], qpos[..., 1]
        tip = np.stack([REACHER_L1 * np.cos(q1) + REACHER_L2 * np.cos(q1 + q2),
                        REACHER_L1 * np.sin(q1) + REACHER_L2 * np.sin(q1 + q2)], -1)
        dist = np.linalg.norm(tip - target_xy, axis=-1)
        c = (cfg.get("w_dist", 1.0) * dist + cfg.get("w_dist_sq", 5.0) * dist**2
             + cfg.get("w_qdot", 0.01) * (qvel[..., :2] ** 2).sum(-1) + cfg.get("w_u", 0.1) * (ctrl_norm**2).sum(-1))
        return c.sum(1)


def double_pendulum_tip(qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tip position for the standard two 0.6m links, verified against MuJoCo FK."""
    x, a, b = qpos[..., 0], qpos[..., 1], qpos[..., 2]
    return x + 0.6 * np.sin(a) + 0.6 * np.sin(a + b), 0.6 * np.cos(a) + 0.6 * np.cos(a + b)

