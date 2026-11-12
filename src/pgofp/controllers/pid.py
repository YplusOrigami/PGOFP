"""Vector PID controllers for MuJoCo tasks (gains come from config, never hard-coded task indices).

Features are extracted by named public-observation accessors (envs/mujoco/features.py). Integral states carry
anti-windup (conditional integration + clamp). Outputs are in NORMALIZED action coordinates [-1, 1].
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ..core.exceptions import ConfigError
from ..core.registry import CONTROLLERS
from ..core.specs import CapabilitySpec
from ..core.types import ControllerPrediction
from ..envs.mujoco.features import inverted_pendulum_features, reacher_features, reacher_ik, wrap_angle
from .base import BaseController, State


class InvertedPendulumPID(BaseController):
    """u_phys = Ktheta*theta + Komega*theta_dot + Kx*x + Kv*x_dot + Ki*I,  I' = clamp(I + x*dt).

    Signs were fixed empirically by small-perturbation response of InvertedPendulum-v5 (positive gains
    stabilize; negative gains fail within a few steps) -- see tests/integration/test_controllers.py.
    """

    capability = CapabilitySpec(deterministic=True, provides_std=False, stateful=True)
    n_components = 1
    ctrl_dim = 1

    def __init__(self, cfg: Dict, action_dim: int = 1):
        g = cfg.get("gains", {})
        self.k_theta = float(g.get("theta", 10.0))
        self.k_omega = float(g.get("theta_dot", 1.0))
        self.k_x = float(g.get("x", 0.5))
        self.k_v = float(g.get("x_dot", 0.5))
        self.k_i = float(g.get("integral_x", 0.0))
        self.dt = float(cfg.get("dt", 0.04))
        self.u_max = float(cfg.get("u_max_phys", 3.0))
        self.i_limit = float(cfg.get("integral_limit", 5.0))
        self.version = str(cfg.get("version", "ipend_pid_v1"))
        if action_dim != 1:
            raise ConfigError("InvertedPendulumPID needs a 1-D action")

    def initial_state(self, n_envs: int) -> State:
        return {"integral": np.zeros((n_envs, 1), np.float64)}

    def _u_unsat(self, obs, I):
        f = inverted_pendulum_features(obs)
        return (self.k_theta * f["theta"] + self.k_omega * f["theta_dot"] + self.k_x * f["x"]
                + self.k_v * f["x_dot"] + self.k_i * I[:, 0])

    def predict(self, obs, ref, state) -> ControllerPrediction:
        t0 = time.perf_counter()
        u = self._u_unsat(obs, state["integral"])
        norm = np.clip(u / self.u_max, -1.0, 1.0)[:, None, None]
        valid = np.isfinite(norm[:, :, 0]).all(-1, keepdims=True)
        norm = np.where(np.isfinite(norm), norm, 0.0)
        return ControllerPrediction(center_norm=norm, valid=valid, latency_s=time.perf_counter() - t0, version=self.version)

    def advance(self, state, obs, ref, applied_cmd_norm) -> State:
        I = state["integral"]
        u = self._u_unsat(obs, I)
        saturated = np.abs(u) > self.u_max
        x = inverted_pendulum_features(obs)["x"]
        # conditional integration: freeze while saturated unless the error drives the integral back inside
        grow = ~saturated | (np.sign(x) != np.sign(u))
        newI = np.clip(I[:, 0] + np.where(grow, x * self.dt, 0.0), -self.i_limit, self.i_limit)
        return {"integral": newI[:, None]}

    def features(self, state) -> np.ndarray:
        return (state["integral"] / self.i_limit).astype(np.float32)


class ReacherPID(BaseController):
    """Joint-space PD(+I) around a fixed-elbow analytic IK reference of the public target coordinates."""

    capability = CapabilitySpec(deterministic=True, provides_std=False, stateful=True)
    n_components = 1
    ctrl_dim = 2

    def __init__(self, cfg: Dict, action_dim: int = 2):
        g = cfg.get("gains", {})
        self.kp = np.asarray(g.get("kp", [20.0, 20.0]), np.float64)
        self.kd = np.asarray(g.get("kd", [1.0, 1.0]), np.float64)
        self.ki = np.asarray(g.get("ki", [0.0, 0.0]), np.float64)
        self.dt = float(cfg.get("dt", 0.02))
        self.elbow = int(cfg.get("elbow", 1))
        self.i_limit = float(cfg.get("integral_limit", 1.0))
        self.u_max = 1.0
        self.version = str(cfg.get("version", "reacher_pid_v1"))
        if action_dim != 2 or self.kp.shape != (2,):
            raise ConfigError("ReacherPID needs 2 joints / 2 gains")

    def initial_state(self, n_envs: int) -> State:
        return {"integral": np.zeros((n_envs, 2), np.float64)}

    def _err(self, obs):
        f = reacher_features(obs)
        q_ref = reacher_ik(f["target"], self.elbow)
        e = q_ref - f["q"]
        e[..., 0] = wrap_angle(e[..., 0])  # joint 0 is unlimited; joint 1 has a hard range, no wrap
        return e, f

    def _u(self, obs, I):
        e, f = self._err(obs)
        return self.kp * e + self.ki * I - self.kd * f["qdot"], e

    def predict(self, obs, ref, state) -> ControllerPrediction:
        t0 = time.perf_counter()
        u, _ = self._u(obs, state["integral"])
        norm = np.clip(u / self.u_max, -1.0, 1.0)[:, None, :]
        valid = np.isfinite(norm).all(-1)
        norm = np.where(np.isfinite(norm), norm, 0.0)
        return ControllerPrediction(center_norm=norm, valid=valid, latency_s=time.perf_counter() - t0, version=self.version)

    def advance(self, state, obs, ref, applied_cmd_norm) -> State:
        u, e = self._u(obs, state["integral"])
        saturated = np.abs(u) > self.u_max
        grow = ~saturated | (np.sign(e) != np.sign(u))
        I = np.clip(state["integral"] + np.where(grow, e * self.dt, 0.0), -self.i_limit, self.i_limit)
        return {"integral": I}

    def features(self, state) -> np.ndarray:
        return (state["integral"] / self.i_limit).astype(np.float32)


@CONTROLLERS.register("inverted_pendulum_pid")
def _build_ipend(cfg, env_spec, **kw):
    return InvertedPendulumPID(cfg, env_spec.action_dim)


@CONTROLLERS.register("reacher_pid")
def _build_reacher(cfg, env_spec, **kw):
    return ReacherPID(cfg, env_spec.action_dim)
