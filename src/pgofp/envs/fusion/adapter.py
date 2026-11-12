"""Gymnasium-style fusion environment over a FusionBackend protocol (mock now, HFM later).

7-D normalized command -> (safety filter) -> 12-D voltages via the verified coil map -> optional wrapper actuator stages
(only those the wrapper OWNS) -> backend.advance.  Critic action = the issued 7-D command; reward/transition come from
the backend result.  Terminal semantics: terminated vs truncated are reported exactly as the backend states them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from ...core.exceptions import BackendUnavailableError, ConfigError
from ...core.specs import ActionSpec, EnvSpec
from .actuator import ActuatorWrapper, CoilMap
from .mock_backend import MOCK_FIELDS, MockFusionBackend
from .observation import ObservationAssembler
from .protocol import FusionBackend
from .safety import SafetyFilter

N_CMD = 7
CFG_DIR = Path(__file__).resolve().parents[4] / "configs" / "fusion"


class FusionEnv:
    def __init__(self, backend: FusionBackend, coil_map: CoilMap, lookahead: int = 0, safety: Optional[SafetyFilter] = None,
                 shot_ids: Optional[List[str]] = None, reset_params: Optional[dict] = None, seed: int = 0):
        self.backend = backend
        self.desc = backend.describe()
        if self.desc.n_commands != coil_map.n_cmd:
            raise ConfigError(f"backend expects {self.desc.n_commands} commands, coil map has {coil_map.n_cmd}")
        if not self.desc.is_mock and not coil_map.verified:
            raise BackendUnavailableError("real fusion backend requires a verified coil map")
        self.coil = coil_map
        self.assembler = ObservationAssembler(self.desc.obs_schema)
        self.actuator = ActuatorWrapper(self.desc.ownership, delay_steps=0, rate_limit_frac=None)
        self.safety = safety or SafetyFilter(enabled=False)
        self.lookahead = lookahead
        self.shot_ids = shot_ids or ["shot0"]
        self.reset_params = reset_params
        self._shot_i = 0
        self.t = 0
        self.horizon = getattr(backend, "horizon", None)
        ref0 = backend.get_reference(0, lookahead)
        self.spec = EnvSpec(
            env_id="FusionMock-v0" if self.desc.is_mock else "Fusion-v0", obs_dim=self.assembler.dim,
            action=ActionSpec((-1.0,) * N_CMD, (1.0,) * N_CMD), ref_dim=int(ref0.size), phase_dim=1,
            max_episode_steps=self.horizon, control_dt=self.desc.control_period_s,
            obs_names=tuple(self.assembler.names))
        self.last_info: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None, options=None):
        shot = self.shot_ids[self._shot_i % len(self.shot_ids)]
        self._shot_i += 1
        vals, info = self.backend.reset(shot, self.reset_params, seed)
        self.actuator.reset()
        self.safety.reset()
        self.t = 0
        vec, _ = self.assembler.assemble(vals)
        return vec, info

    def step(self, a_cmd: np.ndarray):
        fr = self.safety(a_cmd)
        v12 = self.coil.expand(fr.filtered_command)
        v_applied = self.actuator.apply(v12)  # only wrapper-owned stages act
        out = self.backend.advance(v_applied)
        self.t += 1
        vec, mask = self.assembler.assemble(out.observation, out.obs_mask)
        info = {"raw_command": fr.raw_command, "filtered_command": fr.filtered_command, "intervention": fr.intervention,
                "intervention_reason": fr.reason, "reward_terms": out.reward_terms, "termination_reason": out.termination_reason,
                "applied_voltage": out.applied_voltage, "obs_mask": mask}
        self.last_info = info
        return vec, float(out.reward), bool(out.terminated), bool(out.truncated), info

    def reference(self) -> np.ndarray:
        return np.asarray(self.backend.get_reference(self.t, self.lookahead), np.float32)

    def phase(self, step_in_episode: int) -> np.ndarray:
        h = self.horizon or 1
        return np.asarray([1.0 - step_in_episode / h], np.float32)

    def get_snapshot(self):
        return {"backend": self.backend.get_snapshot(), "t": self.t, "shot_i": self._shot_i}

    def set_snapshot(self, snap) -> None:
        self.backend.set_snapshot(snap["backend"])
        self.t, self._shot_i = snap["t"], snap["shot_i"]

    def close(self) -> None:
        pass


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CFG_DIR / name).read_text())


def validate_fusion_config(cfg) -> List[str]:
    warns: List[str] = []
    if cfg.env.backend == "fusion":
        for fn in ("observation_schema.yaml", "coil_map.yaml", "tasks.yaml"):
            d = _load_yaml(fn)
            if not d.get("verified", False):
                raise BackendUnavailableError(f"configs/fusion/{fn} has verified:false; the real fusion backend refuses to run "
                                              "until real device/SDK information is filled in (see docs/fusion_integration.md)")
    else:
        warns.append("fusion_mock backend: proves interface contracts only, nothing about the real HFM simulator")
    return warns


def make_fusion_env_fn(cfg, kwargs: dict) -> Callable[[int], FusionEnv]:
    if cfg.env.backend == "fusion_mock":
        def make(i: int) -> FusionEnv:
            be = MockFusionBackend(horizon=int(kwargs.get("horizon", 60)), delay_steps=int(kwargs.get("delay_steps", 2)),
                                   terminal_on_horizon=bool(kwargs.get("terminal_on_horizon", False)), seed=i)
            safety = SafetyFilter(max_delta=kwargs.get("safety_max_delta"), enabled=bool(kwargs.get("safety", False)))
            return FusionEnv(be, CoilMap.mock_symmetric(), int(kwargs.get("lookahead", 0)), safety, seed=i)
        return make
    factory = kwargs.get("sdk_factory")
    if not factory:
        raise BackendUnavailableError("env.backend=fusion needs env.kwargs.sdk_factory='module:function' returning a FusionBackend; "
                                      "the real HFM SDK interface has not been provided")
    import importlib

    mod, fn = factory.split(":")
    try:
        create = getattr(importlib.import_module(mod), fn)
    except Exception as e:
        raise BackendUnavailableError(f"cannot import SDK factory {factory!r}: {e}") from e
    coil = CoilMap.from_yaml(CFG_DIR / "coil_map.yaml")
    return lambda i: FusionEnv(create(**dict(kwargs.get("sdk_kwargs", {}))), coil, int(kwargs.get("lookahead", 0)))
