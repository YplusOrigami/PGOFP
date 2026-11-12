"""Evaluation with INDEPENDENT environment, controller instances and RNG streams (never touches training state)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..controllers.manager import ControllerManager
from ..core.specs import EnvSpec
from ..core.types import ContextBatch
from ..envs.vector import SyncEnvPool

PolicyFn = Callable[[Dict[str, np.ndarray]], np.ndarray]  # context dict (n=1 rows) -> normalized action [n, d]


@dataclass
class EpisodeResult:
    seed: int
    ret: float
    length: int
    terminated: bool
    truncated: bool
    extras: Dict[str, float] = field(default_factory=dict)


class Evaluator:
    def __init__(self, pool: SyncEnvPool, manager: ControllerManager, env_spec: EnvSpec):
        self.pool = pool
        self.manager = manager
        self.spec = env_spec
        self.n = pool.n
        assert self.n == 1, "evaluator uses a single env for strictly separated, reproducible episodes"

    def run(self, policy: PolicyFn, n_episodes: int, seed_base: int, max_steps: Optional[int] = None,
            progress: Optional[Callable[[int, int, int, int], None]] = None) -> List[EpisodeResult]:
        results: List[EpisodeResult] = []
        cap = max_steps or self.spec.max_episode_steps or 10_000
        for k in range(n_episodes):
            seed = seed_base + k
            obs = self.pool.reset_slot(0, seed=seed)[None]
            self.manager.reset_slots(np.ones(1, bool))
            self.manager.reset_stats()
            ref, ph = self._refs()
            ctx = self._ctx(obs, ref, ph, self.manager.predict(obs, ref))
            ret, t, term, trunc = 0.0, 0, False, False
            prev_a = None
            smooth, sat, extras_last = 0.0, 0.0, {}
            while True:
                a = np.asarray(policy(ctx), np.float32)  # [1, d]
                sat += float((np.abs(a) > 0.99).mean())
                if prev_a is not None:
                    smooth += float(np.abs(a - prev_a).mean())
                prev_a = a
                out = self.pool.step_slot(0, self.spec.action.to_physical(a[0]).astype(np.float32))
                ret += out.reward
                t += 1
                if progress:
                    progress(k, n_episodes, t, cap)
                self.manager.advance(ctx["obs_raw"], ctx["reference"], a)
                next_obs = out.next_obs[None].astype(np.float32)
                ref, ph = self._refs()
                term, trunc = out.terminated, out.truncated
                for key in ("reward_dist", "distance_from_origin"):
                    if key in out.info:
                        extras_last[key] = float(out.info[key])
                if term or trunc or t >= cap:
                    break
                ctx = self._ctx(next_obs, ref, ph, self.manager.predict(next_obs, ref))
            ex = {"action_saturation": sat / t, "action_smoothness": smooth / max(t - 1, 1), **extras_last}
            if "reward_dist" in extras_last:
                ex["final_distance"] = -extras_last["reward_dist"]
            ex.update({k: v for k, v in self.manager.stats().items()})
            results.append(EpisodeResult(seed, ret, t, bool(term), bool(trunc or (t >= cap and not term)), ex))
            if progress:
                progress(k + 1, n_episodes, t, cap)
        return results

    def _refs(self):
        ref = self.pool.reference(0)[None] if self.spec.ref_dim else np.zeros((1, 0), np.float32)
        ph = self.pool.phase(0)[None] if self.spec.phase_dim else np.zeros((1, 0), np.float32)
        return ref, ph

    @staticmethod
    def _ctx(obs, ref, ph, prior) -> Dict[str, np.ndarray]:
        return {"obs_raw": obs.astype(np.float32), "reference": ref.astype(np.float32), "phase": ph.astype(np.float32),
                "ctrl_feat": prior["ctrl_feat"], "prior_mu": prior["prior_mu"], "prior_logstd": prior["prior_logstd"],
                "prior_valid": prior["prior_valid"]}


def summarize(results: List[EpisodeResult], prefix: str = "eval") -> Dict[str, float]:
    if not results:
        return {}
    rets = np.asarray([r.ret for r in results])
    lens = np.asarray([r.length for r in results], float)
    out = {
        f"{prefix}/return_mean": float(rets.mean()), f"{prefix}/return_std": float(rets.std()),
        f"{prefix}/length_mean": float(lens.mean()), f"{prefix}/failure_rate": float(np.mean([r.terminated for r in results])),
        f"{prefix}/episodes": float(len(results)),
    }
    for key in ("action_saturation", "action_smoothness", "final_distance"):
        vals = [r.extras[key] for r in results if key in r.extras]
        if vals:
            out[f"{prefix}/{key}"] = float(np.mean(vals))
    return out
