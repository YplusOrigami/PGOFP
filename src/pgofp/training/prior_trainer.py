"""Native tanh-Gaussian PPO: prior制备 (artifact export + calibration tiers) and the classic PPO comparison baseline.

Policy lives in the pre-tanh latent z ~ N(mu(s), std^2); a_norm = tanh(z). The PPO ratio uses the z-space density (the
tanh log-Jacobian is identical in numerator and denominator and cancels).  Observation normalization statistics are
frozen into the artifact: the PGOFP learner must query this prior with THESE statistics, never its own.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from ..controllers.artifacts import save_prior_artifact
from ..core.types import MetricRecord
from ..data.rollout_buffer import RolloutBuffer
from ..envs.factory import make_pool
from ..networks.gaussian_policy import TanhGaussianPolicy, ValueNet
from ..utils.normalization import RunningNorm
from ..utils.seed import RngStreams, derive_seed, seed_everything
from ..utils.torch_utils import clip_grad_norm


class PPOTrainer:
    def __init__(self, cfg, seed: int, device="cpu", n_envs: int = 1):
        p = dict(OmegaConf.to_container(cfg.algorithm.params, resolve=True))
        self.p = p
        self.gamma = float(cfg.algorithm.gamma)
        self.cfg, self.seed = cfg, seed
        self.device = torch.device(device)
        seed_everything(seed)
        self.rng = RngStreams(seed, self.device)
        self.pool = make_pool(cfg, n_envs, seed, "train")
        self.eval_pool = make_pool(cfg, 1, seed, "calib")
        self.spec = self.pool.spec
        self.n = n_envs
        hid = tuple(p.get("hidden", [64, 64]))
        self.policy = TanhGaussianPolicy(self.spec.obs_dim, self.spec.action_dim, hid, "tanh").to(self.device)
        self.value = ValueNet(self.spec.obs_dim, hid, "tanh").to(self.device)
        lr = float(p.get("lr", 3e-4))
        self.opt = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr, eps=1e-5)
        self.lr0 = lr
        self.norm = RunningNorm(self.spec.obs_dim, 10.0, enabled=bool(p.get("normalize_obs", True)))
        self.buf = RolloutBuffer(int(p.get("rollout_steps", 2048)), n_envs, self.spec.obs_dim, self.spec.action_dim)
        self.env_steps = 0
        self.obs = self.pool.reset_all()
        self.ep_ret = np.zeros(n_envs)
        self.ep_len = np.zeros(n_envs, int)
        self.recent: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ helpers
    def _nobs(self, obs: np.ndarray) -> torch.Tensor:
        snap = self.norm.snapshot(self.device)
        return snap.normalize(torch.as_tensor(obs, dtype=torch.float32, device=self.device))

    # ------------------------------------------------------------------ rollout / update
    def collect(self, progress_fn=None, total_steps: int = 0) -> np.ndarray:
        self.buf.reset()
        g = self.rng.torch("behavior_source")
        while not self.buf.full():
            self.norm.update(self.obs)
            x = self._nobs(self.obs)
            with torch.no_grad():
                z, a, logp = self.policy.sample(x, g)
                v = self.value(x)
            a_np = a.cpu().numpy().astype(np.float32)
            a_phys = self.spec.action.to_physical(a_np).astype(np.float32)
            outs = [self.pool.step_slot(i, a_phys[i]) for i in range(self.n)]
            nxt = np.stack([o.next_obs for o in outs]).astype(np.float32)
            rew = np.asarray([o.reward for o in outs], np.float32)
            term = np.asarray([o.terminated for o in outs], np.float32)
            trunc = np.asarray([o.truncated for o in outs], np.float32)
            tv = np.zeros(self.n, np.float32)
            if trunc.any():
                with torch.no_grad():
                    tv = (self.value(self._nobs(nxt)).cpu().numpy() * trunc).astype(np.float32)
            self.buf.add(x.cpu().numpy(), z.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy(), rew, term, trunc, tv)
            self.ep_ret += rew
            self.ep_len += 1
            self.env_steps += self.n
            if progress_fn and (self.env_steps % 64 == 0 or self.buf.full()):
                progress_fn("ppo_rollout", self.env_steps, total_steps)
            for i in range(self.n):
                if term[i] or trunc[i]:
                    self.recent.append({"ret": float(self.ep_ret[i]), "len": float(self.ep_len[i]), "term": float(term[i])})
                    self.ep_ret[i], self.ep_len[i] = 0.0, 0
                    nxt[i] = self.pool.reset_slot(i)
            self.obs = nxt
        with torch.no_grad():
            return self.value(self._nobs(self.obs)).cpu().numpy()

    def update(self, last_value: np.ndarray, frac_done: float) -> Dict[str, float]:
        p = self.p
        for gp in self.opt.param_groups:
            gp["lr"] = self.lr0 * (1.0 - frac_done)
        gae = self.buf.compute_gae(last_value, self.gamma, float(p.get("gae_lambda", 0.95)))
        data = self.buf.flatten(gae["adv"], gae["ret"], self.device)
        n = data["obs"].shape[0]
        mb = int(p.get("minibatch_size", 64))
        clip = float(p.get("clip_ratio", 0.2))
        gen = self.rng.numpy("replay")
        stats: Dict[str, List[float]] = {"loss/policy": [], "loss/value": [], "entropy/gaussian_z": [], "ppo/approx_kl": [], "ppo/clip_frac": []}
        for _ in range(int(p.get("epochs", 10))):
            perm = gen.permutation(n)
            for s in range(0, n, mb):
                idx = torch.as_tensor(perm[s:s + mb], device=self.device)
                x, z, old, adv, ret = data["obs"][idx], data["z"][idx], data["logp"][idx], data["adv"][idx], data["ret"][idx]
                if p.get("normalize_advantage", True):
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                mu, ls = self.policy.dist_params(x)
                logp = self.policy.log_prob_z(z, mu, ls)
                ratio = (logp - old).exp()
                pol = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                vloss = ((self.value(x) - ret) ** 2).mean()
                ent = self.policy.entropy_z(ls).mean()
                loss = pol + float(p.get("value_coef", 0.5)) * vloss - float(p.get("entropy_coef", 0.0)) * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm(list(self.policy.parameters()) + list(self.value.parameters()), float(p.get("max_grad_norm", 0.5)))
                self.opt.step()
                with torch.no_grad():
                    stats["loss/policy"].append(float(pol)); stats["loss/value"].append(float(vloss)); stats["entropy/gaussian_z"].append(float(ent))
                    stats["ppo/approx_kl"].append(float((old - logp).mean())); stats["ppo/clip_frac"].append(float(((ratio - 1).abs() > clip).float().mean()))
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------ calibration / evaluation
    @torch.no_grad()
    def evaluate(self, n_episodes: int, seed_base: int, deterministic: bool, random_actions: bool = False,
                 progress_fn=None) -> Dict[str, float]:
        rets, lens, fails = [], [], []
        g = self.rng.torch("eval_policy")
        snap = self.norm.snapshot(self.device)
        rs = np.random.default_rng(derive_seed(seed_base, "rand"))
        for k in range(n_episodes):
            obs = self.eval_pool.reset_slot(0, seed=seed_base + k)
            ret, t, term = 0.0, 0, False
            while True:
                if random_actions:
                    a = rs.uniform(-1, 1, size=(self.spec.action_dim,)).astype(np.float32)
                else:
                    x = snap.normalize(torch.as_tensor(obs[None], dtype=torch.float32, device=self.device))
                    _, a_t, _ = self.policy.sample(x, g, deterministic)
                    a = a_t[0].cpu().numpy()
                out = self.eval_pool.step_slot(0, self.spec.action.to_physical(a).astype(np.float32))
                obs, ret, t = out.next_obs, ret + out.reward, t + 1
                if progress_fn:
                    progress_fn("ppo_calibration", k, n_episodes, unit="episodes", message=f"episode_step={t}; deterministic={deterministic}")
                if out.terminated or out.truncated:
                    term = out.terminated
                    break
            rets.append(ret), lens.append(t), fails.append(float(term))
            if progress_fn:
                progress_fn("ppo_calibration", k + 1, n_episodes, unit="episodes")
        return {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)), "length_mean": float(np.mean(lens)),
                "failure_rate": float(np.mean(fails)), "episodes": n_episodes}

    def export(self, path: Path, calibration: Dict[str, Any], budget: Dict[str, Any], extra: Optional[dict] = None) -> Path:
        s = self.spec
        snap = self.norm.snapshot("cpu")
        return save_prior_artifact(
            path, weights={"policy": self.policy.state_dict(), "value": self.value.state_dict()},
            normalizer={"mean": snap.mean.numpy(), "std": snap.std.numpy(), "clip": np.asarray(snap.clip)},
            config={"hidden_dims": list(self.p.get("hidden", [64, 64])), "activation": "tanh", "algorithm": "ppo"},
            env_id=s.env_id, obs_dim=s.obs_dim, action_dim=s.action_dim, action_low=s.action.low, action_high=s.action.high,
            calibration=calibration, budget=budget, prior_seed=self.seed, extra=extra)

    # ------------------------------------------------------------------ driver
    def run(self, total_steps: int, *, log_fn: Optional[Callable[[MetricRecord], None]] = None, eval_interval: int = 0,
            eval_episodes: int = 5, eval_seed_base: int = 10_000, stage_steps: Optional[List[int]] = None,
            on_stage: Optional[Callable[[int, "PPOTrainer"], None]] = None, progress_fn=None) -> None:
        stage_steps = sorted(stage_steps or [])
        next_eval = eval_interval if eval_interval else None
        t0 = time.time()
        rollout_size = int(self.p.get("rollout_steps", 2048)) * self.n
        actual_total = ((total_steps + rollout_size - 1) // rollout_size) * rollout_size
        while self.env_steps < total_steps:
            last_v = self.collect(progress_fn, actual_total)
            if progress_fn:
                progress_fn("ppo_update", self.env_steps, actual_total)
            st = self.update(last_v, min(self.env_steps / total_steps, 1.0))
            scal = dict(st)
            if self.recent:
                scal["charts/episode_return"] = float(np.mean([e["ret"] for e in self.recent]))
                scal["charts/episode_length"] = float(np.mean([e["len"] for e in self.recent]))
                scal["charts/terminated_frac"] = float(np.mean([e["term"] for e in self.recent]))
                self.recent.clear()
            scal["charts/env_steps"] = float(self.env_steps)
            scal["charts/steps_per_second"] = self.env_steps / max(time.time() - t0, 1e-9)
            if next_eval and self.env_steps >= next_eval:
                ev = self.evaluate(eval_episodes, eval_seed_base, deterministic=False, progress_fn=progress_fn)
                scal.update({f"eval/{k}": v for k, v in ev.items()})
                while next_eval <= self.env_steps:
                    next_eval += eval_interval
            if log_fn:
                log_fn(MetricRecord(self.env_steps, scal, wall_time=time.time() - t0))
            while stage_steps and self.env_steps >= stage_steps[0]:
                s = stage_steps.pop(0)
                if on_stage:
                    on_stage(s, self)

    def close(self) -> None:
        self.pool.close()
        self.eval_pool.close()


def assign_tiers(stage_calib: Dict[str, Dict[str, float]], random_return: float) -> Dict[str, str]:
    """good / medium / poor from INDEPENDENT calibration returns, normalized between the random-action return and the best stage.
    The PGOFP test return is never used to choose a prior."""
    best = max(c["return_mean"] for c in stage_calib.values())
    span = max(best - random_return, 1e-8)
    out = {}
    for name, c in stage_calib.items():
        score = (c["return_mean"] - random_return) / span
        out[name] = "good" if score >= 0.8 else ("medium" if score >= 0.4 else "poor")
    return out
