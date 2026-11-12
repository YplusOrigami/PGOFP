"""Frozen tanh-Gaussian PPO prior. Queries only; no optimizer, and it keeps its OWN observation normalizer."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from ..core.exceptions import ConfigError
from ..core.registry import CONTROLLERS
from ..core.specs import CapabilitySpec
from ..core.types import ControllerPrediction
from ..networks.gaussian_policy import TanhGaussianPolicy
from .artifacts import PriorArtifact, load_prior_artifact
from .base import BaseController


class PPOPriorController(BaseController):
    n_components = 1
    ctrl_dim = 0

    def __init__(self, artifact: PriorArtifact, distribution: str = "native"):
        if distribution not in ("native", "center_gaussian"):
            raise ConfigError("prior.distribution must be native|center_gaussian")
        cfg = artifact.config
        self.policy = TanhGaussianPolicy(artifact.manifest["obs_dim"], artifact.manifest["action_dim"],
                                         tuple(cfg["hidden_dims"]), cfg.get("activation", "tanh"))
        self.policy.load_state_dict(artifact.weights["policy"])
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.policy.eval()
        self.mean = artifact.normalizer["mean"].astype(np.float32)
        self.std = artifact.normalizer["std"].astype(np.float32)
        self.clip = float(artifact.normalizer.get("clip", 10.0))
        self.distribution = distribution
        # native => the exported pre-tanh log-std IS the source scale (no extra noise).
        self.capability = CapabilitySpec(deterministic=False, provides_std=(distribution == "native"), stateful=False,
                                         latent_coordinates=True)
        self.version = f"ppo:{artifact.content_hash[:12]}:{distribution}"

    @torch.no_grad()
    def predict(self, obs, ref, state) -> ControllerPrediction:
        t0 = time.perf_counter()
        x = np.clip((np.asarray(obs, np.float32) - self.mean) / self.std, -self.clip, self.clip)
        mu, ls = self.policy.dist_params(torch.as_tensor(x))
        mu, ls = mu.numpy()[:, None, :], ls.numpy()[:, None, :]
        valid = np.isfinite(mu).all(-1)
        return ControllerPrediction(center_latent=mu, log_std_latent=ls if self.distribution == "native" else None,
                                    valid=valid, latency_s=time.perf_counter() - t0, version=self.version)


def build_ppo_controller(cfg: Dict, env_spec, distribution: str = "native", artifact_path: str | None = None, expect_hash=None):
    if not artifact_path:
        raise ConfigError("prior.type=ppo requires prior.artifact_path")
    art = load_prior_artifact(artifact_path, expect_env=env_spec.env_id, expect_obs_dim=env_spec.obs_dim,
                              expect_action_dim=env_spec.action_dim, expect_hash=expect_hash)
    return PPOPriorController(art, distribution)


@CONTROLLERS.register("ppo")
def _build(cfg, env_spec, **kw):
    return build_ppo_controller(cfg, env_spec, kw.get("distribution", "native"), kw.get("artifact_path"))
