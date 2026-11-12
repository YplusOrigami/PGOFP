"""Policy bundle export + minimal inference loader. A bundle is self-contained: flow weights alone are NOT a deployable
control-prior method, so the prior (controller config / PPO artifact), normalizer, source spec and condition schema travel with it."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from ..core.specs import SourceSpec
from ..core.types import ContextBatch
from ..utils.provenance import collect_provenance, file_sha256, utc_now


def export_bundle(runner, out_dir: str | Path, smoke: bool = True, latency: Optional[Dict[str, Any]] = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = runner.cfg
    spec = runner.source_sched.spec_at(None) if runner.source_sched._last is not None else runner.source_sched.spec_at(runner.env_transitions)
    torch.save({"actor": runner.agent.actor.state_dict(), "action_dim": runner.env_spec.action_dim, "cond_dim": runner.builder.dim,
                "hidden": list(cfg.network.actor_hidden), "time_embed_dim": cfg.network.time_embed_dim,
                "activation": cfg.network.actor_activation, "layer_norm": cfg.network.layer_norm,
                "time_max_freq": cfg.network.time_max_freq}, out / "actor.pt")
    snap = runner.norm.snapshot("cpu")
    np.savez(out / "normalizer.npz", mean=snap.mean.numpy(), std=snap.std.numpy(), clip=np.asarray(snap.clip))
    (out / "source_spec.json").write_text(json.dumps(spec.__dict__, indent=2, default=list))
    policy_spec = {
        "env_id": runner.env_spec.env_id, "obs_dim": runner.env_spec.obs_dim, "action_dim": runner.env_spec.action_dim,
        "action_low": list(runner.env_spec.action.low), "action_high": list(runner.env_spec.action.high),
        "action_chart": "latent z -> a_norm=tanh(z) -> a_phys=bias+scale*a_norm",
        "condition": OmegaConf.to_container(cfg.condition), "cond_dim": runner.builder.dim,
        "context": {"ref_dim": runner.ctx_spec.ref_dim, "ctrl_dim": runner.ctx_spec.ctrl_dim, "phase_dim": runner.ctx_spec.phase_dim,
                    "n_prior": runner.ctx_spec.n_prior},
        "flow_inference": "z_out = eps - u(eps, r=0, t=1; c), NFE=1", "algorithm": cfg.algorithm.name,
    }
    (out / "policy_spec.json").write_text(json.dumps(policy_spec, indent=2))
    # prior: controller config (+ PPO artifact copied verbatim) -- required for the control-prior method to run alone
    prior_dir = out / "prior"
    prior_dir.mkdir(exist_ok=True)
    (prior_dir / "prior_config.yaml").write_text(OmegaConf.to_yaml(cfg.prior))
    if cfg.prior.type == "ppo" and cfg.prior.artifact_path:
        shutil.copytree(cfg.prior.artifact_path, prior_dir / "ppo_artifact", dirs_exist_ok=True)
    elif cfg.prior.type == "multi":
        for i, c in enumerate(cfg.prior.components):
            if c.get("artifact_path"):
                shutil.copytree(c["artifact_path"], prior_dir / f"ppo_artifact_{i}", dirs_exist_ok=True)
    (out / "deployment_config.yaml").write_text(yaml.safe_dump({
        "control_period_s": runner.env_spec.control_dt, "device": "cpu", "torch_threads": 1,
        "reference_interface": "pass current reference window per step", "safety_filter": "none",
        "note": "Critic and the training-time density estimator are NOT exported (K=1 default deployment).",
    }))
    manifest = {"created_utc": utc_now(), "files": {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
                "provenance": collect_provenance(), "env_transitions": runner.env_transitions,
                "controller_versions": runner.manager.versions, "latency": latency, "smoke": None}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if smoke:
        manifest["smoke"] = smoke_test_bundle(runner, out)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def smoke_test_bundle(runner, out: Path) -> Dict[str, Any]:
    """The exported actor must reproduce the in-memory agent bit-for-bit on a fixed (condition, eps)."""
    from ..networks.mean_flow import MeanFlowNet

    d = torch.load(out / "actor.pt", map_location="cpu")
    net = MeanFlowNet(d["action_dim"], d["cond_dim"], tuple(d["hidden"]), d["time_embed_dim"], d["activation"], d["layer_norm"], d["time_max_freq"])
    net.load_state_dict(d["actor"])
    g = torch.Generator().manual_seed(0)
    cond = torch.randn(4, d["cond_dim"], generator=g)
    eps = torch.randn(4, d["action_dim"], generator=g)
    with torch.no_grad():
        a = net.one_step(eps, cond)
        b = runner.agent.actor.to("cpu").one_step(eps, cond)
    runner.agent.actor.to(runner.device)
    return {"max_abs_diff_vs_in_memory": float((a - b).abs().max()), "ok": bool(torch.allclose(a, b, atol=1e-6))}


class PolicyBundle:
    """Minimal inference object: bundle dir -> act(ctx_dict) with the exported source spec / normalizer / actor."""

    def __init__(self, path: str | Path):
        from ..networks.mean_flow import MeanFlowNet
        from ..priors.mixture import build_source
        from ..utils.normalization import NormSnapshot

        p = Path(path)
        d = torch.load(p / "actor.pt", map_location="cpu")
        self.net = MeanFlowNet(d["action_dim"], d["cond_dim"], tuple(d["hidden"]), d["time_embed_dim"], d["activation"], d["layer_norm"], d["time_max_freq"])
        self.net.load_state_dict(d["actor"])
        self.net.eval()
        z = np.load(p / "normalizer.npz")
        self.norm = NormSnapshot(torch.as_tensor(z["mean"]), torch.as_tensor(z["std"]), float(z["clip"]), 0)
        sd = json.loads((p / "source_spec.json").read_text())
        sd["component_weights"] = tuple(sd["component_weights"])
        self.spec = SourceSpec(**sd)
        self.policy_spec = json.loads((p / "policy_spec.json").read_text())
        self._build_source = build_source

    @torch.no_grad()
    def act_from_condition(self, builder, ctx: ContextBatch, generator=None) -> torch.Tensor:
        cond = builder.build(ctx, self.norm, self.spec)
        eps = self._build_source(self.spec, ctx, self.policy_spec["action_dim"]).sample((), generator)
        return torch.tanh(self.net.one_step(eps, cond))
