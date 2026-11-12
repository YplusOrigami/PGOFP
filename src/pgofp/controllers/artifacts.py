"""PriorArtifact: versioned, hash-checked storage for frozen priors (PPO weights, controller configs).

  weights.pt                 PPO actor (+ value head optional)
  manifest.json              schema/version/checksums/env/action chart/distribution semantics
  normalizer.npz             FROZEN observation statistics used by the PPO prior (never replaced by PGOFP's own)
  controller_config.yaml     network structure, inputs
  calibration_metrics.json   independent calibration return/failure + quality tier basis
  budget.json                pretrain env steps, seed, gpu hours
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from ..core.exceptions import PriorUnavailableError
from ..utils.provenance import dir_sha256, file_sha256, utc_now

SCHEMA = "pgofp.prior_artifact.v1"
REQUIRED = ("weights.pt", "manifest.json", "normalizer.npz", "controller_config.yaml", "calibration_metrics.json", "budget.json")


@dataclass
class PriorArtifact:
    path: Path
    manifest: Dict[str, Any]
    config: Dict[str, Any]
    normalizer: Dict[str, np.ndarray]
    weights: Dict[str, Any]
    calibration: Dict[str, Any]
    budget: Dict[str, Any]

    @property
    def content_hash(self) -> str:
        return self.manifest["content_hash"]


def save_prior_artifact(path: str | Path, *, weights: Dict[str, Any], normalizer: Dict[str, np.ndarray], config: Dict[str, Any],
                        env_id: str, obs_dim: int, action_dim: int, action_low, action_high, distribution: str = "tanh_gaussian",
                        calibration: Optional[Dict] = None, budget: Optional[Dict] = None, prior_seed: int = 0,
                        extra: Optional[Dict] = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(weights, path / "weights.pt")
    np.savez(path / "normalizer.npz", **normalizer)
    (path / "controller_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (path / "calibration_metrics.json").write_text(json.dumps(calibration or {}, indent=2))
    (path / "budget.json").write_text(json.dumps(budget or {}, indent=2))
    manifest = {
        "schema": SCHEMA, "created_utc": utc_now(), "env_id": env_id, "obs_dim": obs_dim, "action_dim": action_dim,
        "action_low": list(map(float, action_low)), "action_high": list(map(float, action_high)),
        "distribution": distribution,  # tanh_gaussian: pre-tanh (mu, log_std) are exported as-is
        "prior_seed": prior_seed, "stateful": False, **(extra or {}),
    }
    manifest["content_hash"] = dir_sha256_excluding_manifest(path)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def dir_sha256_excluding_manifest(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in sorted(REQUIRED):
        if name == "manifest.json":
            continue
        f = path / name
        if f.is_file():
            h.update(name.encode())
            h.update(file_sha256(f).encode())
    return h.hexdigest()


def load_prior_artifact(path: str | Path, *, expect_env: Optional[str] = None, expect_obs_dim: Optional[int] = None,
                        expect_action_dim: Optional[int] = None, expect_hash: Optional[str] = None) -> PriorArtifact:
    path = Path(path)
    for name in REQUIRED:
        if not (path / name).is_file():
            raise PriorUnavailableError(f"prior artifact {path} is missing {name}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise PriorUnavailableError(f"unsupported prior artifact schema {manifest.get('schema')!r}")
    actual = dir_sha256_excluding_manifest(path)
    if actual != manifest["content_hash"]:
        raise PriorUnavailableError("prior artifact content hash mismatch (files modified after export)")
    if expect_hash and expect_hash != actual:
        raise PriorUnavailableError(f"prior artifact hash {actual} != pinned {expect_hash}")
    if manifest.get("stateful"):
        raise PriorUnavailableError("stateful / recurrent PPO checkpoints are rejected in v1 (needs a hidden-state history protocol)")
    if expect_env and manifest["env_id"] != expect_env:
        raise PriorUnavailableError(f"prior was trained on {manifest['env_id']} not {expect_env}")
    if expect_obs_dim is not None and manifest["obs_dim"] != expect_obs_dim:
        raise PriorUnavailableError(f"prior obs_dim {manifest['obs_dim']} != env obs_dim {expect_obs_dim}")
    if expect_action_dim is not None and manifest["action_dim"] != expect_action_dim:
        raise PriorUnavailableError(f"prior action_dim {manifest['action_dim']} != env action_dim {expect_action_dim}")
    z = np.load(path / "normalizer.npz")
    return PriorArtifact(
        path=path, manifest=manifest, config=yaml.safe_load((path / "controller_config.yaml").read_text()),
        normalizer={k: z[k] for k in z.files}, weights=torch.load(path / "weights.pt", map_location="cpu", weights_only=False),
        calibration=json.loads((path / "calibration_metrics.json").read_text()),
        budget=json.loads((path / "budget.json").read_text()),
    )
