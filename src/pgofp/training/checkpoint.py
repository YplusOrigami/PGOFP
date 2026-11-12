"""Atomic checkpoints with checksums, compatibility fingerprints and resume modes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from ..core.exceptions import ResumeIncompatibleError
from ..core.specs import SCHEMA_VERSION, stable_hash
from ..utils.provenance import file_sha256

COMPAT_KEYS = ("env", "network", "condition", "prior", "replay", "algorithm.name", "algorithm.actor_objective",
               "algorithm.K_local", "algorithm.zt_proposal", "algorithm.source_scope", "algorithm.use_entropy",
               "source.kind", "source.schedule", "source.total_steps", "source.sigma_base", "optimizer")


def compat_fingerprint(cfg, extra: Optional[Dict[str, Any]] = None) -> str:
    d = OmegaConf.to_container(cfg, resolve=True)
    sel = {}
    for k in COMPAT_KEYS:
        cur: Any = d
        for part in k.split("."):
            cur = cur[part]
        sel[k] = cur
    sel["extra"] = extra or {}
    return stable_hash(sel)


class CheckpointManager:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.pt"

    def save(self, name: str, state: Dict[str, Any], replay=None) -> Path:
        """tmp file -> fsync -> rename; checksum sidecar. A failed save never corrupts the previous file."""
        final = self.path(name)
        tmp = final.with_suffix(".pt.tmp")
        state = dict(state)
        state["schema_version"] = SCHEMA_VERSION
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        digest = file_sha256(tmp)
        replay_meta = None
        if replay is not None:
            rdir = self.dir / "replay"
            replay.save(rdir / "replay.npz")
            replay_meta = {"file": "replay/replay.npz", "size": replay.size, "sha256": file_sha256(rdir / "replay.npz")}
        tmp.replace(final)
        (self.dir / f"{name}.json").write_text(json.dumps({"sha256": digest, "replay": replay_meta,
                                                           "env_transitions": state.get("counters", {}).get("env_transitions")}))
        return final

    def load(self, path: str | Path, verify: bool = True) -> Dict[str, Any]:
        path = Path(path)
        meta_path = path.with_suffix(".json")
        if verify and meta_path.is_file():
            want = json.loads(meta_path.read_text())["sha256"]
            if file_sha256(path) != want:
                raise ResumeIncompatibleError(f"checkpoint {path} failed its checksum")
        return torch.load(path, map_location="cpu", weights_only=False)

    def replay_path(self, name: str = "last") -> Optional[Path]:
        meta = self.dir / f"{name}.json"
        if meta.is_file():
            r = json.loads(meta.read_text()).get("replay")
            if r:
                p = self.dir / r["file"]
                if p.is_file() and file_sha256(p) == r["sha256"]:
                    return p
        return None


def check_resume_compat(state: Dict[str, Any], cfg, extra: Optional[Dict[str, Any]], mode: str) -> None:
    if mode == "weights_only":
        return
    want = state.get("compat")
    got = compat_fingerprint(cfg, extra)
    if want != got:
        raise ResumeIncompatibleError(
            "checkpoint is incompatible with the current configuration (task / action chart / prior artifact / network / "
            "condition / source schema changed). Use resume_mode=weights_only for a new run with new budget.")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ResumeIncompatibleError("checkpoint schema version mismatch")
