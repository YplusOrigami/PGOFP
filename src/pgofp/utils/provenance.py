"""Run provenance: dependency/hardware/hash capture (UTC timestamps)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_sha256(path: str | Path) -> str:
    """Hash of all files (relative name + content) under a directory."""
    h = hashlib.sha256()
    root = Path(path)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("._"):
            h.update(str(p.relative_to(root)).encode())
            h.update(file_sha256(p).encode())
    return h.hexdigest()


def _version(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not-installed"


def git_info(cwd: str | Path = ".") -> Dict[str, Any]:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
        if rev.returncode != 0:
            return {"git": None}
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5)
        return {"git": {"commit": rev.stdout.strip(), "dirty": bool(dirty.stdout.strip())}}
    except Exception:
        return {"git": None}


def collect_provenance() -> Dict[str, Any]:
    import torch

    return {
        "utc_time": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {m: _version(m) for m in ("torch", "numpy", "gymnasium", "mujoco", "omegaconf", "tensorboard", "wandb")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        **git_info(),
    }
