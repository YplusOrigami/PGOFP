"""Fixed condition banks and action-distribution diagnostics (expressiveness / displacement / diversity)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..core.specs import SourceSpec
from ..core.types import CONTEXT_FIELDS, ContextBatch
from ..priors.mixture import build_source


def condition_bank_from_replay(replay, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """A fixed bank of raw conditions (obs, reference, controller memory, frozen prior params) for method-to-method comparison."""
    idx = rng.choice(len(replay), size=min(n, len(replay)), replace=False)
    return {k: replay.data[k][idx].copy() for k in CONTEXT_FIELDS}


def save_condition_bank(path: str | Path, bank: Dict[str, np.ndarray], meta: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bank, **{f"meta_{k}": np.asarray(v) for k, v in (meta or {}).items()})


@torch.no_grad()
def sample_policy_distribution(agent, bank: Dict[str, np.ndarray], spec: SourceSpec, norm, n_samples: int = 2048,
                               generator=None, device="cpu", chunk: int = 256) -> Dict[str, np.ndarray]:
    """For every condition draw `n_samples` independent single-sample policy actions AND the matching source latents.

    Returns dict with source z [C,S,d], output z [C,S,d] and actions a_norm [C,S,d] (full samples are kept, not just 2-D scatter).
    """
    C = bank["obs_raw"].shape[0]
    d = agent.d
    z_src = np.zeros((C, n_samples, d), np.float32)
    z_out = np.zeros_like(z_src)
    for c in range(C):
        ctx = ContextBatch.from_numpy({k: np.repeat(v[c:c + 1], n_samples, 0) for k, v in bank.items()}, device=device)
        res = agent.act(ctx, spec, norm, generator)
        z_out[c] = res.z.cpu().numpy()
        z_src[c] = res.eps.cpu().numpy()
    return {"z_source": z_src, "z_out": z_out, "a_norm": np.tanh(z_out)}


def summarize_distribution(samples: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Scalar diagnostics per bank (averaged over conditions). Cluster COUNTS are deliberately not reported (parameter-dependent)."""
    a, zs, zo = samples["a_norm"], samples["z_source"], samples["z_out"]
    C, S, d = a.shape
    disp = np.linalg.norm(zo - zs, axis=-1)  # displacement source -> output in latent space
    spread = a.std(1).mean()
    cov_logdet = []
    pcs = []
    bc = []
    for c in range(C):
        cov = np.cov(a[c].T) if d > 1 else np.atleast_2d(a[c].var())
        w = np.linalg.eigvalsh(cov)
        cov_logdet.append(np.log(np.clip(w, 1e-12, None)).sum())
        pcs.append(w[-1] / max(w.sum(), 1e-12))
        # bimodality coefficient along the first principal axis (>0.555 is only a hint of multimodality, not proof)
        _, vecs = np.linalg.eigh(cov)
        proj = a[c] @ vecs[:, -1]
        m, s = proj.mean(), proj.std() + 1e-12
        skew = ((proj - m) ** 3).mean() / s**3
        kurt = ((proj - m) ** 4).mean() / s**4
        bc.append((skew**2 + 1) / max(kurt, 1e-12))
    return {
        "dist/displacement_mean": float(disp.mean()), "dist/action_std_mean": float(spread),
        "dist/action_cov_logdet": float(np.mean(cov_logdet)), "dist/first_pc_fraction": float(np.mean(pcs)),
        "dist/bimodality_coeff_pc1": float(np.mean(bc)), "dist/saturation_frac": float((np.abs(a) > 0.99).mean()),
    }
