"""Inference latency: prior query / source sample / flow forward / post-processing / end-to-end (batch=1, p50/p95/p99).

NFE=1 only describes the flow network; a frozen PPO or an online MPC solve is charged separately. Cached-prior timings are
reported SEPARATELY from uncached ones.  CUDA is synchronized before every clock read; CPU thread count is recorded.
"""
from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np
import torch

from ..utils.timing import sync


def _stats(ts: np.ndarray, deadline_s: float) -> Dict[str, float]:
    ms = ts * 1e3
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
            "max_ms": float(ms.max()), "mean_ms": float(ms.mean()), "deadline_miss_rate": float((ts > deadline_s).mean()), "n": int(ts.size)}


def time_callable(fn: Callable[[], object], device="cpu", warmup: int = 50, iters: int = 2000, deadline_s: float = 0.010) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    sync(device)
    ts = np.empty(iters)
    for i in range(iters):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts[i] = time.perf_counter() - t0
    return _stats(ts, deadline_s)


@torch.no_grad()
def benchmark_policy(agent, ctx_np: Dict[str, np.ndarray], spec, norm, device, controller_query: Callable[[], object] | None = None,
                     iters: int = 2000, deadline_s: float = 0.010) -> Dict[str, Dict[str, float]]:
    """ctx_np: a single-condition (batch=1) context dict. Returns stage -> stats."""
    from ..core.types import ContextBatch
    from ..priors.mixture import build_source

    ctx = ContextBatch.from_numpy(ctx_np, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    out: Dict[str, Dict[str, float]] = {}
    if controller_query is not None:
        out["prior_query"] = time_callable(controller_query, device, iters=min(iters, 500), deadline_s=deadline_s)
    cond = agent.builder.build(ctx, norm, spec)
    src = build_source(spec, ctx, agent.d)
    out["condition_build"] = time_callable(lambda: agent.builder.build(ctx, norm, spec), device, iters=iters, deadline_s=deadline_s)
    out["source_sample"] = time_callable(lambda: src.sample((), gen), device, iters=iters, deadline_s=deadline_s)
    eps = src.sample((), gen)
    out["flow_forward_nfe1"] = time_callable(lambda: agent.actor.one_step(eps, cond), device, iters=iters, deadline_s=deadline_s)
    out["policy_act_cached_prior"] = time_callable(lambda: agent.act(ctx, spec, norm, gen), device, iters=iters, deadline_s=deadline_s)
    if controller_query is not None:
        def e2e():
            controller_query()
            agent.act(ctx, spec, norm, gen)

        out["end_to_end_uncached_prior"] = time_callable(e2e, device, iters=min(iters, 500), deadline_s=deadline_s)
    out["meta"] = {"device": str(device), "torch_threads": torch.get_num_threads(), "batch": 1}  # type: ignore[assignment]
    return out
