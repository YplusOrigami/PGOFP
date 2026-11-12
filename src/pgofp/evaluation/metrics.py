"""Statistics: seed is the independent unit; raw data is always kept; censored values are never filled with the budget."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over INDEPENDENT units (training seeds). With few seeds the CI is unstable."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def return_auc(steps: Sequence[float], returns: Sequence[float]) -> float:
    """Area under the eval-return curve divided by the step span (mean return over training)."""
    s, r = np.asarray(steps, float), np.asarray(returns, float)
    if s.size < 2:
        return float(r.mean()) if r.size else float("nan")
    return float(np.trapezoid(r, s) / (s[-1] - s[0]))


def steps_to_threshold(steps: Sequence[float], returns: Sequence[float], threshold: float) -> Optional[float]:
    """First eval step with return >= threshold; None = censored (NOT the total budget)."""
    for s, r in zip(steps, returns):
        if r >= threshold:
            return float(s)
    return None


def tail_mean(values: Sequence[float], frac: float = 0.2) -> float:
    v = np.asarray(values, float)
    k = max(1, int(np.ceil(len(v) * frac)))
    return float(v[-k:].mean()) if v.size else float("nan")


def max_drawdown(returns: Sequence[float]) -> float:
    """Largest drop from a running maximum (0 for monotone curves)."""
    r = np.asarray(returns, float)
    if r.size == 0:
        return float("nan")
    return float((np.maximum.accumulate(r) - r).max())


def smooth(values: Sequence[float], window: int) -> np.ndarray:
    """Display-only moving average with a pre-declared window; significance claims use the raw data."""
    v = np.asarray(values, float)
    if window <= 1 or v.size < window:
        return v
    k = np.ones(window) / window
    return np.convolve(v, k, mode="valid")


def per_seed_summary(curves: Dict[int, Tuple[List[float], List[float]]], tail_frac: float = 0.2, threshold: Optional[float] = None) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"final_tail": [], "auc": [], "drawdown": [], "steps_to_threshold": []}
    for seed, (steps, rets) in sorted(curves.items()):
        out["final_tail"].append(tail_mean(rets, tail_frac))
        out["auc"].append(return_auc(steps, rets))
        out["drawdown"].append(max_drawdown(rets))
        t = None if threshold is None else steps_to_threshold(steps, rets, threshold)
        out["steps_to_threshold"].append(float("nan") if t is None else t)
    return out
