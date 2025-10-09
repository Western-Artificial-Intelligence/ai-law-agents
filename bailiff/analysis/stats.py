"""Analysis helpers: BH FDR, TOST, randomization inference, wild bootstrap."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence, Tuple


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[Sequence[bool], Sequence[float]]:
    """Return (rejections, adjusted p-values)."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        j = n - rank
        val = min(prev, pvals[order[j]] * n / (j + 1))
        adj[order[j]] = val
        prev = val
    rejs = [adj[i] <= alpha for i in range(n)]
    return rejs, adj


def tost_log_odds(est: float, se: float, delta: float = 0.1) -> Tuple[bool, float, float]:
    """Two one-sided tests for equivalence on log-odds with margin `delta`.

    Returns (equivalent, p_lower, p_upper).
    """
    from math import erf, sqrt

    def norm_cdf(z: float) -> float:
        return 0.5 * (1 + erf(z / sqrt(2)))

    z1 = (est - (-delta)) / se
    z2 = (delta - est) / se
    p1 = 1 - norm_cdf(z1)
    p2 = 1 - norm_cdf(z2)
    return (p1 < 0.05 and p2 < 0.05), p1, p2


def randomization_inference(stat_fn: Callable[[Sequence[int], Sequence[int]], float],
                            y_control: Sequence[int], y_treat: Sequence[int],
                            reps: int = 1000, seed: int = 123) -> float:
    """Permutation test on paired outcomes, swapping within pairs under null."""
    import random

    obs = stat_fn(y_control, y_treat)
    cnt = 0
    rng = random.Random(seed)
    for _ in range(reps):
        c, t = [], []
        for yc, yt in zip(y_control, y_treat):
            if rng.random() < 0.5:
                c.append(yc); t.append(yt)
            else:
                c.append(yt); t.append(yc)
        val = stat_fn(c, t)
        if abs(val) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (reps + 1)


def wild_cluster_bootstrap(stat_fn: Callable[[Sequence[int], Sequence[int]], float],
                           y_control: Sequence[int], y_treat: Sequence[int],
                           reps: int = 1000, seed: int = 123) -> Tuple[float, float]:
    """Simple wild bootstrap for paired statistic using Rademacher weights.

    Returns (p5, p95) percentiles of the bootstrap distribution.
    """
    import random
    import numpy as np

    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        signs = [1 if rng.random() < 0.5 else -1 for _ in range(len(y_control))]
        c = [yc * s for yc, s in zip(y_control, signs)]
        t = [yt * s for yt, s in zip(y_treat, signs)]
        vals.append(stat_fn(c, t))
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    return lo, hi

