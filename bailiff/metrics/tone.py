"""Lightweight tone scoring, calibration, and reliability utilities.

This module provides minimal, dependency-light helpers:
- naive_lexicon_score: simple polarity score in [-1, 1]
- fit_platt / apply_platt: logistic calibration of scores
- cohen_kappa: inter-rater reliability
- expected_calibration_error: ECE across bins
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import math


_POS = {"good", "polite", "respect", "calm", "measured", "professional"}
_NEG = {"bad", "rude", "hostile", "aggressive", "angry", "unprofessional"}


def naive_lexicon_score(text: str) -> float:
    tokens = {t.strip(".,:;!?()[]{}\"\'").lower() for t in text.split()}
    pos = len(tokens & _POS)
    neg = len(tokens & _NEG)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


@dataclass
class PlattParams:
    a: float
    b: float


def fit_platt(scores: Iterable[float], labels: Iterable[int]) -> PlattParams:
    # Simple logistic regression with two parameters via Newton steps
    import numpy as np

    x = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    a, b = 0.0, 0.0
    for _ in range(25):
        z = a * x + b
        p = 1 / (1 + np.exp(-z))
        # gradient
        g_a = np.sum((p - y) * x)
        g_b = np.sum(p - y)
        # Hessian
        w = p * (1 - p)
        h_aa = np.sum(w * x * x) + 1e-8
        h_bb = np.sum(w) + 1e-8
        h_ab = np.sum(w * x)
        # Newton step on 2x2
        det = h_aa * h_bb - h_ab * h_ab + 1e-12
        da = (-g_a * h_bb + g_b * h_ab) / det
        db = (-g_b * h_aa + g_a * h_ab) / det
        a += da
        b += db
        if abs(da) + abs(db) < 1e-6:
            break
    return PlattParams(a, b)


def apply_platt(scores: Iterable[float], params: PlattParams) -> Tuple[float, ...]:
    import numpy as np

    x = np.asarray(list(scores), dtype=float)
    z = params.a * x + params.b
    p = 1 / (1 + np.exp(-z))
    return tuple(float(v) for v in p)


def cohen_kappa(a: Iterable[int], b: Iterable[int]) -> float:
    import numpy as np

    a = np.asarray(list(a), dtype=int)
    b = np.asarray(list(b), dtype=int)
    assert a.size == b.size and a.size > 0
    pa = float(np.mean(a == b))
    p_yes = (np.mean(a == 1) * np.mean(b == 1)) + (np.mean(a == 0) * np.mean(b == 0))
    denom = 1 - p_yes
    return 0.0 if denom == 0 else (pa - p_yes) / denom


def expected_calibration_error(probs: Iterable[float], labels: Iterable[int], bins: int = 10) -> float:
    import numpy as np

    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    assert p.size == y.size and p.size > 0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        w = float(np.mean(mask))
        ece += w * abs(conf - acc)
    return ece

