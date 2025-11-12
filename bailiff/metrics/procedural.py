"""Procedural metrics including byte share and measurement corrections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class ShareRecord:
    """Byte share contribution for a single phase within a trial."""

    trial_id: str
    phase: str
    pros_bytes: int
    def_bytes: int

    @property
    def total(self) -> int:
        return self.pros_bytes + self.def_bytes

    def delta(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.pros_bytes - self.def_bytes) / self.total


def aggregate_share(records: Iterable[ShareRecord]) -> float:
    """Inverse-variance weighted aggregation of byte share deltas."""

    deltas = []
    weights = []
    for record in records:
        var = max(record.total, 1)
        deltas.append(record.delta())
        weights.append(1 / var)
    if not weights:
        return 0.0
    weights = np.asarray(weights)
    deltas = np.asarray(deltas)
    return float(np.average(deltas, weights=weights))


def correct_measurement(mean_observed: float, alpha: float, beta: float) -> float:
    """Apply classical measurement error correction for binary rates."""

    denom = 1 - alpha - beta
    if denom == 0:
        raise ValueError("Invalid measurement error parameters: denom equals zero.")
    return (mean_observed - alpha) / denom


def estimate_misclassification(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[float, float]:
    """Estimate (alpha, beta) where alpha=false positive rate, beta=false negative rate.

    alpha = P(pred=1|true=0), beta = P(pred=0|true=1).
    """

    import numpy as np

    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    n0 = np.sum(yt == 0)
    n1 = np.sum(yt == 1)
    alpha = float(np.sum((yt == 0) & (yp == 1)) / n0) if n0 > 0 else 0.0
    beta = float(np.sum((yt == 1) & (yp == 0)) / n1) if n1 > 0 else 0.0
    return alpha, beta


def summarize_objections(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize objection outcomes by side and cue."""

    required = {"cue", "side", "sustained"}
    if missing := required.difference(df.columns):
        raise KeyError(f"Missing columns for objection summary: {sorted(missing)}")
    summary = (
        df.groupby(["cue", "side"])  # type: ignore[arg-type]
        .agg(sustain_rate=("sustained", "mean"), objections=("sustained", "size"))
        .reset_index()
    )
    return summary


def tone_gap(df: pd.DataFrame) -> Tuple[float, float]:
    """Compute mean tone difference between treatment and control cues."""

    required = {"cue", "tone"}
    if missing := required.difference(df.columns):
        raise KeyError(f"Missing columns for tone analysis: {sorted(missing)}")
    grouped = df.groupby("cue")["tone"].mean()
    if len(grouped) != 2:
        raise ValueError("Tone gap requires exactly two cue conditions.")
    control, treatment = grouped.iloc[0], grouped.iloc[1]
    return float(control), float(treatment)


@dataclass
class CalibrationResult:
    """Container describing measurement-error adjustments."""

    alpha: float
    beta: float
    corrected_rate: float
    alpha_ci: Tuple[float, float]
    beta_ci: Tuple[float, float]
    corrected_ci: Tuple[float, float]


def measurement_error_calibration(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    observed_rate: float,
    *,
    reps: int = 1000,
    seed: int | None = None,
) -> CalibrationResult:
    """Estimate (alpha, beta) and corrected rate with bootstrap uncertainty."""

    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.size == 0:
        raise ValueError("At least one labeled example is required for calibration.")
    alpha_hat, beta_hat = estimate_misclassification(true, pred)
    corrected = correct_measurement(observed_rate, alpha_hat, beta_hat)
    rng = np.random.default_rng(seed)
    boot_alpha: list[float] = []
    boot_beta: list[float] = []
    boot_corrected: list[float] = []
    n = true.size
    reps = max(reps, 1)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        sample_true = true[idx]
        sample_pred = pred[idx]
        a, b = estimate_misclassification(sample_true, sample_pred)
        boot_alpha.append(a)
        boot_beta.append(b)
        boot_corrected.append(correct_measurement(observed_rate, a, b))

    def _ci(samples: list[float]) -> Tuple[float, float]:
        arr = np.asarray(samples)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return CalibrationResult(
        alpha=alpha_hat,
        beta=beta_hat,
        corrected_rate=corrected,
        alpha_ci=_ci(boot_alpha),
        beta_ci=_ci(boot_beta),
        corrected_ci=_ci(boot_corrected),
    )
