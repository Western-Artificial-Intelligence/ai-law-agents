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
