"""Outcome metrics and estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class PairedOutcome:
    """Represents a matched pair of trials with opposing cue assignments."""

    control: int
    treatment: int

    def flip(self) -> bool:
        """Return True if the pair results in divergent verdicts."""

        return self.control != self.treatment


def mcnemar_log_odds(pairs: Iterable[PairedOutcome]) -> Tuple[float, float]:
    """Compute McNemar log-odds estimate and standard error."""

    n01 = sum(1 for p in pairs if p.control == 0 and p.treatment == 1)
    n10 = sum(1 for p in pairs if p.control == 1 and p.treatment == 0)
    if n01 == 0 or n10 == 0:
        raise ValueError("Insufficient discordant pairs for McNemar estimate.")
    estimate = np.log(n01 / n10)
    se = np.sqrt(1 / n01 + 1 / n10)
    return estimate, se


def flip_rate(pairs: Iterable[PairedOutcome]) -> float:
    """Compute the share of outcome flips across pairs."""

    flips = [p.flip() for p in pairs]
    if not flips:
        return float("nan")
    return float(np.mean(flips))


def normalized_sentence(imposed: float, minimum: float, maximum: float) -> float:
    """Normalize a sentence severity value onto 0-1 scale."""

    if maximum == minimum:
        raise ValueError("Maximum and minimum sentence bounds must differ.")
    return (imposed - minimum) / (maximum - minimum)


def summarize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Produce aggregate conviction rates by cue assignment."""

    required = {"cue", "verdict"}
    if missing := required.difference(df.columns):
        raise KeyError(f"Missing columns for outcome summary: {sorted(missing)}")
    summary = (
        df.groupby("cue")
        .agg(conviction_rate=("verdict", "mean"), trials=("verdict", "size"))
        .reset_index()
    )
    return summary
