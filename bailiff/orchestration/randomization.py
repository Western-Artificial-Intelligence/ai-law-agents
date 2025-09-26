"""Randomization utilities for cue assignments."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Iterator, Sequence


@dataclass(slots=True)
class PairAssignment:
    """Represents a paired cue assignment for control/treatment."""

    seed: int
    control_value: str
    treatment_value: str


def blocked_permutations(values: Sequence[str], seeds: Iterable[int]) -> Iterator[PairAssignment]:
    """Yield pair assignments by shuffling within seeds."""

    for seed in seeds:
        rng = Random(seed)
        shuffled = list(values)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError("At least two values required for blocked permutations.")
        yield PairAssignment(seed=seed, control_value=shuffled[0], treatment_value=shuffled[1])
