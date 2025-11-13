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
    cue_name: str | None = None
    block_key: str | None = None
    case_identifier: str | None = None
    model_identifier: str | None = None
    is_placebo: bool = False


@dataclass(slots=True)
class RandomizationBlock:
    """Definition of a block (case × model × cue) to randomize within."""

    case_identifier: str
    model_identifier: str
    cue_name: str
    values: Sequence[str]
    seeds: Sequence[int]
    is_placebo: bool = False

    @property
    def block_key(self) -> str:
        return block_identifier(self.case_identifier, self.model_identifier)


def block_identifier(case_identifier: str, model_identifier: str) -> str:
    """Return a canonical identifier for a case × model block."""

    return f"{case_identifier}:{model_identifier}"


def blocked_permutations(
    values: Sequence[str],
    seeds: Iterable[int],
    *,
    cue_name: str | None = None,
    block_key: str | None = None,
    case_identifier: str | None = None,
    model_identifier: str | None = None,
    is_placebo: bool = False,
) -> Iterator[PairAssignment]:
    """Yield pair assignments by shuffling within seeds."""

    for seed in seeds:
        rng = Random(seed)
        shuffled = list(values)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError("At least two values required for blocked permutations.")
        yield PairAssignment(
            seed=seed,
            control_value=shuffled[0],
            treatment_value=shuffled[1],
            cue_name=cue_name,
            block_key=block_key,
            case_identifier=case_identifier,
            model_identifier=model_identifier,
            is_placebo=is_placebo,
        )


def blockwise_permutations(blocks: Iterable[RandomizationBlock]) -> Iterator[PairAssignment]:
    """Yield assignments for each block definition."""

    for block in blocks:
        yield from blocked_permutations(
            block.values,
            block.seeds,
            cue_name=block.cue_name,
            block_key=block.block_key,
            case_identifier=block.case_identifier,
            model_identifier=block.model_identifier,
            is_placebo=block.is_placebo,
        )
