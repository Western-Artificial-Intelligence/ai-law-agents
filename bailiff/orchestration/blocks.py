"""Helpers for building randomization blocks and resolving placebos."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from bailiff.core.config import CueToggle
from bailiff.datasets.templates import placebo_catalog
from bailiff.orchestration.randomization import RandomizationBlock


def resolve_placebos(keys: Sequence[str]) -> List[CueToggle]:
    """Return CueToggle objects for requested placebo names."""

    if not keys:
        return []
    lookup = {cue.name: cue for cue in placebo_catalog()}
    toggles: List[CueToggle] = []
    for key in keys:
        toggle = lookup.get(key)
        if toggle is None:
            raise KeyError(f"Unknown placebo cue key: {key}")
        toggles.append(toggle)
    return toggles


def build_blocks(
    case_identifier: str,
    model_identifier: str,
    cues: Iterable[CueToggle],
    seeds: Sequence[int],
    placebo_names: Sequence[str],
) -> List[RandomizationBlock]:
    """Build RandomizationBlock definitions for a cue set."""

    seed_list = list(seeds)
    placebo_lookup = set(placebo_names)
    blocks: List[RandomizationBlock] = []
    for cue in cues:
        blocks.append(
            RandomizationBlock(
                case_identifier=case_identifier,
                model_identifier=model_identifier,
                cue_name=cue.name,
                values=[cue.control_value, cue.treatment_value],
                seeds=seed_list,
                is_placebo=cue.name in placebo_lookup,
            )
        )
    return blocks
