"""Case templates and cue catalogs used for experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from bailiff.core.config import CueToggle


@dataclass(slots=True)
class CaseTemplate:
    """Represents a case archetype with placeholders for cue insertion."""

    identifier: str
    description: str
    template_path: Path


def default_cases(root: Path) -> List[CaseTemplate]:
    """Load the baseline set of six case archetypes."""

    return [
        CaseTemplate("traffic", "Routine traffic violation involving alleged red light run.", root / "traffic.yaml"),
        CaseTemplate("simple_assault", "Altercation outside a bar causing minor injuries.", root / "simple_assault.yaml"),
        CaseTemplate("shoplifting", "Retail theft with eyewitness testimony.", root / "shoplifting.yaml"),
        CaseTemplate("dui", "Driving under influence stop with breathalyzer.", root / "dui.yaml"),
        CaseTemplate("vandalism", "Graffiti incident captured on CCTV.", root / "vandalism.yaml"),
        CaseTemplate("petty_theft", "Alleged bicycle theft in public park.", root / "petty_theft.yaml"),
    ]


def cue_catalog() -> Dict[str, CueToggle]:
    """Return a dictionary of canonical cue toggles."""

    return {
        "name_ethnicity": CueToggle(
            name="name_ethnicity",
            control_value="Alex Johnson",
            treatment_value="DeShawn Jackson",
            metadata={"type": "name", "source": "Bertrand-Mullainathan"},
        ),
        "dialect": CueToggle(
            name="dialect",
            control_value="Standard American English phrasing in testimony",
            treatment_value="African American English phrasing in testimony",
            metadata={"type": "dialect", "reference": "Labov"},
        ),
    }


def placebo_catalog() -> Iterable[CueToggle]:
    """Generate placebo toggles expected to have null effects."""

    yield CueToggle(
        name="name_placebo",
        control_value="Alex Johnson",
        treatment_value="Jordan Taylor",
        metadata={"type": "name", "class": "neutral"},
    )

