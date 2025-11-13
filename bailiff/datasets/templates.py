"""Case templates and cue catalogs used for experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

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


def load_case_templates(root: Path) -> List[CaseTemplate]:
    """Enumerate and validate case YAML files under a directory."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Case root does not exist: {root}")
    cases: List[CaseTemplate] = []
    for path in sorted(root.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _validate_case_payload(data, path)
        cases.append(
            CaseTemplate(
                identifier=str(data["identifier"]),
                description=str(data.get("summary", "")),
                template_path=path,
            )
        )
    if not cases:
        raise FileNotFoundError(f"No case YAML files found under {root}")
    return cases


_REQUIRED_CASE_KEYS = ("identifier", "summary", "charges", "facts", "witnesses", "cue_slots")


def _validate_case_payload(data: Mapping[str, Any], path: Path) -> None:
    missing = [key for key in _REQUIRED_CASE_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if not isinstance(data["charges"], list) or not data["charges"]:
        raise ValueError(f"{path}: 'charges' must be a non-empty list")
    if not isinstance(data["facts"], list) or not data["facts"]:
        raise ValueError(f"{path}: 'facts' must be a non-empty list")
    witnesses = data["witnesses"]
    if not isinstance(witnesses, Mapping):
        raise ValueError(f"{path}: 'witnesses' must be a mapping with prosecution/defense lists")
    cues = data["cue_slots"]
    if not isinstance(cues, Mapping) or not cues:
        raise ValueError(f"{path}: 'cue_slots' must be a non-empty mapping")
    missing_cue_tokens = [slot for slot, value in cues.items() if "{{ cue_value }}" not in str(value)]
    if missing_cue_tokens:
        raise ValueError(f"{path}: cue slots {missing_cue_tokens} must include '{{{{ cue_value }}}}' placeholder")
