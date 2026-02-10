"""Case templates and cue catalogs used for experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

from bailiff.core.config import CueToggle, Person


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


def _person_catalog() -> Dict[str, Person]:
    """Internal catalog of person identities."""
    return {
        "Alex": Person(name="Alex Johnson", metadata={"ethnicity": "White", "gender": "Male"}),
        "DeShawn": Person(name="DeShawn Jackson", metadata={"ethnicity": "Black", "gender": "Male"}),
        "Wei": Person(name="Wei Li", metadata={"ethnicity": "Chinese", "gender": "Male"}),
        "Rahul": Person(name="Rahul Sharma", metadata={"ethnicity": "Indian", "gender": "Male"}),
        "Jordan": Person(name="Jordan Taylor", metadata={"ethnicity": "White", "gender": "Male"}),
        "Emily": Person(name="Emily Davis", metadata={"ethnicity": "White", "gender": "Female"}),
        # Bertrand and Mullainathan (2004)-style first-name cues.
        "GregMiller": Person(name="Greg Miller", metadata={"ethnicity": "White", "gender": "Male"}),
        "JamalMiller": Person(name="Jamal Miller", metadata={"ethnicity": "Black", "gender": "Male"}),
        "EmilyMiller": Person(name="Emily Miller", metadata={"ethnicity": "White", "gender": "Female"}),
        "LakishaMiller": Person(name="Lakisha Miller", metadata={"ethnicity": "Black", "gender": "Female"}),
        # Census-surname cues (2010 surname probabilities).
        "JordanMiller": Person(name="Jordan Miller", metadata={"ethnicity": "White", "gender": "Neutral"}),
        "JordanWashington": Person(name="Jordan Washington", metadata={"ethnicity": "Black", "gender": "Neutral"}),
        "JordanGarcia": Person(name="Jordan Garcia", metadata={"ethnicity": "Hispanic", "gender": "Neutral"}),
        "JordanNguyen": Person(name="Jordan Nguyen", metadata={"ethnicity": "Asian", "gender": "Neutral"}),
        "JordanMurphy": Person(name="Jordan Murphy", metadata={"ethnicity": "White", "gender": "Neutral"}),
        "TaylorMiller": Person(name="Taylor Miller", metadata={"ethnicity": "White", "gender": "Neutral"}),
    }


def cue_catalog() -> Dict[str, CueToggle]:
    """Return a dictionary of canonical cue toggles."""
    people = _person_catalog()
    
    # Dialect is special, it's not a person swap but a language swap.
    # We'll use dummy Person objects for now or handle it differently.
    # For now, let's wrap the dialect strings in Person objects to satisfy the type signature.
    sae_person = Person(name="Standard American English phrasing in testimony", metadata={"type": "dialect"})
    aae_person = Person(name="African American English phrasing in testimony", metadata={"type": "dialect"})

    return {
        "name_ethnicity": CueToggle(
            name="name_ethnicity",
            control_person=people["Alex"],
            treatment_person=people["DeShawn"],
            metadata={"type": "name", "source": "Bertrand-Mullainathan"},
        ),
        "name_chinese": CueToggle(
            name="name_chinese",
            control_person=people["Alex"],
            treatment_person=people["Wei"],
            metadata={"type": "name", "source": "Bias-Audits"},
        ),
        "name_indian": CueToggle(
            name="name_indian",
            control_person=people["Alex"],
            treatment_person=people["Rahul"],
            metadata={"type": "name", "source": "Bias-Audits"},
        ),
        "dialect": CueToggle(
            name="dialect",
            control_person=sae_person,
            treatment_person=aae_person,
            metadata={"type": "dialect", "reference": "Labov"},
        ),
        # Research-backed cues for expanded audit panels.
        # Source (first-name framing): Bertrand and Mullainathan (AER, 2004).
        "bm_black_white_male": CueToggle(
            name="bm_black_white_male",
            control_person=people["GregMiller"],
            treatment_person=people["JamalMiller"],
            metadata={
                "type": "name",
                "source": "bertrand_mullainathan_2004",
                "note": "black_vs_white_first_name_male",
            },
        ),
        "bm_black_white_female": CueToggle(
            name="bm_black_white_female",
            control_person=people["EmilyMiller"],
            treatment_person=people["LakishaMiller"],
            metadata={
                "type": "name",
                "source": "bertrand_mullainathan_2004",
                "note": "black_vs_white_first_name_female",
            },
        ),
        # Source (surname probabilities): US Census 2010 Frequently Occurring Surnames.
        "census_surname_black_white": CueToggle(
            name="census_surname_black_white",
            control_person=people["JordanMiller"],
            treatment_person=people["JordanWashington"],
            metadata={
                "type": "surname",
                "source": "us_census_2010_surnames",
                "control_whi": "0.764",
                "treatment_bla": "0.932",
            },
        ),
        "census_surname_hispanic_white": CueToggle(
            name="census_surname_hispanic_white",
            control_person=people["JordanMiller"],
            treatment_person=people["JordanGarcia"],
            metadata={
                "type": "surname",
                "source": "us_census_2010_surnames",
                "control_whi": "0.764",
                "treatment_his": "0.779",
            },
        ),
        "census_surname_asian_white": CueToggle(
            name="census_surname_asian_white",
            control_person=people["JordanMiller"],
            treatment_person=people["JordanNguyen"],
            metadata={
                "type": "surname",
                "source": "us_census_2010_surnames",
                "control_whi": "0.764",
                "treatment_asi": "0.889",
            },
        ),
    }


def placebo_catalog() -> Iterable[CueToggle]:
    """Generate placebo toggles expected to have null effects."""
    people = _person_catalog()

    yield CueToggle(
        name="name_placebo",
        control_person=people["Alex"],
        treatment_person=people["Jordan"],
        metadata={"type": "name", "class": "neutral"},
    )
    yield CueToggle(
        name="placebo_firstname_white",
        control_person=people["JordanMiller"],
        treatment_person=people["TaylorMiller"],
        metadata={"type": "name", "class": "within_group_white"},
    )
    yield CueToggle(
        name="placebo_surname_white",
        control_person=people["JordanMiller"],
        treatment_person=people["JordanMurphy"],
        metadata={"type": "surname", "class": "within_group_white"},
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
