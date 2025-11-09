"""Structured logging schema used during simulations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from .config import Phase, Role


class ObjectionRuling(str, Enum):
    SUSTAINED = "sustained"
    OVERRULED = "overruled"


@dataclass(slots=True)
class EventTag:
    """Lightweight tag for procedural events (objections, safety, etc.)."""

    name: str
    value: Optional[str] = None


@dataclass(slots=True)
class UtteranceLog:
    """Atomic log record for a single agent utterance."""

    role: Role
    phase: Phase
    content: str
    byte_count: int
    token_count: Optional[int]
    addressed_to: Optional[Role]
    timestamp: datetime
    interruption: bool = False
    objection_raised: bool = False
    objection_ruling: Optional[ObjectionRuling] = None
    safety_triggered: bool = False
    tags: List[EventTag] = field(default_factory=list)


@dataclass(slots=True)
class TrialLog:
    """Container aggregating utterance-level records and metadata."""

    trial_id: str
    case_identifier: str
    model_identifier: str
    cue_name: str
    cue_condition: Optional[str]
    cue_value: Optional[str]
    seed: int
    started_at: datetime
    completed_at: Optional[datetime]
    utterances: List[UtteranceLog] = field(default_factory=list)
    verdict: Optional[str] = None
    sentence: Optional[str] = None
    schema_version: str = "0.1"

    def append(self, record: UtteranceLog) -> None:
        """Append an utterance record to the log."""

        self.utterances.append(record)
