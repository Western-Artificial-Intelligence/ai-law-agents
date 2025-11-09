"""Configuration primitives for trial simulations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


class Role(str, Enum):
    """Canonical roles supported in the simulator."""

    JUDGE = "judge"
    PROSECUTION = "prosecution"
    DEFENSE = "defense"


class Phase(str, Enum):
    """High-level procedural phases imposed by the state machine."""

    OPENING = "opening"
    DIRECT = "direct"
    CROSS = "cross"
    REDIRECT = "redirect"
    CLOSING = "closing"
    VERDICT = "verdict"
    AUDIT = "audit"


@dataclass(slots=True)
class AgentBudget:
    """Byte/token accounting for a specific agent-role combination."""

    max_bytes: int
    max_tokens: Optional[int] = None
    max_turns: Optional[int] = None


@dataclass(slots=True)
class PhaseBudget:
    """Constraints enforced per procedural phase."""

    phase: Phase
    max_messages: int = 2
    allow_interruptions: bool = False


@dataclass(slots=True)
class CueToggle:
    """Represents a demographic or sociolinguistic cue that can be switched."""

    name: str
    control_value: str
    treatment_value: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TrialConfig:
    """Complete configuration bundle for a simulated trial."""

    case_template: Path
    cue: CueToggle
    model_identifier: str
    seed: int
    agent_budgets: Mapping[Role, AgentBudget]
    phase_budgets: Sequence[PhaseBudget]
    negative_controls: Sequence[CueToggle] = field(default_factory=tuple)
    # Active cue assignment details (set by orchestration when pairing)
    cue_condition: Optional[str] = None  # "control" | "treatment"
    cue_value: Optional[str] = None
    # Policy toggles
    judge_blinding: bool = False
    notes: Optional[str] = None

    def budget_for(self, role: Role) -> AgentBudget:
        """Return the configured budget for a role."""

        return self.agent_budgets[role]

    def phase_budget_for(self, phase: Phase) -> PhaseBudget:
        """Return the PhaseBudget entry for a phase (defaults if missing)."""

        for pb in self.phase_budgets:
            if pb.phase == phase:
                return pb
        return PhaseBudget(phase=phase)


DEFAULT_PHASE_ORDER: List[Phase] = [
    Phase.OPENING,
    Phase.DIRECT,
    Phase.CROSS,
    Phase.REDIRECT,
    Phase.CLOSING,
    Phase.VERDICT,
    Phase.AUDIT,
]
