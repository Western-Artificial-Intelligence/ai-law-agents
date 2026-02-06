"""Configuration primitives for trial simulations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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


class PolicyViolation(str, Enum):
    """Types of policy violations that can be detected and enforced.
    
    NEW: Added to track different types of policy violations during trial execution.
    Violations are counted and can be audited after trial completion.
    """

    # Triggered when an agent attempts to interrupt in a phase that doesn't allow it
    INTERRUPTION_NOT_ALLOWED = "interruption_not_allowed"
    # Triggered when judge prompt or output contains demographic cue values under blinding
    JUDGE_CUE_EXPOSURE = "judge_cue_exposure"
    # Triggered when a role speaks in a phase they're not authorized for
    ROLE_PHASE_MISMATCH = "role_phase_mismatch"


@dataclass(slots=True)
class AgentBudget:
    """Byte/token accounting for a specific agent-role combination."""

    max_bytes: int = 4096
    max_tokens: Optional[int] = None
    max_turns: Optional[int] = None


@dataclass(slots=True)
class PhaseBudget:
    """Constraints enforced per procedural phase."""

    phase: Phase
    max_messages: int = 2
    allow_interruptions: bool = False


@dataclass(slots=True)
class Person:
    """Represents a specific identity used in a cue."""
    
    name: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CueToggle:
    """Represents a demographic or sociolinguistic cue that can be switched."""

    name: str
    control_person: Person
    treatment_person: Person
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        control_person: Person | str | None = None,
        treatment_person: Person | str | None = None,
        metadata: Mapping[str, str] | None = None,
        control_value: str | None = None,
        treatment_value: str | None = None,
    ) -> None:
        """Support both Person-based and legacy string-based construction.

        Valid examples:
        - CueToggle("cue", control_person=Person(...), treatment_person=Person(...))
        - CueToggle("cue", "control name", "treatment name")
        - CueToggle("cue", control_value="control name", treatment_value="treatment name")
        """

        control = control_person if control_person is not None else control_value
        treatment = treatment_person if treatment_person is not None else treatment_value
        if control is None or treatment is None:
            raise TypeError("CueToggle requires control and treatment values.")

        control_obj = control if isinstance(control, Person) else Person(name=str(control))
        treatment_obj = treatment if isinstance(treatment, Person) else Person(name=str(treatment))
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "control_person", control_obj)
        object.__setattr__(self, "treatment_person", treatment_obj)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def control_value(self) -> str:
        """Backwards compatibility: return the name of the control person."""
        return self.control_person.name

    @property
    def treatment_value(self) -> str:
        """Backwards compatibility: return the name of the treatment person."""
        return self.treatment_person.name


@dataclass(slots=True)
class TrialConfig:
    """Complete configuration bundle for a simulated trial."""

    case_template: Path
    cue: CueToggle
    model_identifier: str
    seed: int
    agent_budgets: Mapping[Role, AgentBudget]
    phase_budgets: Sequence[PhaseBudget]
    backend_name: Optional[str] = None
    model_parameters: Mapping[str, object] = field(default_factory=dict)
    negative_controls: Sequence[CueToggle] = field(default_factory=tuple)
    # Active cue assignment details (set by orchestration when pairing)
    cue_condition: Optional[str] = None  # "control" | "treatment"
    cue_value: Optional[str] = None
    block_key: Optional[str] = None
    is_placebo: bool = False
    # Policy toggles
    judge_blinding: bool = False
    # NEW: Enhanced blinding mode that redacts BOTH control and treatment cue values
    # from judge prompts, not just the active one. Prevents any demographic leakage.
    strict_blinding: bool = False
    # NEW: When enabled, validates that roles only speak in their designated phases
    # (e.g., defense can't speak during verdict). Raises ValueError if violated.
    enforce_role_phase_policy: bool = True
    notes: Optional[str] = None
    output_dir: Optional[Path] = None

    def budget_for(self, role: Role) -> AgentBudget:
        """Return the configured budget for a role."""

        return self.agent_budgets[role]

    def get_role_budget(self, role: Role) -> AgentBudget:
        """Backward-compatible alias used by token budget utilities."""

        return self.budget_for(role)

    @property
    def role_budgets(self) -> Mapping[Role, AgentBudget]:
        """Backward-compatible alias used by token budget summaries."""

        return self.agent_budgets

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
