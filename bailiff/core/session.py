"""Session orchestration for multi-agent trials."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, Optional

from .config import DEFAULT_PHASE_ORDER, Phase, Role, TrialConfig
from \.events import TrialLog, UtteranceLog
from \.logging import mark_completed

AgentResponder = Callable[[Role, Phase, str], str]


@dataclass
class TrialSession:
    """Coordinates role agents to execute a single trial under constraints."""

    config: TrialConfig
    responders: Dict[Role, AgentResponder]
    log_factory: Callable[[TrialConfig], TrialLog]
    policy_hooks: Optional[Dict[str, Callable[[TrialLog], None]]] = None
    _log: Optional[TrialLog] = field(default=None, init=False, repr=False)

    def run(self) -> TrialLog:
        """Execute the state machine and return the populated trial log."""

        self._log = self.log_factory(self.config)
        for phase in DEFAULT_PHASE_ORDER:
            self._run_phase(phase)
        if self.policy_hooks:
            for hook in self.policy_hooks.values():
                hook(self._log)
        return self._log

    def _run_phase(self, phase: Phase) -> None:
        """Run a phase by iterating over active roles."""

        active_roles = self._roles_for_phase(phase)
        for role in active_roles:
            content = self._emit(role, phase)
            record = self._build_record(role, phase, content)
            self._log.append(record)  # type: ignore[arg-type]

    def _roles_for_phase(self, phase: Phase) -> Iterable[Role]:
        """Return roles expected to contribute during a phase."""

        if phase in (Phase.OPENING, Phase.CLOSING):
            return (Role.PROSECUTION, Role.DEFENSE)
        if phase in (Phase.DIRECT, Phase.REDIRECT):
            return (Role.PROSECUTION,)
        if phase is Phase.CROSS:
            return (Role.DEFENSE,)
        return (Role.JUDGE,)

    def _emit(self, role: Role, phase: Phase) -> str:
        """Invoke a responder while respecting budgets (enforced later)."""

        responder = self.responders[role]
        prompt = self._build_prompt(role, phase)
        return responder(role, phase, prompt)

    def _build_prompt(self, role: Role, phase: Phase) -> str:
        """Construct the shared prompt context passed to an agent responder."""

        return (
            f"Case: {self.config.case_template}\n"
            f"Cue assignment: {self.config.cue.name}\n"
            f"Phase: {phase.value}\n"
            f"Role: {role.value}"
        )

    def _build_record(self, role: Role, phase: Phase, content: str) -> UtteranceLog:
        """Create a minimal log entry for downstream metric extraction."""

        return UtteranceLog(
            role=role,
            phase=phase,
            content=content,
            byte_count=len(content.encode("utf-8")),
            token_count=None,
            addressed_to=None,
            timestamp=datetime.utcnow(),
        )
