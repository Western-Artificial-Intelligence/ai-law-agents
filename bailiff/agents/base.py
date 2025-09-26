"""Agent abstractions and prompt scaffolds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol

from bailiff.core.config import Phase, Role


class AgentBackend(Protocol):
    """Callable protocol for LLM backends."""

    def __call__(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover - interface
        ...


@dataclass(slots=True)
class AgentSpec:
    """Minimal specification for instantiating a role agent."""

    role: Role
    system_prompt: str
    backend: AgentBackend
    default_params: Optional[Mapping[str, object]] = None

    def to_responder(self) -> Callable[[Role, Phase, str], str]:
        """Return a TrialSession-compatible responder closure."""

        def _responder(role: Role, phase: Phase, prompt: str) -> str:
            composed = self._compose_prompt(role, phase, prompt)
            params = dict(self.default_params or {})
            return self.backend(composed, **params)

        return _responder

    def _compose_prompt(self, role: Role, phase: Phase, prompt: str) -> str:
        """Attach the system directive to the shared prompt."""

        return (
            f"[SYSTEM]\n{self.system_prompt}\n"
            f"[ROLE]\n{role.value}\n[PHASE]\n{phase.value}\n"
            f"[CONTEXT]\n{prompt}"
        )


def build_responder_map(specs: Mapping[Role, AgentSpec]) -> Dict[Role, Callable[[Role, Phase, str], str]]:
    """Convert AgentSpec instances into the responder map expected by the session."""

    return {role: spec.to_responder() for role, spec in specs.items()}
