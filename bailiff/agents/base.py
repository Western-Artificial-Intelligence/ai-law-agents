"""Agent abstractions and prompt scaffolds."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from bailiff.core.config import Phase, Role


class AgentBackend(Protocol):
    """Callable protocol for LLM backends."""

    def __call__(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover - interface
        ...


@dataclass(slots=True)
class RetryPolicy:
    """Simple retry/backoff controls applied to backend calls."""

    max_retries: int = 2
    initial_backoff: float = 1.0
    backoff_multiplier: float = 2.0
    timeout_seconds: float = 30.0
    rate_limit_seconds: float = 0.0


class BackendTimeoutError(RuntimeError):
    """Raised when the backend exceeds the configured timeout."""


@dataclass(slots=True)
class AgentSpec:
    """Minimal specification for instantiating a role agent."""

    role: Role
    system_prompt: str
    backend: AgentBackend
    default_params: Optional[Mapping[str, object]] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def to_responder(self) -> Callable[[Role, Phase, str], str]:
        """Return a TrialSession-compatible responder closure."""

        def _responder(role: Role, phase: Phase, prompt: str) -> str:
            composed = self._compose_prompt(role, phase, prompt)
            params = dict(self.default_params or {})
            return self._call_with_retry(composed, params)

        return _responder

    def _compose_prompt(self, role: Role, phase: Phase, prompt: str) -> str:
        """Attach the system directive to the shared prompt."""

        return (
            f"[SYSTEM]\n{self.system_prompt}\n"
            f"[ROLE]\n{role.value}\n[PHASE]\n{phase.value}\n"
            f"[CONTEXT]\n{prompt}"
        )

    def _call_with_retry(self, prompt: str, params: Dict[str, object]) -> str:
        policy = self.retry_policy
        backoff = max(policy.initial_backoff, 0.0)
        attempt = 0
        while True:
            if policy.rate_limit_seconds > 0:
                time.sleep(policy.rate_limit_seconds)
            try:
                return _call_with_timeout(lambda: self.backend(prompt, **params), policy.timeout_seconds)
            except Exception:
                attempt += 1
                if attempt > policy.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= max(policy.backoff_multiplier, 1.0)


def build_responder_map(specs: Mapping[Role, AgentSpec]) -> Dict[Role, Callable[[Role, Phase, str], str]]:
    """Convert AgentSpec instances into the responder map expected by the session."""

    return {role: spec.to_responder() for role, spec in specs.items()}


def _call_with_timeout(func: Callable[[], Any], timeout: float) -> Any:
    """Execute callable with a timeout enforced via worker thread."""

    if timeout is None or timeout <= 0:
        return func()

    result: Dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = func()
        except Exception as exc:  # pragma: no cover - pass-through
            result["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise BackendTimeoutError(f"Backend call exceeded {timeout} seconds.")
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result.get("value")
