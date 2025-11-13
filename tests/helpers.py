"""Shared helpers for unit tests."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

from bailiff.core.config import AgentBudget, CueToggle, Phase, PhaseBudget, Role, TrialConfig
from bailiff.core.events import TrialLog, UtteranceLog
from bailiff.core.logging import default_log_factory
from bailiff.core.session import TrialSession


def make_session() -> Tuple[TrialSession, TrialLog]:
    cue = CueToggle(
        name="name_ethnicity",
        control_value="Alex Johnson",
        treatment_value="DeShawn Jackson",
        metadata={},
    )
    config = TrialConfig(
        case_template=Path("bailiff/datasets/cases/traffic.yaml"),
        cue=cue,
        model_identifier="echo",
        seed=123,
        agent_budgets={role: AgentBudget(max_bytes=1000, max_tokens=100) for role in Role},
        phase_budgets=[PhaseBudget(phase=p) for p in Phase],
    )
    responders = {role: (lambda *_args: "") for role in Role}
    session = TrialSession(config=config, responders=responders, log_factory=default_log_factory)
    log = default_log_factory(config)
    session._log = log
    return session, log


def make_utterance(
    content: str,
    *,
    role: Role = Role.JUDGE,
    phase: Phase = Phase.VERDICT,
) -> UtteranceLog:
    return UtteranceLog(
        role=role,
        phase=phase,
        content=content,
        byte_count=len(content.encode("utf-8")),
        token_count=len(content.split()),
        addressed_to=None,
        timestamp=datetime.utcnow(),
    )
