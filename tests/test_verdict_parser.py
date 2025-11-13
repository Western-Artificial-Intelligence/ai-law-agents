"""Tests for structured verdict parsing."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bailiff.core.config import AgentBudget, CueToggle, Phase, PhaseBudget, Role, TrialConfig
from bailiff.core.events import TrialLog, UtteranceLog
from bailiff.core.logging import default_log_factory
from bailiff.core.session import TrialSession


def _make_session() -> tuple[TrialSession, TrialLog]:
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
        seed=1,
        agent_budgets={role: AgentBudget(max_bytes=1000) for role in Role},
        phase_budgets=[PhaseBudget(phase=p) for p in Phase],
    )
    responders = {role: (lambda _role, _phase, _prompt: "") for role in Role}
    session = TrialSession(config=config, responders=responders, log_factory=default_log_factory)
    log = default_log_factory(config)
    session._log = log
    return session, log


def _utterance(content: str) -> UtteranceLog:
    return UtteranceLog(
        role=Role.JUDGE,
        phase=Phase.VERDICT,
        content=content,
        byte_count=len(content.encode("utf-8")),
        token_count=len(content.split()),
        addressed_to=None,
        timestamp=datetime.utcnow(),
    )


def test_verdict_parser_prefers_structured_json() -> None:
    session, log = _make_session()
    log.utterances.append(_utterance('{"verdict":"guilty","sentence":24}\nFull explanation follows.'))

    session._parse_and_set_verdict_sentence()

    assert log.verdict == "guilty"
    assert log.sentence == "24"


def test_verdict_parser_falls_back_to_regex() -> None:
    session, log = _make_session()
    log.utterances.append(_utterance("After deliberation the court finds the defendant not guilty. Sentence 0 years."))

    session._parse_and_set_verdict_sentence()

    assert log.verdict == "not_guilty"
    assert log.sentence == "0"
