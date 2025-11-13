"""Regression tests for token budget enforcement."""
from pathlib import Path

from bailiff.core.config import AgentBudget, CueToggle, Phase, PhaseBudget, Role, TrialConfig
from bailiff.core.logging import default_log_factory
from bailiff.core.session import TrialSession


def _noop_responder(role: Role, phase: Phase, prompt: str) -> str:
    return ""


def _make_session(max_tokens: int, model_identifier: str = "gpt-4o-mini") -> TrialSession:
    budgets = {
        role: AgentBudget(max_bytes=4096, max_tokens=max_tokens)
        for role in Role
    }
    config = TrialConfig(
        case_template=Path("bailiff/datasets/cases/traffic.yaml"),
        cue=CueToggle(name="name_ethnicity", control_value="Alex", treatment_value="DeShawn"),
        model_identifier=model_identifier,
        seed=123,
        agent_budgets=budgets,
        phase_budgets=[PhaseBudget(phase=phase) for phase in Phase],
    )
    responders = {role: _noop_responder for role in Role}
    session = TrialSession(config=config, responders=responders, log_factory=default_log_factory)
    session._bytes_used = {role: 0 for role in Role}
    session._tokens_used = {role: 0 for role in Role}
    return session


def test_apply_role_budgets_truncates_to_remaining_tokens():
    session = _make_session(max_tokens=3)
    text = "token0 token1 token2 token3 token4"

    clipped, token_count = session._apply_role_budgets(Role.DEFENSE, text)

    assert token_count == 3
    assert session._tokens_used[Role.DEFENSE] == 3
    assert "token3" not in clipped
    assert "token4" not in clipped


def test_apply_role_budgets_returns_empty_when_budget_spent():
    session = _make_session(max_tokens=2)
    session._tokens_used[Role.JUDGE] = 2

    clipped, token_count = session._apply_role_budgets(Role.JUDGE, "one more message")

    assert clipped == ""
    assert token_count == 0
    assert session._tokens_used[Role.JUDGE] == 2
