"""Regression tests for token budget enforcement."""
import json
import tempfile
from pathlib import Path

from bailiff.core.config import AgentBudget, CueToggle, Phase, PhaseBudget, Role, TrialConfig
from bailiff.core.logging import default_log_factory
from bailiff.core.session import TrialSession
from bailiff.core.token_budget import TokenBudget, TokenBudgetEnforcer


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


def test_apply_role_budgets_truncates_to_per_turn_tokens():
    session = _make_session(max_tokens=3)
    text = "token0 token1 token2 token3 token4"

    clipped, token_count = session._apply_role_budgets(Role.DEFENSE, text)

    assert token_count == 3
    assert session._tokens_used[Role.DEFENSE] == 3
    assert "token3" not in clipped
    assert "token4" not in clipped


def test_apply_role_budgets_uses_per_turn_cap_even_after_prior_usage():
    session = _make_session(max_tokens=2)
    session._tokens_used[Role.JUDGE] = 2

    clipped, token_count = session._apply_role_budgets(Role.JUDGE, "one more message")

    assert clipped == "one more"
    assert token_count == 2
    assert session._tokens_used[Role.JUDGE] == 4


def test_token_budget_tracking_and_summary():
    """Test that token usage is tracked and summary is generated correctly."""
    # Setup test config with output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test config with output directory
        config = TrialConfig(
            case_template=Path("bailiff/datasets/cases/traffic.yaml"),
            cue=CueToggle(name="test_cue", control_value="A", treatment_value="B"),
            model_identifier="gpt-4o-mini",
            seed=123,
            output_dir=temp_dir,
            agent_budgets={
                Role.JUDGE: AgentBudget(max_tokens=1000),
                Role.DEFENSE: AgentBudget(max_tokens=500),
                Role.PROSECUTION: AgentBudget(max_tokens=500),
            },
            phase_budgets=[PhaseBudget(phase=phase) for phase in Phase],
        )
        
        # Initialize token budget
        with TokenBudgetEnforcer(config) as budget:
            # Simulate some token usage
            budget.record_usage(Role.JUDGE, prompt_tokens=150, completion_tokens=50)
            budget.record_usage(Role.DEFENSE, prompt_tokens=200, completion_tokens=50)
            budget.record_usage(Role.PROSECUTION, prompt_tokens=100, completion_tokens=25)
            
            # Generate and save summary
            summary_path = budget.save_summary(temp_dir)
            
            # Verify summary file was created
            assert summary_path.exists()
            
            # Load and verify summary content
            with open(summary_path) as f:
                summary = json.load(f)
                
            # Check basic structure
            assert "timestamp" in summary
            assert "usage" in summary
            assert "alerts" in summary
            assert "budgets" in summary
            
            # Check usage was recorded correctly
            assert summary["usage"]["JUDGE"]["prompt_tokens"] == 150
            assert summary["usage"]["JUDGE"]["completion_tokens"] == 50
            assert summary["usage"]["JUDGE"]["total_tokens"] == 200
            
            # Check budget limits
            assert summary["budgets"]["JUDGE"]["max_tokens"] == 1000
            assert summary["budgets"]["DEFENSE"]["max_tokens"] == 500
            
            # Verify no alerts for normal usage
            assert len(summary["alerts"]) == 0
            
        # Test budget enforcement
        with TokenBudgetEnforcer(config) as budget:
            # Exceed budget
            budget.record_usage(Role.DEFENSE, prompt_tokens=600, completion_tokens=0)
            
            # Should generate an alert
            assert len(budget.alerts) > 0
            assert "exceeded" in budget.alerts[0]["message"].lower()
            
            # Get summary with alert
            summary = budget.get_summary()
            assert len(summary["alerts"]) > 0
            assert "DEFENSE" in summary["alerts"][0]["role"]


if __name__ == "__main__":
    test_token_budget_tracking_and_summary()
    print("All token budget tests passed!")

