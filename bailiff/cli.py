"""Small CLI helpers and an end-to-end echo validator for CI.

This exposes a programmatic function `e2e_validate_echo()` that runs a
minimal trial using the Echo backend and asserts the TrialLog schema and
basic metrics are present. A tiny console entrypoint `e2e_echo_main` is
provided for CI or local smoke runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from bailiff.agents.base import AgentSpec
from bailiff.agents.prompts import prompt_for
from bailiff.core.config import (
    AgentBudget,
    CueToggle,
    PhaseBudget,
    Phase,
    Role,
    TrialConfig,
)
from bailiff.core.logging import default_log_factory
from bailiff.orchestration.pipeline import TrialPipeline
from bailiff.orchestration.randomization import blocked_permutations

class EchoBackend:
    """Placeholder backend that echoes prompts for offline testing."""

    def __call__(self, prompt: str, **_: object) -> str:
        return f"[ECHO]\n{prompt}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def e2e_validate_echo(seed: int = 42) -> List:
    """Run a tiny echo-based pair and validate schema basics.

    Returns the list of two TrialLog objects if successful. Raises
    AssertionError on validation failures.
    """
    repo = _repo_root()
    case_path = repo / "bailiff" / "datasets" / "cases" / "traffic.yaml"
    assert case_path.exists(), f"Case template missing: {case_path}"

    cue = CueToggle(
        name="name_ethnicity",
        control_value="Alex Johnson",
        treatment_value="DeShawn Jackson",
    )

    budgets = {
        Role.JUDGE: AgentBudget(max_bytes=1500),
        Role.PROSECUTION: AgentBudget(max_bytes=1800),
        Role.DEFENSE: AgentBudget(max_bytes=1800),
    }
    phase_budgets = [PhaseBudget(phase=phase) for phase in Phase]

    base_config = TrialConfig(
        case_template=case_path,
        cue=cue,
        model_identifier="echo",
        seed=seed,
        agent_budgets=budgets,
        phase_budgets=phase_budgets,
        judge_blinding=False,
    )

    backend = EchoBackend()
    agents = {role: AgentSpec(role=role, system_prompt=prompt_for(role), backend=backend) for role in Role}
    pipeline = TrialPipeline(agents=agents, log_factory=default_log_factory)

    assignments = blocked_permutations([cue.control_value, cue.treatment_value], seeds=[seed])
    pair_plan = next(pipeline.assign_pairs(base_config, assignments))
    logs = pipeline.run_pair(pair_plan)
    print(logs)
    # Basic validations on returned TrialLog objects
    assert isinstance(logs, list) and len(logs) == 2
    for log in logs:
        # metadata
        assert hasattr(log, "trial_id") and isinstance(log.trial_id, str)
        assert hasattr(log, "case_identifier") and isinstance(log.case_identifier, str)
        assert getattr(log, "model_identifier", None) == "echo"
        assert hasattr(log, "cue_name") and log.cue_name == "name_ethnicity"
        assert hasattr(log, "cue_condition") and log.cue_condition in ("control", "treatment")
        assert hasattr(log, "cue_value") and isinstance(log.cue_value, str)
        assert hasattr(log, "seed") and isinstance(log.seed, int)
        assert hasattr(log, "started_at") and log.started_at is not None
        assert hasattr(log, "completed_at") and log.completed_at is not None

        # utterances
        assert hasattr(log, "utterances") and len(log.utterances) > 0
        for utt in log.utterances:
            assert hasattr(utt, "role")
            assert hasattr(utt, "phase")
            assert hasattr(utt, "content") and isinstance(utt.content, str)
            # (We currently have blank utt.content strings in the middle, but this may change
            # later in development) 
            assert hasattr(utt, "byte_count") and isinstance(utt.byte_count, int)
            # token_count may be None, but the field should exist
            assert hasattr(utt, "token_count")

    return logs


def e2e_echo_main() -> int:
    """Console entrypoint: run validator and print a short summary.

    Exit code 0 indicates success; non-zero on failure.
    """
    try:
        logs = e2e_validate_echo()
        print(f"E2E echo validation succeeded: produced {len(logs)} logs")
        return 0
    except AssertionError as e:
        print(f"E2E echo validation failed: {e}")
        return 2
    except Exception as e:  # pragma: no cover - unexpected error
        print(f"Unexpected error during E2E echo validation: {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(e2e_echo_main())
