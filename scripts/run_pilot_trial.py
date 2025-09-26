"""CLI entry point for kicking off a pilot paired trial."""
from __future__ import annotations

import argparse
from pathlib import Path

from bailiff.agents.base import AgentSpec
from bailiff.agents.prompts import prompt_for
from bailiff.core.config import AgentBudget, CueToggle, PhaseBudget, Phase, Role, TrialConfig
from bailiff.core.logging import default_log_factory
from bailiff.orchestration.pipeline import TrialPipeline
from bailiff.orchestration.randomization import blocked_permutations


class EchoBackend:
    """Placeholder backend that echoes prompts for offline testing."""

    def __call__(self, prompt: str, **_: object) -> str:
        return f"[ECHO]\n{prompt}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pilot B.A.I.L.I.F.F. trial pair.")
    parser.add_argument("case", type=Path, help="Path to the case template file.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        case_template=args.case,
        cue=cue,
        model_identifier="echo",
        seed=args.seed,
        agent_budgets=budgets,
        phase_budgets=phase_budgets,
    )
    backend = EchoBackend()
    agents = {
        role: AgentSpec(role=role, system_prompt=prompt_for(role), backend=backend)
        for role in Role
    }
    pipeline = TrialPipeline(agents=agents, log_factory=default_log_factory)
    assignments = blocked_permutations([cue.control_value, cue.treatment_value], seeds=[args.seed])
    pair_plan = next(pipeline.assign_pairs(base_config, assignments))
    logs = pipeline.run_pair(pair_plan)
    for log in logs:
        print(f"Trial {log.trial_id} produced {len(log.utterances)} utterances")


if __name__ == "__main__":
    main()
