"""CLI entry point for kicking off a pilot paired trial."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from bailiff.agents.base import AgentSpec
from bailiff.agents.prompts import prompt_for
from bailiff.core.config import AgentBudget, CueToggle, PhaseBudget, Phase, Role, TrialConfig
from bailiff.core.logging import default_log_factory
from bailiff.orchestration.pipeline import TrialPipeline
from bailiff.orchestration.randomization import blocked_permutations
from bailiff.core.io import write_jsonl
from bailiff.datasets.templates import cue_catalog
import yaml


class EchoBackend:
    """Placeholder backend that echoes prompts for offline testing."""

    def __call__(self, prompt: str, **_: object) -> str:
        return f"[ECHO]\n{prompt}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pilot B.A.I.L.I.F.F. trial pair.")
    parser.add_argument("case", nargs="?", type=Path, help="Path to the case template file.")
    parser.add_argument("--config", type=Path, help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--backend", choices=["echo", "groq", "gemini"], default="echo")
    parser.add_argument("--model", type=str, default=None, help="Model identifier for backend.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSONL output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load base config either from YAML or CLI fallbacks
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        cue_key = cfg.get("cue", "name_ethnicity")
        catalog = cue_catalog()
        cue_def = catalog.get(cue_key)
        if cue_def is None:
            raise SystemExit(f"Unknown cue key in config: {cue_key}")
        cue = cue_def
        case_path = Path(cfg.get("case_template", args.case or "")).resolve()
        model_id = cfg.get("model_identifier", args.model or "echo")
        seed = int(cfg.get("seed", args.seed))
        judge_blinding = bool(cfg.get("judge_blinding", False))
        # Budgets
        budgets = {
            Role.JUDGE: AgentBudget(max_bytes=int(cfg.get("agent_budgets", {}).get("judge", {}).get("max_bytes", 1500))),
            Role.PROSECUTION: AgentBudget(max_bytes=int(cfg.get("agent_budgets", {}).get("prosecution", {}).get("max_bytes", 1800))),
            Role.DEFENSE: AgentBudget(max_bytes=int(cfg.get("agent_budgets", {}).get("defense", {}).get("max_bytes", 1800))),
        }
        # Phase budgets (fallback defaults if missing)
        pb_cfg = cfg.get("phase_budgets", [])
        phase_budgets = [
            PhaseBudget(phase=Phase(item["phase"]), max_messages=int(item.get("max_messages", 1)))
            if isinstance(item, dict)
            else PhaseBudget(phase=Phase(item))
            for item in pb_cfg
        ] or [PhaseBudget(phase=ph) for ph in Phase]
    else:
        if args.case is None:
            raise SystemExit("Provide a case path or a --config file")
        cue = CueToggle(
            name="name_ethnicity",
            control_value="Alex Johnson",
            treatment_value="DeShawn Jackson",
        )
        case_path = args.case.resolve()
        model_id = args.model or "echo"
        seed = args.seed
        judge_blinding = False
        budgets = {
            Role.JUDGE: AgentBudget(max_bytes=1500),
            Role.PROSECUTION: AgentBudget(max_bytes=1800),
            Role.DEFENSE: AgentBudget(max_bytes=1800),
        }
        phase_budgets = [PhaseBudget(phase=phase) for phase in Phase]
    base_config = TrialConfig(
        case_template=case_path,
        cue=cue,
        model_identifier=model_id,
        seed=seed,
        agent_budgets=budgets,
        phase_budgets=phase_budgets,
        judge_blinding=judge_blinding,
    )
    # Select backend
    if args.backend == "echo":
        backend = EchoBackend()
    elif args.backend == "groq":
        try:
            from bailiff.agents.backends import GroqBackend  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise SystemExit(f"Groq backend unavailable: {e}")
        backend = GroqBackend(model=args.model or "llama3-8b-8192")
    else:  # gemini
        try:
            from bailiff.agents.backends import GeminiBackend  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise SystemExit(f"Gemini backend unavailable: {e}")
        backend = GeminiBackend(model=args.model or "gemini-1.5-flash")
    agents = {
        role: AgentSpec(role=role, system_prompt=prompt_for(role), backend=backend)
        for role in Role
    }
    pipeline = TrialPipeline(agents=agents, log_factory=default_log_factory)
    assignments = blocked_permutations([cue.control_value, cue.treatment_value], seeds=[seed])
    pair_plan = next(pipeline.assign_pairs(base_config, assignments))
    logs = pipeline.run_pair(pair_plan)
    for log in logs:
        print(f"Trial {log.trial_id} produced {len(log.utterances)} utterances")
    if args.out:
        write_jsonl(logs, args.out)


if __name__ == "__main__":
    main()
