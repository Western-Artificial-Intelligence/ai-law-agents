"""CLI entry point for kicking off a pilot paired trial."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from bailiff.agents.base import AgentSpec
from bailiff.agents.prompts import prompt_for
from bailiff.core.config import AgentBudget, CueToggle, PhaseBudget, Phase, Role, TrialConfig
from bailiff.core.logging import default_log_factory
from bailiff.orchestration.pipeline import TrialPipeline
from bailiff.orchestration.randomization import blocked_permutations
from bailiff.core.io import write_jsonl
from bailiff.datasets.templates import cue_catalog
import yaml

load_dotenv()  # Load environment variables from .env file

class Backend(str, Enum):
    ECHO = "echo"
    GROQ = "groq"
    GEMINI = "gemini"

class PilotConfig(BaseSettings):
    """Configuration for pilot trial runs with environment variable support."""
    case: Optional[Path] = Field(None, description="Path to the case template file")
    config: Optional[Path] = Field(None, description="Path to YAML config file")
    seed: int = Field(42, description="Base random seed")
    backend: Backend = Field(Backend.ECHO, description="LLM backend to use")
    model: Optional[str] = Field(None, description="Model identifier for backend")
    out: Optional[Path] = Field(None, description="Optional JSONL output path")

    class Config:
        env_prefix = "BAILIFF_"  # Environment variables will be prefixed with BAILIFF_


class EchoBackend:
    """Placeholder backend that echoes prompts for offline testing."""

    def __call__(self, prompt: str, **_: object) -> str:
        return f"[ECHO]\n{prompt}"


def parse_args() -> dict[str, object]:
    """Parse command line arguments into a dictionary."""
    import argparse
    parser = argparse.ArgumentParser(description="Run a pilot B.A.I.L.I.F.F. trial pair.")
    parser.add_argument("--case", type=Path, help="Path to the case template file")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, help="Base random seed")
    parser.add_argument("--backend", choices=["echo", "groq", "gemini"], help="LLM backend to use")
    parser.add_argument("--model", help="Model identifier for backend")
    parser.add_argument("--out", type=Path, help="Optional JSONL output path")
    args = parser.parse_args()
    # Convert to dict and remove None values
    return {k: v for k, v in vars(args).items() if v is not None}

def main() -> None:
    # Load configuration from environment variables and command line arguments
    args = PilotConfig(**parse_args())
    
    # Load additional config from YAML file if specified
    if args.config:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        cue_key = cfg.get("cue", "name_ethnicity")
        catalog = cue_catalog()
        cue_def = catalog.get(cue_key)
        if cue_def is None:
            raise SystemExit(f"Unknown cue key in config: {cue_key}")
        cue = cue_def
        case_template = cfg.get("case_template")
        if case_template is not None:
            case_path = Path(case_template).resolve()
        elif args.case is not None:
            case_path = args.case.resolve()
        else:
            raise SystemExit("No case template file specified in config or command line arguments.")
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
    
    # Select backend based on config
    if args.backend == Backend.ECHO:
        backend = EchoBackend()
    elif args.backend == Backend.GROQ:
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
