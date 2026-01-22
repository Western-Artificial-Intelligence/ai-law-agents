"""CLI entry point for kicking off a pilot paired trial."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings
import yaml

from bailiff.agents.base import AgentBackend, AgentSpec, RetryPolicy
from bailiff.agents.prompts import prompt_for
from bailiff.core.config import AgentBudget, CueToggle, PhaseBudget, Phase, Role, TrialConfig
from bailiff.core.io import RunManifest, RunManifestEntry, compute_prompt_hash, write_jsonl
from bailiff.core.logging import default_log_factory
from bailiff.core.token_budget import TokenBudgetAuditor, TokenQuotaBudget, set_auditor
from bailiff.datasets.templates import cue_catalog
from bailiff.orchestration.pipeline import TrialPipeline
from bailiff.orchestration.blocks import build_blocks, resolve_placebos
from bailiff.orchestration.randomization import block_identifier, blockwise_permutations

load_dotenv()  # Load environment variables from .env file


class Backend(str, Enum):
    ECHO = "echo"
    GROQ = "groq"
    GEMINI = "gemini"
    LOCAL = "local"


class TokenConfig(BaseSettings):
    """Token budget configuration settings."""

    enforce_budget: bool = Field(True, description="Whether to enforce token budget limits")
    token_log_path: Optional[Path] = Field(None, description="Path to token usage log file")
    budget_report_path: Optional[Path] = Field(None, description="Path for token budget reports")
    budget_config_path: Optional[Path] = Field(None, description="Path to token budget configuration file")


class PilotConfig(BaseSettings):
    """Configuration for pilot trial runs with environment variable support."""

    case: Optional[Path] = Field(None, description="Path to the case template file")
    config: Optional[Path] = Field(None, description="Path to YAML config file")
    cues: Optional[Path] = Field(None, description="Path to YAML file containing custom cue definitions")
    seed: int = Field(42, description="Base random seed")
    backend: Backend = Field(Backend.ECHO, description="LLM backend to use")
    model: Optional[str] = Field(None, description="Model identifier for backend")
    out: Optional[Path] = Field(None, description="Optional JSONL output path")
    placebos: List[str] = Field(default_factory=list, description="Placebo cue keys to schedule")
    manifest: Optional[Path] = Field(None, description="Optional manifest path to append run metadata")
    timeout_seconds: float = Field(30.0, description="Backend timeout in seconds")
    max_retries: int = Field(2, description="Maximum number of backend retries")
    backoff_seconds: float = Field(1.0, description="Initial backoff between retries")
    backoff_multiplier: float = Field(2.0, description="Multiplicative backoff factor")
    rate_limit_seconds: float = Field(0.0, description="Sleep between calls to respect rate limits")
    backend_params: Dict[str, object] = Field(default_factory=dict, description="Backend parameter overrides")
    token: TokenConfig = Field(default_factory=TokenConfig, description="Token budget settings")

    class Config:
        env_prefix = "BAILIFF_"  # Environment variables will be prefixed with BAILIFF_


class EchoBackend:
    """Placeholder backend that echoes prompts for offline testing."""

    def __init__(self, enforce_budget: bool = False) -> None:
        self.enforce_budget = enforce_budget

    def __call__(self, prompt: str, **_: object) -> str:
        import time

        time.sleep(0.1)
        if self.enforce_budget:
            from bailiff.core.token_budget import check_run_allowed, register_token_usage
            import uuid

            run_id = str(uuid.uuid4())
            tokens = len(prompt) // 4
            allowed, reason = check_run_allowed(
                run_id=run_id,
                model_identifier="echo",
                api_key_id="echo",
                estimated_tokens=tokens * 2,
            )
            if not allowed:
                raise RuntimeError(f"Token budget exceeded: {reason}")
            register_token_usage(
                run_id=run_id,
                model_identifier="echo",
                api_key_id="echo",
                tokens_prompt=tokens,
                tokens_completion=tokens,
            )
        return f"[ECHO]\n{prompt}"


def _load_backend(
    choice: Backend,
    model_identifier: str,
    runtime_params: Dict[str, object],
    metadata: Dict[str, object],
    enforce_budget: bool,
) -> AgentBackend:
    """Instantiate the requested backend, mutating parameter dictionaries in-place."""

    if choice == Backend.ECHO:
        return EchoBackend(enforce_budget=enforce_budget)
    if choice == Backend.GROQ:
        try:
            from bailiff.agents.backends import GroqBackend  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise SystemExit(f"Groq backend unavailable: {exc}")
        return GroqBackend(model=model_identifier or "llama3-8b-8192", enforce_budget=enforce_budget)
    if choice == Backend.GEMINI:
        try:
            from bailiff.agents.backends import GeminiBackend  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise SystemExit(f"Gemini backend unavailable: {exc}")
        return GeminiBackend(model=model_identifier or "gemini-1.5-flash", enforce_budget=enforce_budget)
    if choice == Backend.LOCAL:
        try:
            from bailiff.agents.backends_local import LlamaCppBackend, LocalTransformersBackend  # type: ignore
        except Exception as exc:
            raise SystemExit(f"Local backend unavailable: {exc}")
        provider = str(runtime_params.pop("provider", "transformers")).lower()
        metadata.setdefault("provider", provider)
        if provider == "llama_cpp":
            model_path = str(runtime_params.pop("model_path", model_identifier))
            if not model_path:
                raise SystemExit("Provide --backend-param model_path=<gguf> for llama_cpp provider.")
            metadata.setdefault("model_path", model_path)
            n_ctx = int(runtime_params.pop("n_ctx", 2048))
            metadata.setdefault("n_ctx", n_ctx)
            n_threads_value = runtime_params.pop("n_threads", None)
            n_threads = int(n_threads_value) if n_threads_value is not None else None
            if n_threads is not None:
                metadata.setdefault("n_threads", n_threads)
            return LlamaCppBackend(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                enforce_budget=enforce_budget,
            )
        model_name = str(runtime_params.pop("model_name", model_identifier))
        if not model_name:
            raise SystemExit("Provide --backend-param model_name=<hf-id> or --model for local transformers backend.")
        metadata.setdefault("model_name", model_name)
        device = runtime_params.pop("device", None)
        if device is not None:
            metadata.setdefault("device", device)
        return LocalTransformersBackend(
            model_name_or_path=model_name,
            device=device,
            enforce_budget=enforce_budget,
        )
    raise SystemExit(f"Unsupported backend choice: {choice.value}")


def parse_args() -> tuple[dict[str, object], dict[str, object]]:
    """Parse command line arguments into a configuration dict and token overrides."""

    import argparse

    parser = argparse.ArgumentParser(description="Run a pilot B.A.I.L.I.F.F. trial pair.")
    parser.add_argument("--case", type=Path, help="Path to the case template file")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    parser.add_argument("--cues", type=Path, help="Path to YAML file containing custom cue definitions")
    parser.add_argument("--seed", type=int, help="Base random seed")
    parser.add_argument("--backend", choices=["echo", "groq", "gemini", "local"], help="LLM backend to use")
    parser.add_argument("--model", help="Model identifier for backend")
    parser.add_argument("--out", type=Path, help="Optional JSONL output path")
    parser.add_argument("--manifest", type=Path, help="Optional manifest JSONL path")
    parser.add_argument(
        "--placebo",
        action="append",
        dest="placebos",
        default=[],
        help="Placebo cue key to schedule as a negative control (repeatable).",
    )
    parser.add_argument("--timeout-seconds", type=float, help="Backend timeout in seconds")
    parser.add_argument("--max-retries", type=int, help="Maximum backend retries")
    parser.add_argument("--backoff-seconds", type=float, help="Initial retry backoff seconds")
    parser.add_argument("--backoff-multiplier", type=float, help="Backoff multiplier per retry")
    parser.add_argument("--rate-limit-seconds", type=float, help="Sleep between backend calls")
    parser.add_argument(
        "--backend-param",
        action="append",
        dest="backend_params",
        default=[],
        help="Backend parameter override in key=value form (repeatable).",
    )

    parser.add_argument("--disable-budget", action="store_true", help="Disable token budget enforcement")
    parser.add_argument("--token-log", type=Path, help="Path to token usage log")
    parser.add_argument("--token-report", type=Path, help="Path for token budget report")
    parser.add_argument("--budget-config", type=Path, help="Path to token budget configuration file")
    parser.add_argument(
        "--model-limit",
        action="append",
        nargs=2,
        metavar=("MODEL", "LIMIT"),
        help="Set token limit for a model (repeatable).",
    )
    parser.add_argument(
        "--key-limit",
        action="append",
        nargs=2,
        metavar=("KEY", "LIMIT"),
        help="Set token limit for an API key (repeatable).",
    )
    parser.add_argument("--alert-threshold", type=int, help="Percentage threshold for alerts")

    args = parser.parse_args()
    token_cli = {
        "disable_budget": args.disable_budget,
        "token_log": args.token_log,
        "token_report": args.token_report,
        "budget_config": args.budget_config,
        "model_limit": args.model_limit or [],
        "key_limit": args.key_limit or [],
        "alert_threshold": args.alert_threshold,
    }
    token_keys = set(token_cli.keys())
    parsed = {k: v for k, v in vars(args).items() if k not in token_keys and v is not None}
    if "backend_params" in parsed:
        param_dict: Dict[str, object] = {}
        for item in parsed.pop("backend_params"):
            if "=" not in item:
                raise SystemExit(f"Invalid --backend-param '{item}', expected key=value")
            key, value = item.split("=", 1)
            param_dict[key] = value
        parsed["backend_params"] = param_dict
    return parsed, token_cli


def setup_token_budget(config: PilotConfig, token_cli: Dict[str, object]) -> None:
    """Initialize the token budget auditor with the provided configuration."""

    token_config = config.token

    if token_cli.get("disable_budget"):
        token_config.enforce_budget = False
    if token_cli.get("token_log"):
        token_config.token_log_path = token_cli["token_log"]
    if token_cli.get("token_report"):
        token_config.budget_report_path = token_cli["token_report"]
    if token_cli.get("budget_config"):
        token_config.budget_config_path = token_cli["budget_config"]

    alert_threshold = token_cli.get("alert_threshold")
    budget = TokenQuotaBudget(
        alert_threshold_percent=alert_threshold if alert_threshold is not None else 80
    )

    model_limits = token_cli.get("model_limit") or []
    key_limits = token_cli.get("key_limit") or []

    for model, limit in model_limits:
        budget.quota_limits[str(model)] = int(limit)
    for key, limit in key_limits:
        budget.quota_limits[str(key)] = int(limit)

    use_file_config = not model_limits and not key_limits

    if use_file_config and token_config.budget_config_path and token_config.budget_config_path.exists():
        try:
            with open(token_config.budget_config_path, "r") as f:
                budget_config = json.load(f)

            if "model_limits" in budget_config:
                for model, limit in budget_config["model_limits"].items():
                    budget.quota_limits[str(model)] = int(limit)
            if "key_limits" in budget_config:
                for key, limit in budget_config["key_limits"].items():
                    budget.quota_limits[str(key)] = int(limit)
            if alert_threshold is None and "alert_threshold_percent" in budget_config:
                budget.alert_threshold_percent = int(budget_config["alert_threshold_percent"])
        except Exception as exc:
            raise SystemExit(f"Error loading token budget config: {exc}")
    elif use_file_config and config.config and Path(config.config).exists():
        yaml_config = yaml.safe_load(Path(config.config).read_text(encoding="utf-8"))
        if "token_budget" in yaml_config:
            tb_config = yaml_config["token_budget"]
            if "model_limits" in tb_config:
                for model, limit in tb_config["model_limits"].items():
                    budget.quota_limits[str(model)] = int(limit)
            if "key_limits" in tb_config:
                for key, limit in tb_config["key_limits"].items():
                    budget.quota_limits[str(key)] = int(limit)
            if alert_threshold is None and "alert_threshold_percent" in tb_config:
                budget.alert_threshold_percent = int(tb_config["alert_threshold_percent"])

    token_log_path = token_config.token_log_path or Path("runs/token_usage.jsonl")
    auditor = TokenBudgetAuditor(
        budget=budget,
        token_log_path=token_log_path,
        load_existing=True,
    )
    set_auditor(auditor)

    if token_config.budget_report_path:
        auditor.export_report(token_config.budget_report_path)


def _load_cues_from_file(cues_path: Path) -> Dict[str, CueToggle]:
    """Load custom cue definitions from a YAML file."""

    try:
        data = yaml.safe_load(cues_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to load cues file '{cues_path}': {exc}")

    if not isinstance(data, dict):
        raise SystemExit(f"Cues file '{cues_path}' must contain a dictionary mapping cue keys to definitions")

    catalog: Dict[str, CueToggle] = {}
    for cue_key, cue_data in data.items():
        if not isinstance(cue_data, dict):
            raise SystemExit(f"Invalid cue definition for '{cue_key}' in '{cues_path}': expected dictionary")

        name = cue_data.get("name", cue_key)
        control_value = cue_data.get("control_value")
        treatment_value = cue_data.get("treatment_value")

        if control_value is None or treatment_value is None:
            raise SystemExit(
                f"Cue '{cue_key}' in '{cues_path}' missing required fields: "
                f"control_value and treatment_value are required"
            )

        metadata = cue_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        catalog[cue_key] = CueToggle(
            name=str(name),
            control_value=str(control_value),
            treatment_value=str(treatment_value),
            metadata={k: str(v) for k, v in metadata.items()},
        )

    return catalog


def _case_blob(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.name


def _run_id(case_identifier: str, model_identifier: str, cue_name: str, seed: int, backend: str) -> str:
    token = f"{case_identifier}|{model_identifier}|{cue_name}|{seed}|{backend}"
    return compute_prompt_hash(token)


def _prompt_hash_for_log(log) -> str:
    return compute_prompt_hash(
        log.trial_id,
        log.case_identifier,
        log.cue_name,
        log.cue_value or "",
        log.model_identifier,
        log.backend_name or "",
        log.cue_condition or "",
    )


def main() -> None:
    # Load configuration from environment variables and command line arguments
    config_args, token_cli = parse_args()
    args = PilotConfig(**config_args)
    setup_token_budget(args, token_cli)

    placebo_keys: List[str] = list(args.placebos)
    backend_params: Dict[str, object] = dict(args.backend_params)
    policy_cfg: Dict[str, object] = {}

    # Load custom cues from file if specified, otherwise use default catalog
    if args.cues:
        custom_catalog = _load_cues_from_file(args.cues)
        # Merge with default catalog (custom cues override defaults with same key)
        catalog = {**cue_catalog(), **custom_catalog}
    else:
        catalog = cue_catalog()

    # Load additional config from YAML file if specified
    if args.config:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        cue_key = cfg.get("cue", "name_ethnicity")
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
        model_id = args.model or cfg.get("model_identifier") or args.backend.value
        seed = int(cfg.get("seed", args.seed))
        judge_blinding = bool(cfg.get("judge_blinding", False))
        backend_params.update(cfg.get("backend_params", {}) or {})
        policy_cfg = cfg.get("backend_policy", {}) or {}
        budgets = {
            Role.JUDGE: AgentBudget(max_bytes=int(cfg.get("agent_budgets", {}).get("judge", {}).get("max_bytes", 1500))),
            Role.PROSECUTION: AgentBudget(
                max_bytes=int(cfg.get("agent_budgets", {}).get("prosecution", {}).get("max_bytes", 1800))
            ),
            Role.DEFENSE: AgentBudget(max_bytes=int(cfg.get("agent_budgets", {}).get("defense", {}).get("max_bytes", 1800))),
        }
        pb_cfg = cfg.get("phase_budgets", [])
        phase_budgets = [
            PhaseBudget(phase=Phase(item["phase"]), max_messages=int(item.get("max_messages", 1)))
            if isinstance(item, dict)
            else PhaseBudget(phase=Phase(item))
            for item in pb_cfg
        ] or [PhaseBudget(phase=ph) for ph in Phase]
        placebo_keys.extend(cfg.get("placebos", []) or [])
    else:
        if args.case is None:
            raise SystemExit("Provide a case path or a --config file")
        default_cue = catalog.get("name_ethnicity")
        if default_cue is not None:
            cue = default_cue
        else:
            cue = CueToggle(
                name="name_ethnicity",
                control_value="Alex Johnson",
                treatment_value="DeShawn Jackson",
            )
        case_path = args.case.resolve()
        model_id = args.model or args.backend.value
        seed = args.seed
        judge_blinding = False
        budgets = {
            Role.JUDGE: AgentBudget(max_bytes=1500),
            Role.PROSECUTION: AgentBudget(max_bytes=1800),
            Role.DEFENSE: AgentBudget(max_bytes=1800),
        }
        phase_budgets = [PhaseBudget(phase=phase) for phase in Phase]
        policy_cfg = {}
    placebo_keys = list(dict.fromkeys(placebo_keys))
    placebo_toggles = resolve_placebos(placebo_keys)
    case_identifier = case_path.stem
    block_key = block_identifier(case_identifier, model_id)
    retry_policy = RetryPolicy(
        max_retries=int(policy_cfg.get("max_retries", args.max_retries)),
        initial_backoff=float(policy_cfg.get("backoff_seconds", args.backoff_seconds)),
        backoff_multiplier=float(policy_cfg.get("backoff_multiplier", args.backoff_multiplier)),
        timeout_seconds=float(policy_cfg.get("timeout_seconds", args.timeout_seconds)),
        rate_limit_seconds=float(policy_cfg.get("rate_limit_seconds", args.rate_limit_seconds)),
    )
    runtime_params = dict(backend_params)
    metadata_params = dict(backend_params)
    backend = _load_backend(
        args.backend,
        model_id,
        runtime_params,
        metadata_params,
        args.token.enforce_budget,
    )
    param_snapshot = dict(runtime_params)
    base_config = TrialConfig(
        case_template=case_path,
        cue=cue,
        model_identifier=model_id,
        backend_name=args.backend.value,
        model_parameters=dict(metadata_params),
        seed=seed,
        agent_budgets=budgets,
        phase_budgets=phase_budgets,
        judge_blinding=judge_blinding,
        negative_controls=tuple(placebo_toggles),
        block_key=block_key,
    )

    agents = {
        role: AgentSpec(
            role=role,
            system_prompt=prompt_for(role),
            backend=backend,
            default_params=param_snapshot,
            retry_policy=retry_policy,
        )
        for role in Role
    }
    pipeline = TrialPipeline(
        agents=agents,
        log_factory=default_log_factory,
        enforce_budget=args.token.enforce_budget,
    )
    cues_for_blocks: List[CueToggle] = [cue, *placebo_toggles]
    placebo_names = [toggle.name for toggle in placebo_toggles]
    block_defs = build_blocks(case_identifier, model_id, cues_for_blocks, seeds=[seed], placebo_names=placebo_names)
    logs: List = []
    manifest = RunManifest(args.manifest) if args.manifest else None
    case_blob = _case_blob(case_path)
    for assignment in blockwise_permutations(block_defs):
        plan_iter = pipeline.assign_pairs(base_config, [assignment])
        pair_plan = next(plan_iter)
        pair_logs = pipeline.run_pair(pair_plan)
        logs.extend(pair_logs)
        if manifest:
            cue_name = assignment.cue_name or cue.name
            run_id = _run_id(case_identifier, model_id, cue_name, assignment.seed, args.backend.value)
            control_hash = _prompt_hash_for_log(pair_logs[0])
            treatment_hash = _prompt_hash_for_log(pair_logs[1])
            manifest.append(
                RunManifestEntry(
                    run_id=run_id,
                    case_identifier=case_identifier,
                    model_identifier=model_id,
                    backend=args.backend.value,
                    cue_name=cue_name,
                    cue_control=assignment.control_value,
                    cue_treatment=assignment.treatment_value,
                    control_seed=assignment.seed,
                    treatment_seed=assignment.seed + 1,
                    block_key=assignment.block_key,
                    is_placebo=assignment.is_placebo,
                    prompt_hash=compute_prompt_hash(control_hash, treatment_hash, case_blob),
                    prompt_hash_control=control_hash,
                    prompt_hash_treatment=treatment_hash,
                    params=param_snapshot,
                    trial_ids=tuple(log.trial_id for log in pair_logs),
                    log_path=str(args.out) if args.out else None,
                )
            )
    for log in logs:
        placebo_tag = " [placebo]" if log.is_placebo else ""
        print(f"Trial {log.trial_id}{placebo_tag} produced {len(log.utterances)} utterances")
    if args.out:
        write_jsonl(logs, args.out)


if __name__ == "__main__":
    main()
