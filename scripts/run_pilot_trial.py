"""CLI entry point for kicking off a pilot paired trial with token budget support."""
from __future__ import annotations

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
from bailiff.core.token_budget import TokenBudget, TokenBudgetAuditor, get_auditor
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
    # Token budget settings
    token: TokenConfig = Field(default_factory=TokenConfig, description="Token budget settings")

    class Config:
        env_prefix = "BAILIFF_"  # Environment variables will be prefixed with BAILIFF_

def echo_backend_factory() -> AgentBackend:
    """Build a deterministic echo backend for offline testing."""
    return EchoBackend()

def groq_backend_factory(model: str, config: PilotConfig) -> AgentBackend:
    """Build a Groq backend with the configured retry policy."""
    from bailiff.agents.backends import GroqBackend
    return GroqBackend(model=model, enforce_budget=config.token.enforce_budget)

def gemini_backend_factory(model: str, config: PilotConfig) -> AgentBackend:
    """Build a Gemini backend with the configured retry policy."""
    from bailiff.agents.backends import GeminiBackend
    return GeminiBackend(model=model, enforce_budget=config.token.enforce_budget)

def local_backend_factory(model: str, config: PilotConfig) -> AgentBackend:
    """Build a local backend with the configured parameters."""
    provider = config.backend_params.get("provider", "transformers")
    if provider == "transformers":
        from bailiff.agents.backends_local import LocalTransformersBackend
        return LocalTransformersBackend(
            model_name_or_path=config.backend_params.get("model_name", model),
            device=config.backend_params.get("device"),
            enforce_budget=config.token.enforce_budget,
        )
    elif provider == "llama_cpp":
        from bailiff.agents.backends_local import LlamaCppBackend
        return LlamaCppBackend(
            model_path=config.backend_params.get("model_path", model),
            n_ctx=int(config.backend_params.get("n_ctx", 4096)),
            n_threads=int(config.backend_params.get("n_threads", 4)),
            enforce_budget=config.token.enforce_budget,
        )
    else:
        raise ValueError(f"Unknown local backend provider: {provider}")

class EchoBackend:
    """Mock LLM implementation that returns prompted text."""

    def __init__(self, enforce_budget: bool = False):
        self.enforce_budget = enforce_budget

    def __call__(self, prompt: str, **kwargs: object) -> str:
        import time
        time.sleep(0.1)  # Brief delay to simulate network latency
        
        if self.enforce_budget:
            from bailiff.core.token_budget import check_run_allowed, register_token_usage
            import uuid
            
            run_id = str(uuid.uuid4())
            tokens = len(prompt) // 4
            
            # Check token budget
            allowed, reason = check_run_allowed(
                run_id=run_id,
                model_identifier="echo",
                api_key_id="echo",
                estimated_tokens=tokens * 2  # Double for prompt + completion
            )
            
            if not allowed:
                raise RuntimeError(f"Token budget exceeded: {reason}")
            
            # Register token usage
            register_token_usage(
                run_id=run_id,
                model_identifier="echo",
                api_key_id="echo",
                tokens_prompt=tokens,
                tokens_completion=tokens
            )
        
        return f"ECHO: {prompt[:100]}..."

def setup_token_budget(config: PilotConfig, args=None) -> None:
    """Initialize the token budget auditor with the provided configuration."""
    token_config = config.token
    
    # Create the token budget configuration with custom alert threshold
    alert_threshold = args.alert_threshold if args and hasattr(args, 'alert_threshold') else 80
    budget = TokenBudget(alert_threshold_percent=alert_threshold)
    
    # First priority: Direct CLI arguments
    if args:
        # Process model limits
        if hasattr(args, 'model_limit') and args.model_limit:
            for model, limit in args.model_limit:
                budget.quota_limits[model] = int(limit)
                print(f"Setting model limit: {model} = {int(limit):,} tokens")
        
        # Process key limits
        if hasattr(args, 'key_limit') and args.key_limit:
            for key, limit in args.key_limit:
                budget.quota_limits[key] = int(limit)
                print(f"Setting key limit: {key} = {int(limit):,} tokens")
    
    # Second priority: Try loading from dedicated config file if no direct limits were specified
    use_file_config = not args or (
        not (hasattr(args, 'model_limit') and args.model_limit) and 
        not (hasattr(args, 'key_limit') and args.key_limit)
    )
    
    # Try loading from dedicated budget config file if needed
    if use_file_config and token_config.budget_config_path and token_config.budget_config_path.exists():
        try:
            with open(token_config.budget_config_path, "r") as f:
                budget_config = json.load(f)
                
            # Load model limits
            if "model_limits" in budget_config:
                for model, limit in budget_config["model_limits"].items():
                    budget.quota_limits[model] = int(limit)
            
            # Load key limits
            if "key_limits" in budget_config:
                for key, limit in budget_config["key_limits"].items():
                    budget.quota_limits[key] = int(limit)
            
            # Load alert threshold (only if not already set by CLI)
            if "alert_threshold_percent" in budget_config and not args.alert_threshold:
                budget.alert_threshold_percent = int(budget_config["alert_threshold_percent"])
                
            print(f"Loaded token budget configuration from {token_config.budget_config_path}")
        except Exception as e:
            print(f"Error loading token budget config from {token_config.budget_config_path}: {e}")
    
    # Fall back to trial config YAML if needed
    elif use_file_config and config.config and Path(config.config).exists():
        yaml_config = yaml.safe_load(Path(config.config).read_text())
        if "token_budget" in yaml_config:
            tb_config = yaml_config["token_budget"]
            
            # Load model limits
            if "model_limits" in tb_config:
                for model, limit in tb_config["model_limits"].items():
                    budget.quota_limits[model] = int(limit)
            
            # Load key limits
            if "key_limits" in tb_config:
                for key, limit in tb_config["key_limits"].items():
                    budget.quota_limits[key] = int(limit)
            
            # Load alert threshold (only if not already set by CLI)
            if "alert_threshold_percent" in tb_config and not args.alert_threshold:
                budget.alert_threshold_percent = int(tb_config["alert_threshold_percent"])
    
    # Print budget summary
    print(f"Token budget initialized with {len(budget.quota_limits)} limits, alert threshold: {budget.alert_threshold_percent}%")
    for entity, limit in budget.quota_limits.items():
        print(f"  - {entity}: {limit:,} tokens")
    
    # Initialize the global auditor with our configuration
    token_log_path = token_config.token_log_path or Path("runs/token_usage.jsonl")
    auditor = TokenBudgetAuditor(
        budget=budget,
        token_log_path=token_log_path,
        load_existing=True
    )
    
    # If a report path is specified, generate a report
    if token_config.budget_report_path:
        auditor.export_report(token_config.budget_report_path)

def main() -> None:
    """Entry point for running a pilot paired trial with token budget tracking."""
    import argparse
    import json
    import os
    import sys
    from bailiff.datasets import load_case_templates
    
    parser = argparse.ArgumentParser(description="Run a pilot paired trial.")
    parser.add_argument("--case", type=Path, help="Path to the case template YAML file")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--backend", type=Backend, choices=list(Backend), default=Backend.ECHO, help="LLM backend")
    parser.add_argument("--model", type=str, help="Model identifier for backend")
    parser.add_argument("--out", type=Path, help="JSONL output path")
    parser.add_argument("--placebo", action="append", default=[], dest="placebos", help="Placebo cue keys to schedule")
    parser.add_argument("--manifest", type=Path, help="Path to append run metadata")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Backend timeout seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum number of backend retries")
    parser.add_argument("--backoff-seconds", type=float, default=1.0, help="Initial backoff between retries")
    parser.add_argument("--backoff-multiplier", type=float, default=2.0, help="Multiplicative backoff factor")
    parser.add_argument("--rate-limit-seconds", type=float, default=0.0, help="Sleep between calls for rate limits")
    parser.add_argument("--backend-param", action="append", default=[], help="Backend parameter (key=value)")
    
    # Token budget CLI arguments
    parser.add_argument("--disable-budget", action="store_true", help="Disable token budget enforcement")
    parser.add_argument("--token-log", type=Path, help="Path to token usage log")
    parser.add_argument("--token-report", type=Path, help="Path for token budget report")
    parser.add_argument("--budget-config", type=Path, help="Path to custom token budget configuration file")
    
    # Direct token limit arguments
    parser.add_argument("--model-limit", action="append", nargs=2, metavar=("MODEL", "LIMIT"),
                        help="Set token limit for a model (can be used multiple times)")
    parser.add_argument("--key-limit", action="append", nargs=2, metavar=("KEY", "LIMIT"),
                        help="Set token limit for an API key (can be used multiple times)")
    parser.add_argument("--alert-threshold", type=int, default=80,
                        help="Percentage threshold for alerts (default: 80)")
    
    args = parser.parse_args()
    
    # Parse backend parameters
    backend_params = {}
    for param in args.backend_param:
        if "=" not in param:
            parser.error(f"Backend parameter must be key=value format: {param}")
        key, value = param.split("=", 1)
        backend_params[key] = value
    
    # Initialize configuration
    config = PilotConfig(
        case=args.case,
        config=args.config,
        seed=args.seed,
        backend=args.backend,
        model=args.model,
        out=args.out,
        placebos=args.placebos,
        manifest=args.manifest,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        backoff_multiplier=args.backoff_multiplier,
        rate_limit_seconds=args.rate_limit_seconds,
        backend_params=backend_params,
    )
    
    # Override token budget settings from CLI arguments
    if args.disable_budget:
        config.token.enforce_budget = False
    if args.token_log:
        config.token.token_log_path = args.token_log
    if args.token_report:
        config.token.budget_report_path = args.token_report
    if args.budget_config:
        config.token.budget_config_path = args.budget_config
    
    # Setup token budget with command line arguments
    setup_token_budget(config, args)
    
    # The rest of the main implementation goes here
    print("Token budget configured and ready to use")
    print(f"Enforcement: {'ENABLED' if config.token.enforce_budget else 'DISABLED'}")
    print(f"Log path: {config.token.token_log_path}")
    if config.token.budget_report_path:
        print(f"Report path: {config.token.budget_report_path}")

if __name__ == "__main__":
    main()
