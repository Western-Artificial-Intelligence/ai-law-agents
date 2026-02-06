"""Token budget tracking for trial runs and global usage auditing."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import Role, TrialConfig

logger = logging.getLogger(__name__)

# Default file locations for persistent storage
DEFAULT_TOKEN_LOG_PATH = Path("runs/token_usage.jsonl")
DEFAULT_BUDGET_CONFIG_PATH = Path("configs/token_budget.json")


@dataclass
class TokenUsage:
    """Track token usage for a single role."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Add token usage to the current totals."""

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens


@dataclass
class TokenBudget:
    """Manage token budgets for a trial run."""

    config: TrialConfig
    usage: Dict[Role, TokenUsage] = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize usage tracking for all roles."""

        self.usage = {role: TokenUsage() for role in Role}

    def record_usage(
        self,
        role: Role,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record token usage for a role."""

        if role not in self.usage:
            self.usage[role] = TokenUsage()
        self.usage[role].add_usage(prompt_tokens, completion_tokens)
        self._check_budgets(role)

    def _check_budgets(self, role: Role) -> None:
        """Check if usage exceeds configured budgets and log alerts."""

        role_usage = self.usage[role]
        role_budget = self.config.get_role_budget(role)

        if not role_budget:
            return

        if role_budget.max_tokens is None:
            return

        if role_usage.total_tokens > role_budget.max_tokens:
            self._alert(
                level="ERROR",
                role=role,
                message=f"Token budget exceeded: {role_usage.total_tokens}/{role_budget.max_tokens}",
                usage=asdict(role_usage),
            )

    def _alert(self, level: str, role: Role, message: str, usage: dict) -> None:
        """Record an alert with usage details."""

        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "role": role.name,
            "message": message,
            "usage": usage,
        }
        self.alerts.append(alert)
        logging.log(
            getattr(logging, level, logging.WARNING),
            f"[{role.value}] {message}",
        )

    def get_summary(self) -> dict:
        """Generate a summary of token usage and alerts."""

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "trial_id": getattr(self.config, "trial_id", "unknown"),
            "usage": {
                role.name: asdict(usage)
                for role, usage in self.usage.items()
            },
            "alerts": self.alerts,
            "budgets": {
                role.name: {"max_tokens": budget.max_tokens}
                for role, budget in self.config.role_budgets.items()
            }
            if hasattr(self.config, "role_budgets")
            else {},
        }

    def save_summary(self, output_dir: Union[str, Path]) -> Path:
        """Save summary to a JSON file."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trial_id = getattr(self.config, "trial_id", "unknown")
        output_path = output_dir / f"token_usage_{trial_id}_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        return output_path


class TokenBudgetEnforcer:
    """Context manager for enforcing token budgets during trial execution."""

    def __init__(self, config: TrialConfig) -> None:
        self.budget = TokenBudget(config)
        self._original_responder = None

    def __enter__(self) -> TokenBudget:
        return self.budget

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Save summary on exit if output directory is configured
        if getattr(self.budget.config, "output_dir", None):
            try:
                output_path = self.budget.save_summary(self.budget.config.output_dir)
                logging.info(f"Token usage summary saved to {output_path}")
            except Exception as exc:
                logging.error(f"Failed to save token usage summary: {exc}")
        return False  # Don't suppress exceptions


@dataclass
class TokenUsageRecord:
    """Records token usage for a specific run."""

    run_id: str
    model_identifier: str
    api_key_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tokens_prompt: int = 0
    tokens_completion: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate the total token usage."""

        return self.tokens_prompt + self.tokens_completion

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for serialization."""

        return {
            "run_id": self.run_id,
            "model_identifier": self.model_identifier,
            "api_key_id": self.api_key_id,
            "timestamp": self.timestamp,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "total_tokens": self.total_tokens,
        }


@dataclass
class TokenQuotaBudget:
    """Defines token quota constraints by model or key."""

    quota_limits: Dict[str, int] = field(default_factory=dict)
    alert_threshold_percent: int = 80


class TokenBudgetAuditor:
    """Central token usage tracking and quota enforcement system."""

    def __init__(
        self,
        budget: Optional[TokenQuotaBudget] = None,
        token_log_path: Optional[Path] = None,
        load_existing: bool = True,
    ) -> None:
        """Initialize the token budget auditor."""

        self.budget = budget or self._load_default_budget()
        self.token_log_path = token_log_path or DEFAULT_TOKEN_LOG_PATH
        self._usage_records: List[TokenUsageRecord] = []
        self._usage_by_model: Dict[str, int] = {}
        self._usage_by_key: Dict[str, int] = {}

        if load_existing and self.token_log_path.exists():
            self._load_usage_records()

        # Ensure directory exists
        if self.token_log_path.parent and not self.token_log_path.parent.exists():
            self.token_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_default_budget(self) -> TokenQuotaBudget:
        """Load budget configuration from default location or create defaults."""

        if DEFAULT_BUDGET_CONFIG_PATH.exists():
            try:
                with open(DEFAULT_BUDGET_CONFIG_PATH, "r") as f:
                    config = json.load(f)

                quota_limits: Dict[str, int] = {}
                if "model_limits" in config:
                    quota_limits.update(config["model_limits"])
                if "key_limits" in config:
                    quota_limits.update(config["key_limits"])

                return TokenQuotaBudget(
                    quota_limits=quota_limits,
                    alert_threshold_percent=config.get("alert_threshold_percent", 80),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Error loading token budget config: {exc}")

        return TokenQuotaBudget()

    def _load_usage_records(self) -> None:
        """Load historical usage records and aggregate by model/key."""

        try:
            with open(self.token_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    record = TokenUsageRecord(
                        run_id=data["run_id"],
                        model_identifier=data["model_identifier"],
                        api_key_id=data["api_key_id"],
                        timestamp=data["timestamp"],
                        tokens_prompt=data["tokens_prompt"],
                        tokens_completion=data["tokens_completion"],
                    )
                    self._usage_records.append(record)
                    self._update_aggregated_usage(record)
        except Exception as exc:
            logger.error(f"Error loading token usage records: {exc}")

    def _update_aggregated_usage(self, record: TokenUsageRecord) -> None:
        """Update the aggregated usage counters."""

        model_id = record.model_identifier
        key_id = record.api_key_id

        self._usage_by_model[model_id] = self._usage_by_model.get(model_id, 0) + record.total_tokens
        self._usage_by_key[key_id] = self._usage_by_key.get(key_id, 0) + record.total_tokens

    def record_usage(self, usage: TokenUsageRecord) -> None:
        """Record new token usage."""

        self._usage_records.append(usage)
        self._update_aggregated_usage(usage)
        self._persist_usage(usage)
        self._check_thresholds(usage)

    def _persist_usage(self, usage: TokenUsageRecord) -> None:
        """Persist usage record to storage."""

        try:
            with open(self.token_log_path, "a") as f:
                f.write(json.dumps(usage.to_dict()) + "\n")
        except Exception as exc:
            logger.error(f"Failed to persist token usage: {exc}")

    def _check_thresholds(self, usage: TokenUsageRecord) -> None:
        """Check if usage is approaching configured thresholds."""

        model_id = usage.model_identifier
        key_id = usage.api_key_id

        if model_id in self.budget.quota_limits:
            limit = self.budget.quota_limits[model_id]
            current = self._usage_by_model.get(model_id, 0)
            percent = (current / limit) * 100 if limit else 0

            if percent >= self.budget.alert_threshold_percent:
                logger.warning(
                    f"ALERT: Token usage for model {model_id} at "
                    f"{percent:.1f}% of quota ({current}/{limit})"
                )

        if key_id in self.budget.quota_limits:
            limit = self.budget.quota_limits[key_id]
            current = self._usage_by_key.get(key_id, 0)
            percent = (current / limit) * 100 if limit else 0

            if percent >= self.budget.alert_threshold_percent:
                logger.warning(
                    f"ALERT: Token usage for API key {key_id} at "
                    f"{percent:.1f}% of quota ({current}/{limit})"
                )

    def check_run_allowed(
        self,
        run_id: str,
        model_identifier: str,
        api_key_id: str,
        estimated_tokens: int,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a run should be allowed based on quota constraints."""

        if model_identifier in self.budget.quota_limits:
            limit = self.budget.quota_limits[model_identifier]
            current = self._usage_by_model.get(model_identifier, 0)
            if current + estimated_tokens > limit:
                return (
                    False,
                    f"Model {model_identifier} quota would be exceeded ({current + estimated_tokens} > {limit})",
                )

        if api_key_id in self.budget.quota_limits:
            limit = self.budget.quota_limits[api_key_id]
            current = self._usage_by_key.get(api_key_id, 0)
            if current + estimated_tokens > limit:
                return (
                    False,
                    f"API key {api_key_id} quota would be exceeded ({current + estimated_tokens} > {limit})",
                )

        return True, None

    def generate_summary(self) -> Dict[str, object]:
        """Generate a summary of token usage."""

        total_tokens = sum(record.total_tokens for record in self._usage_records)

        model_usage = {}
        for model, usage in self._usage_by_model.items():
            limit = self.budget.quota_limits.get(model)
            percent = (usage / limit) * 100 if limit else None
            model_usage[model] = {"usage": usage, "limit": limit, "percent": percent}

        key_usage = {}
        for key, usage in self._usage_by_key.items():
            limit = self.budget.quota_limits.get(key)
            percent = (usage / limit) * 100 if limit else None
            key_usage[key] = {"usage": usage, "limit": limit, "percent": percent}

        return {
            "report_time": datetime.utcnow().isoformat(),
            "total_tokens": total_tokens,
            "total_runs": len(self._usage_records),
            "model_usage": model_usage,
            "key_usage": key_usage,
        }

    def export_report(self, output_path: Optional[Path] = None) -> None:
        """Export a token usage report to JSON."""

        output_path = output_path or Path("runs/token_report.json")

        summary = self.generate_summary()

        recent_records = sorted(
            self._usage_records,
            key=lambda record: record.timestamp,
            reverse=True,
        )[:100]

        detailed_records = [record.to_dict() for record in recent_records]

        full_report = {
            "summary": summary,
            "recent_records": detailed_records,
        }

        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)

        logger.info(f"Token usage report exported to {output_path}")


_global_auditor: Optional[TokenBudgetAuditor] = None


def set_auditor(auditor: TokenBudgetAuditor) -> None:
    """Set the global token budget auditor instance."""

    global _global_auditor
    _global_auditor = auditor


def get_auditor() -> TokenBudgetAuditor:
    """Get or initialize the global token budget auditor instance."""

    global _global_auditor
    if _global_auditor is None:
        _global_auditor = TokenBudgetAuditor()
    return _global_auditor


def register_token_usage(
    run_id: str,
    model_identifier: str,
    api_key_id: str,
    tokens_prompt: int,
    tokens_completion: int,
) -> None:
    """Register token usage with the global auditor."""

    usage = TokenUsageRecord(
        run_id=run_id,
        model_identifier=model_identifier,
        api_key_id=api_key_id,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
    )
    get_auditor().record_usage(usage)


def check_run_allowed(
    run_id: str,
    model_identifier: str,
    api_key_id: str,
    estimated_tokens: int,
) -> Tuple[bool, Optional[str]]:
    """Check if a run is allowed based on quota constraints."""

    return get_auditor().check_run_allowed(
        run_id=run_id,
        model_identifier=model_identifier,
        api_key_id=api_key_id,
        estimated_tokens=estimated_tokens,
    )
