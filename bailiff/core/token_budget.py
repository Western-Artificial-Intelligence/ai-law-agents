"""Token Budget Auditor Module for tracking and enforcing token usage limits."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Default file locations for persistent storage
DEFAULT_TOKEN_LOG_PATH = Path("runs/token_usage.jsonl")
DEFAULT_BUDGET_CONFIG_PATH = Path("configs/token_budget.json")


@dataclass
class TokenUsage:
    """Records token usage for a specific run."""
    
    run_id: str
    model_identifier: str
    api_key_id: str  # Masked/hashed ID of the API key
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
            "total_tokens": self.total_tokens
        }


@dataclass
class TokenBudget:
    """Defines token budget constraints."""
    
    quota_limits: Dict[str, int] = field(default_factory=dict)  # Limits by model or key
    alert_threshold_percent: int = 80


class TokenBudgetAuditor:
    """Central token budget tracking and enforcement system."""
    
    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        token_log_path: Optional[Path] = None,
        load_existing: bool = True,
    ):
        """Initialize the token budget auditor.
        
        Args:
            budget: TokenBudget configuration
            token_log_path: Path to store token usage logs
            load_existing: Whether to load existing logs
        """
        self.budget = budget or self._load_default_budget()
        self.token_log_path = token_log_path or DEFAULT_TOKEN_LOG_PATH
        self._usage_records: List[TokenUsage] = []
        self._usage_by_model: Dict[str, int] = {}
        self._usage_by_key: Dict[str, int] = {}
        
        if load_existing and self.token_log_path.exists():
            self._load_usage_records()
        
        # Ensure directory exists
        if self.token_log_path.parent and not self.token_log_path.parent.exists():
            self.token_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_default_budget(self) -> TokenBudget:
        """Load budget configuration from default location or create defaults."""
        if DEFAULT_BUDGET_CONFIG_PATH.exists():
            try:
                with open(DEFAULT_BUDGET_CONFIG_PATH, "r") as f:
                    config = json.load(f)
                
                quota_limits = {}
                # Load model limits if present
                if "model_limits" in config:
                    quota_limits.update(config["model_limits"])
                # Load key limits if present
                if "key_limits" in config:
                    quota_limits.update(config["key_limits"])
                
                return TokenBudget(
                    quota_limits=quota_limits,
                    alert_threshold_percent=config.get("alert_threshold_percent", 80)
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading token budget config: {e}")
        
        # Default empty budget
        return TokenBudget()
    
    def _load_usage_records(self) -> None:
        """Load historical usage records and aggregate by model/key."""
        try:
            with open(self.token_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    record = TokenUsage(
                        run_id=data["run_id"],
                        model_identifier=data["model_identifier"],
                        api_key_id=data["api_key_id"],
                        timestamp=data["timestamp"],
                        tokens_prompt=data["tokens_prompt"],
                        tokens_completion=data["tokens_completion"]
                    )
                    self._usage_records.append(record)
                    
                    # Update aggregated usage
                    self._update_aggregated_usage(record)
        except Exception as e:
            logger.error(f"Error loading token usage records: {e}")
    
    def _update_aggregated_usage(self, record: TokenUsage) -> None:
        """Update the aggregated usage counters."""
        model_id = record.model_identifier
        key_id = record.api_key_id
        
        # Update model usage
        if model_id not in self._usage_by_model:
            self._usage_by_model[model_id] = 0
        self._usage_by_model[model_id] += record.total_tokens
        
        # Update key usage
        if key_id not in self._usage_by_key:
            self._usage_by_key[key_id] = 0
        self._usage_by_key[key_id] += record.total_tokens
    
    def record_usage(self, usage: TokenUsage) -> None:
        """Record new token usage."""
        self._usage_records.append(usage)
        self._update_aggregated_usage(usage)
        self._persist_usage(usage)
        
        # Check if we're approaching limits
        self._check_thresholds(usage)
    
    def _persist_usage(self, usage: TokenUsage) -> None:
        """Persist usage record to storage."""
        try:
            with open(self.token_log_path, "a") as f:
                f.write(json.dumps(usage.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist token usage: {e}")
    
    def _check_thresholds(self, usage: TokenUsage) -> None:
        """Check if usage is approaching configured thresholds."""
        model_id = usage.model_identifier
        key_id = usage.api_key_id
        
        # Check model quota if defined
        if model_id in self.budget.quota_limits:
            limit = self.budget.quota_limits[model_id]
            current = self._usage_by_model.get(model_id, 0)
            percent = (current / limit) * 100 if limit else 0
            
            if percent >= self.budget.alert_threshold_percent:
                logger.warning(
                    f"ALERT: Token usage for model {model_id} at "
                    f"{percent:.1f}% of quota ({current}/{limit})"
                )
        
        # Check key quota if defined
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
        estimated_tokens: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if a run should be allowed based on quota constraints.
        
        Args:
            run_id: Unique identifier for the run
            model_identifier: Model to be used
            api_key_id: API key identifier
            estimated_tokens: Estimated token usage for the run
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check model quota if defined
        if model_identifier in self.budget.quota_limits:
            limit = self.budget.quota_limits[model_identifier]
            current = self._usage_by_model.get(model_identifier, 0)
            
            if current + estimated_tokens > limit:
                return False, f"Model {model_identifier} quota would be exceeded ({current + estimated_tokens} > {limit})"
        
        # Check key quota if defined
        if api_key_id in self.budget.quota_limits:
            limit = self.budget.quota_limits[api_key_id]
            current = self._usage_by_key.get(api_key_id, 0)
            
            if current + estimated_tokens > limit:
                return False, f"API key {api_key_id} quota would be exceeded ({current + estimated_tokens} > {limit})"
        
        return True, None
    
    def generate_summary(self) -> Dict[str, object]:
        """Generate a summary of token usage."""
        # Calculate total usage
        total_tokens = sum(record.total_tokens for record in self._usage_records)
        
        # Get usage by model
        model_usage = {}
        for model, usage in self._usage_by_model.items():
            limit = self.budget.quota_limits.get(model)
            percent = (usage / limit) * 100 if limit else None
            model_usage[model] = {
                "usage": usage,
                "limit": limit,
                "percent": percent
            }
        
        # Get usage by key
        key_usage = {}
        for key, usage in self._usage_by_key.items():
            limit = self.budget.quota_limits.get(key)
            percent = (usage / limit) * 100 if limit else None
            key_usage[key] = {
                "usage": usage,
                "limit": limit,
                "percent": percent
            }
        
        return {
            "report_time": datetime.utcnow().isoformat(),
            "total_tokens": total_tokens,
            "total_runs": len(self._usage_records),
            "model_usage": model_usage,
            "key_usage": key_usage
        }
    
    def export_report(self, output_path: Optional[Path] = None) -> None:
        """Export a token usage report to JSON."""
        output_path = output_path or Path("runs/token_report.json")
        
        summary = self.generate_summary()
        
        # Add detailed records (up to a reasonable number)
        recent_records = sorted(
            self._usage_records, 
            key=lambda r: r.timestamp,
            reverse=True
        )[:100]  # Limit to most recent records
        
        detailed_records = [record.to_dict() for record in recent_records]
        
        full_report = {
            "summary": summary,
            "recent_records": detailed_records
        }
        
        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)
        
        logger.info(f"Token usage report exported to {output_path}")


# Global instance for use throughout the application
_global_auditor: Optional[TokenBudgetAuditor] = None


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
    usage = TokenUsage(
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
    estimated_tokens: int
) -> Tuple[bool, Optional[str]]:
    """Check if a run is allowed based on quota constraints."""
    return get_auditor().check_run_allowed(
        run_id=run_id,
        model_identifier=model_identifier,
        api_key_id=api_key_id,
        estimated_tokens=estimated_tokens
    )
