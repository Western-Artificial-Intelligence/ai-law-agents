"""Token budget tracking and enforcement for trial runs."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from .config import Role, TrialConfig


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
    
    def __post_init__(self):
        """Initialize usage tracking for all roles."""
        self.usage = {role: TokenUsage() for role in Role}
    
    def record_usage(
        self, 
        role: Role, 
        prompt_tokens: int, 
        completion_tokens: int
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
            
        if role_usage.total_tokens > role_budget.max_tokens:
            self._alert(
                level="ERROR",
                role=role,
                message=f"Token budget exceeded: {role_usage.total_tokens}/{role_budget.max_tokens}",
                usage=asdict(role_usage)
            )
    
    def _alert(self, level: str, role: Role, message: str, usage: dict) -> None:
        """Record an alert with usage details."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "role": role.value,
            "message": message,
            "usage": usage
        }
        self.alerts.append(alert)
        logging.log(
            getattr(logging, level, logging.WARNING),
            f"[{role.value}] {message}"
        )
    
    def get_summary(self) -> dict:
        """Generate a summary of token usage and alerts."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "trial_id": getattr(self.config, "trial_id", "unknown"),
            "usage": {
                role.value: asdict(usage) 
                for role, usage in self.usage.items()
            },
            "alerts": self.alerts,
            "budgets": {
                role.value: {"max_tokens": budget.max_tokens} 
                for role, budget in self.config.role_budgets.items()
            } if hasattr(self.config, "role_budgets") else {}
        }
    
    def save_summary(self, output_dir: Union[str, Path]) -> Path:
        """Save summary to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trial_id = getattr(self.config, "trial_id", "unknown")
        output_path = output_dir / f"token_usage_{trial_id}_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
            
        return output_path


class TokenBudgetEnforcer:
    """Context manager for enforcing token budgets during trial execution."""
    
    def __init__(self, config: TrialConfig):
        self.budget = TokenBudget(config)
        self._original_responder = None
    
    def __enter__(self):
        return self.budget
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save summary on exit if output directory is configured
        if hasattr(self.budget.config, 'output_dir'):
            try:
                output_path = self.budget.save_summary(self.budget.config.output_dir)
                logging.info(f"Token usage summary saved to {output_path}")
            except Exception as e:
                logging.error(f"Failed to save token usage summary: {e}")
        return False  # Don't suppress exceptions
