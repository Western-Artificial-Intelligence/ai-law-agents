"""Token Budget Auditor Module.

This module provides token usage tracking, quota enforcement, and reporting
for AI model interactions.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..core.tokenizer import Tokenizer


@dataclass
class BudgetUsage:
    """Track token usage for a specific key/run."""
    
    key: str
    max_tokens: int
    used_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    @property
    def remaining(self) -> int:
        """Return remaining tokens in the budget."""
        return max(0, self.max_tokens - self.used_tokens)
    
    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.used_tokens >= self.max_tokens
    
    def use(self, tokens: int) -> bool:
        """Use tokens from the budget.
        
        Returns:
            bool: True if tokens were successfully used, False if budget would be exceeded
        """
        if self.used_tokens + tokens > self.max_tokens:
            return False
        self.used_tokens += tokens
        self.last_updated = time.time()
        return True
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        data = asdict(self)
        data["start_time"] = datetime.fromtimestamp(self.start_time).isoformat()
        data["last_updated"] = datetime.fromtimestamp(self.last_updated).isoformat()
        return data


class TokenBudgetAuditor:
    """Track and enforce token usage across multiple runs/keys."""
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize the auditor.
        
        Args:
            storage_path: Path to persist budget data (optional)
        """
        self._budgets: Dict[str, BudgetUsage] = {}
        self._storage_path = Path(storage_path) if storage_path else None
        self._load_state()
    
    def register_budget(
        self, 
        key: str, 
        max_tokens: int,
        metadata: Optional[dict] = None
    ) -> BudgetUsage:
        """Register a new budget for tracking.
        
        Args:
            key: Unique identifier for this budget
            max_tokens: Maximum tokens allowed
            metadata: Optional metadata for this budget
            
        Returns:
            The created BudgetUsage instance
        """
        if key in self._budgets:
            raise ValueError(f"Budget with key '{key}' already exists")
            
        budget = BudgetUsage(
            key=key,
            max_tokens=max_tokens,
            metadata=metadata or {}
        )
        self._budgets[key] = budget
        self._save_state()
        return budget
    
    def get_budget(self, key: str) -> Optional[BudgetUsage]:
        """Get a budget by key."""
        return self._budgets.get(key)
    
    def use_tokens(self, key: str, tokens: int) -> bool:
        """Use tokens from a budget.
        
        Returns:
            bool: True if tokens were successfully used, False if budget would be exceeded
        """
        budget = self._budgets.get(key)
        if not budget:
            raise KeyError(f"No budget found for key: {key}")
            
        success = budget.use(tokens)
        if success:
            self._save_state()
        return success
    
    def get_usage_summary(self) -> List[dict]:
        """Get summary of all budgets."""
        return [budget.to_dict() for budget in self._budgets.values()]
    
    def _save_state(self) -> None:
        """Persist current state to storage."""
        if not self._storage_path:
            return
            
        data = {
            "version": "1.0",
            "budgets": [budget.to_dict() for budget in self._budgets.values()]
        }
        
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_state(self) -> None:
        """Load state from storage."""
        if not (self._storage_path and self._storage_path.exists()):
            return
            
        try:
            with open(self._storage_path, 'r') as f:
                data = json.load(f)
                
            for budget_data in data.get("budgets", []):
                # Convert ISO format timestamps back to timestamps
                budget_data["start_time"] = datetime.fromisoformat(budget_data["start_time"]).timestamp()
                budget_data["last_updated"] = datetime.fromisoformat(budget_data["last_updated"]).timestamp()
                self._budgets[budget_data["key"]] = BudgetUsage(**budget_data)
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # If loading fails, start with empty state
            pass


# Global instance for convenience
auditor = TokenBudgetAuditor()
