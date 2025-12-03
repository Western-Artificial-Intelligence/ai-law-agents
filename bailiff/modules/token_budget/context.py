"""Context managers for token budget management."""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Type, TypeVar, Union

from ..core.tokenizer import Tokenizer
from . import BudgetUsage, TokenBudgetAuditor, auditor as default_auditor


T = TypeVar('T')


class TokenBudgetExceededError(Exception):
    """Raised when a token budget would be exceeded."""
    def __init__(self, key: str, requested: int, remaining: int):
        self.key = key
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Token budget exceeded for '{key}': "
            f"requested {requested}, but only {remaining} remaining"
        )


@contextlib.contextmanager
def budget_context(
    key: str,
    max_tokens: int,
    auditor: Optional[TokenBudgetAuditor] = None,
    metadata: Optional[Dict[str, Any]] = None,
    raise_on_exceed: bool = True,
):
    """Context manager for token budget management.
    
    Example:
        with budget_context("my_task", max_tokens=1000) as budget:
            # Use tokens
            budget.use(100)  # Returns True if successful
            
            # Or use the decorator pattern
            @budget.use_tokens(200)
            def process_text(text: str) -> str:
                # Process text that uses tokens
                return text.upper()
    
    Args:
        key: Unique identifier for this budget
        max_tokens: Maximum tokens allowed
        auditor: TokenBudgetAuditor instance (uses default if None)
        metadata: Optional metadata for this budget
        raise_on_exceed: If True, raises TokenBudgetExceededError when budget is exceeded
    """
    auditor = auditor or default_auditor
    
    # Get or create budget
    budget = auditor.get_budget(key)
    if budget is None:
        budget = auditor.register_budget(key, max_tokens, metadata or {})
    
    # Create a context object with token usage methods
    class BudgetContext:
        def __init__(self, budget: BudgetUsage):
            self.budget = budget
            self._used_tokens = 0
        
        def use(self, tokens: int) -> bool:
            """Use tokens from the budget.
            
            Returns:
                bool: True if tokens were successfully used, False if budget would be exceeded
            """
            if tokens <= 0:
                return True
                
            success = auditor.use_tokens(key, tokens)
            if success:
                self._used_tokens += tokens
            return success
        
        def use_tokens(self, tokens: int):
            """Decorator to track token usage for a function."""
            def decorator(func):
                def wrapper(*args, **kwargs):
                    if not self.use(tokens):
                        if raise_on_exceed:
                            raise TokenBudgetExceededError(
                                key, tokens, self.budget.remaining
                            )
                        return None
                    return func(*args, **kwargs)
                return wrapper
            return decorator
        
        def count_tokens(self, text: str, model: Optional[str] = None) -> int:
            """Count tokens in text using the specified model."""
            tokenizer = Tokenizer(model) if model else Tokenizer()
            return tokenizer.count(text)
        
        @property
        def remaining(self) -> int:
            """Get remaining tokens in the budget."""
            return self.budget.remaining
        
        @property
        def used(self) -> int:
            """Get number of tokens used in this context."""
            return self._used_tokens
    
    ctx = BudgetContext(budget)
    
    try:
        yield ctx
    finally:
        # Cleanup if needed
        pass


def with_token_budget(
    key: str,
    max_tokens: int,
    auditor: Optional[TokenBudgetAuditor] = None,
    metadata: Optional[Dict[str, Any]] = None,
    raise_on_exceed: bool = True,
):
    """Decorator for token budget management.
    
    Example:
        @with_token_budget("my_task", max_tokens=1000)
        def process_text(text: str) -> str:
            # This function has a 1000 token budget
            return text.upper()
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with budget_context(
                key,
                max_tokens,
                auditor=auditor,
                metadata=metadata,
                raise_on_exceed=raise_on_exceed,
            ) as budget_ctx:
                # Pass the budget context as a keyword argument if the function accepts it
                if 'budget' in func.__code__.co_varnames:
                    kwargs['budget'] = budget_ctx
                return func(*args, **kwargs)
        return wrapper
    return decorator
