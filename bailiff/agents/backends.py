"""Optional LLM backends: Groq and Gemini with token budget tracking."""
from __future__ import annotations

import hashlib
import os
import uuid
from typing import Dict, List, Optional

from ..core.token_budget_simplified import check_run_allowed, register_token_usage
from .groq_pool import GroqKeyPool


def _hash_api_key(key: str) -> str:
    """Create a masked identifier for an API key for tracking purposes."""
    if not key:
        return "unknown"
    h = hashlib.md5(key.encode()).hexdigest()
    return f"{key[:4]}...{h[:8]}"


class GroqBackend:
    """Adapter for Groq chat.completions API with key rotation and token tracking."""

    def __init__(self, model: str, api_key: Optional[str] = None, api_keys: Optional[List[str]] = None, enforce_budget: bool = True):
        try:  # pragma: no cover - optional dep
            from groq import Groq  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("groq package not installed") from e
        self._Groq = Groq
        self._model = model
        explicit_keys = [api_key] if api_key else api_keys
        self._pool = GroqKeyPool.from_env(api_keys=explicit_keys)
        self._clients: Dict[str, object] = {}
        self._enforce_budget = enforce_budget
        self._run_id = str(uuid.uuid4())

    def __call__(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover - network
        """Make a chat completion request with token tracking."""
        import time
        
        # Estimate token usage
        estimated_prompt_tokens = len(prompt) // 4  # Simple heuristic
        estimated_completion_tokens = kwargs.get("max_tokens", 512)
        estimated_total = estimated_prompt_tokens + estimated_completion_tokens
        
        while True:
            try:
                # Try to acquire a key, waiting if necessary
                with self._pool.acquire(wait=True, max_wait=30.0) as lease:
                    # Check token budget before proceeding
                    if self._enforce_budget:
                        key_id = _hash_api_key(lease.key)
                        allowed, reason = check_run_allowed(
                            run_id=self._run_id,
                            model_identifier=self._model,
                            api_key_id=key_id,
                            estimated_tokens=estimated_total
                        )
                        
                        if not allowed:
                            raise RuntimeError(f"Token budget exceeded: {reason}")
                    
                    client = self._clients.get(lease.key)
                    if client is None:
                        client = self._Groq(api_key=lease.key)
                        self._clients[lease.key] = client
                    try:
                        resp = client.chat.completions.create(
                            model=self._model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=kwargs.get("temperature", 0.2),
                            max_tokens=kwargs.get("max_tokens", 512),
                        )
                        lease.mark_success()
                        
                        # Record token usage
                        if self._enforce_budget:
                            # Get actual usage from response
                            usage = getattr(resp, "usage", None)
                            prompt_tokens = getattr(usage, "prompt_tokens", estimated_prompt_tokens)
                            completion_tokens = getattr(usage, "completion_tokens", len(resp.choices[0].message.content) // 4)
                            
                            # Register with token budget auditor
                            register_token_usage(
                                run_id=self._run_id,
                                model_identifier=self._model,
                                api_key_id=_hash_api_key(lease.key),
                                tokens_prompt=prompt_tokens,
                                tokens_completion=completion_tokens
                            )
                        
                        return resp.choices[0].message.content or ""
                    except Exception as exc:  # pragma: no cover - network
                        if _is_rate_limit_error(exc):
                            lease.mark_rate_limited(exc)
                            # The backoff is already set in the key status, so we'll wait
                            # when trying to acquire the next key. Just continue the loop.
                        else:
                            lease.mark_failure(exc)
                            # For non-rate-limit errors, wait a bit before retry
                            time.sleep(1.0)
                        # Continue loop to retry with a different key
            except RuntimeError:
                # No keys available - wait and retry
                time.sleep(1.0)
                continue


class GeminiBackend:
    """Adapter for Google Gemini via `google-generativeai` with token tracking."""

    def __init__(self, model: str, api_key: Optional[str] = None, enforce_budget: bool = True):
        try:  # pragma: no cover - optional dep
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("google-generativeai package not installed") from e
        self._genai = genai
        self._model_name = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(model)
        self._enforce_budget = enforce_budget
        self._run_id = str(uuid.uuid4())

    def __call__(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover - network
        # Create a masked version of the API key for tracking
        key_id = _hash_api_key(self._api_key)
        
        # Estimate token usage for budget enforcement
        estimated_prompt_tokens = len(prompt) // 4
        estimated_completion_tokens = 500  # Default estimate
        estimated_total = estimated_prompt_tokens + estimated_completion_tokens
        
        if self._enforce_budget:
            allowed, reason = check_run_allowed(
                run_id=self._run_id,
                model_identifier=self._model_name,
                api_key_id=key_id,
                estimated_tokens=estimated_total
            )
            
            if not allowed:
                raise RuntimeError(f"Token budget exceeded: {reason}")
        
        try:
            resp = self._model.generate_content(prompt)
            
            # Get token counts if available
            usage = getattr(resp, "usage", None)
            tokens_prompt = getattr(usage, "prompt_token_count", estimated_prompt_tokens)
            tokens_completion = getattr(usage, "candidates_token_count", len(getattr(resp, "text", "")) // 4)
            
            # Register usage with the token budget auditor
            if self._enforce_budget:
                register_token_usage(
                    run_id=self._run_id,
                    model_identifier=self._model_name,
                    api_key_id=key_id,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion
                )
            
            return getattr(resp, "text", "") or ""
        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                # Handle rate limits by failing gracefully
                raise RuntimeError(f"API rate limit or quota exceeded: {e}")
            raise


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of Groq rate limit responses."""

    if exc.__class__.__name__ == "RateLimitError":
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    message = str(exc)
    return "429" in message or "rate limit" in message.lower()
