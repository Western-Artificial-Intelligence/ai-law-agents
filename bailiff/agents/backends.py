"""Optional LLM backends: Groq and Gemini.

These are thin adapters around third-party SDKs. They are only imported
when selected; install extras via `pip install -e .[agent]` with the
optional packages present.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from .groq_pool import GroqKeyPool


class GroqBackend:
    """Adapter for Groq chat.completions API with key rotation and backoff."""

    def __init__(self, model: str, api_key: Optional[str] = None, api_keys: Optional[List[str]] = None):
        try:  # pragma: no cover - optional dep
            from groq import Groq  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("groq package not installed") from e
        self._Groq = Groq
        self._model = model
        explicit_keys = [api_key] if api_key else api_keys
        self._pool = GroqKeyPool.from_env(api_keys=explicit_keys)
        self._clients: Dict[str, object] = {}

    def __call__(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover - network
        """Make a chat completion request with automatic retry on rate limits.
        
        Retries infinitely with exponential backoff (capped at 30s) until a request succeeds.
        Automatically rotates through available keys and waits for backoff periods to expire.
        """
        import time
        
        while True:
            try:
                # Try to acquire a key, waiting if necessary
                with self._pool.acquire(wait=True, max_wait=30.0) as lease:
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
    """Adapter for Google Gemini via `google-generativeai`."""

    def __init__(self, model: str, api_key: Optional[str] = None):
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

    def __call__(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover - network
        resp = self._model.generate_content(prompt)
        return getattr(resp, "text", "") or ""


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of Groq rate limit responses."""

    if exc.__class__.__name__ == "RateLimitError":
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    message = str(exc)
    return "429" in message or "rate limit" in message.lower()

