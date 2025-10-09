"""Optional LLM backends: Groq and Gemini.

These are thin adapters around third-party SDKs. They are only imported
when selected; install extras via `pip install -e .[agent]` with the
optional packages present.
"""
from __future__ import annotations

import os
from typing import Optional


class GroqBackend:
    """Adapter for Groq chat.completions API."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        try:  # pragma: no cover - optional dep
            from groq import Groq  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("groq package not installed") from e
        self._Groq = Groq
        self._model = model
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        self._client = Groq(api_key=self._api_key)

    def __call__(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover - network
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return resp.choices[0].message.content or ""


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

