"""Token counting/truncation helpers for budget enforcement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore


@dataclass(slots=True)
class Tokenizer:
    """Small wrapper that prefers tiktoken but gracefully degrades.

    When `tiktoken` is installed we use its byte-pair encodings to approximate
    actual LLM tokenization. Otherwise we fall back to a simple character-count
    model so that budgets are *still* enforced even in lean environments. The
    fallback slightly over-constrains outputs (one character == one token), but
    it guarantees deterministic truncation without extra dependencies.
    """

    encoding: str = "cl100k_base"
    _encoder: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if tiktoken is not None:
            self._encoder = tiktoken.get_encoding(self.encoding)

    # Public API ---------------------------------------------------------
    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` under the active encoder."""

        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> Tuple[str, int]:
        """Truncate ``text`` to ``max_tokens`` and return (text, token_count)."""

        if max_tokens <= 0 or not text:
            return "", 0
        if self._encoder is not None:
            encoded = self._encoder.encode(text, disallowed_special=())
            if len(encoded) <= max_tokens:
                return text, len(encoded)
            clipped = encoded[:max_tokens]
            return self._encoder.decode(clipped), len(clipped)
        # Fallback path: character-based clipping
        if len(text) <= max_tokens:
            return text, len(text)
        truncated = text[:max_tokens]
        return truncated, len(truncated)
