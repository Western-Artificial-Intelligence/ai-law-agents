"""Lightweight tokenization helpers used for budget enforcement."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when unavailable
    tiktoken = None

_WORD_RE = re.compile(r"\S+")


@dataclass(slots=True)
class Tokenizer:
    """Wrapper around tiktoken with a safe fallback."""

    model: Optional[str] = None
    _encoder: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if tiktoken is None:
            return
        if self.model:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
                return
            except Exception:
                pass
        # Fall back to a generic encoding when model lookup fails.
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        """Return the number of tokens contained in ``text``."""

        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return sum(1 for _ in _WORD_RE.finditer(text))

    def clip(self, text: str, max_tokens: int) -> tuple[str, int]:
        """Clip ``text`` to ``max_tokens`` and return (clipped_text, tokens_used)."""

        if max_tokens <= 0:
            return "", 0
        if self._encoder is not None:
            tokens = self._encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text, len(tokens)
            clipped = tokens[:max_tokens]
            return self._encoder.decode(clipped), len(clipped)
        return _fallback_clip(text, max_tokens)


def _fallback_clip(text: str, max_tokens: int) -> tuple[str, int]:
    """Whitespace-aware tokenizer for environments without tiktoken."""

    if max_tokens <= 0:
        return "", 0
    count = 0
    end_index = len(text)
    for match in _WORD_RE.finditer(text):
        count += 1
        if count == max_tokens:
            end_index = match.end()
            break
    if count < max_tokens:
        return text, count
    return text[:end_index], max_tokens
