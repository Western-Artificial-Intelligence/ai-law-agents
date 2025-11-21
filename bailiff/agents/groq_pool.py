"""Groq API key pool with rotation, concurrency limits, and backoff."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(slots=True)
class GroqKeyStatus:
    """Tracks usage statistics and throttling state for a Groq API key."""

    key: str
    max_concurrency: int
    total_uses: int = 0
    inflight: int = 0
    consecutive_rate_limits: int = 0
    last_error: Optional[str] = None
    backoff_until: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def is_available(self, now: Optional[float] = None) -> bool:
        """Return True if the key can accept another request."""

        current = now if now is not None else time.monotonic()
        return self.inflight < self.max_concurrency and current >= self.backoff_until

    def snapshot(self) -> Dict[str, object]:
        """Produce a serializable summary for logging or debugging."""

        return {
            "key": self.key[-6:],  # avoid dumping full secrets
            "max_concurrency": self.max_concurrency,
            "inflight": self.inflight,
            "total_uses": self.total_uses,
            "consecutive_rate_limits": self.consecutive_rate_limits,
            "backoff_remaining": max(self.backoff_until - time.monotonic(), 0),
            "last_error": self.last_error,
        }


class GroqKeyLease:
    """Context manager that represents a reserved Groq API key."""

    def __init__(self, pool: "GroqKeyPool", status: GroqKeyStatus):
        self._pool = pool
        self._status = status
        self.key = status.key
        self._released = False

    def mark_success(self) -> None:
        self._release(success=True)

    def mark_failure(self, error: Exception) -> None:
        self._release(success=False, error=error, rate_limited=False)

    def mark_rate_limited(self, error: Exception) -> None:
        self._release(success=False, error=error, rate_limited=True)

    def _release(self, success: bool, error: Optional[Exception] = None, rate_limited: bool = False) -> None:
        if self._released:
            return
        self._pool._release(self._status, success=success, error=error, rate_limited=rate_limited)
        self._released = True

    def __enter__(self) -> "GroqKeyLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._released:
            return
        if exc is None:
            self.mark_success()
        elif exc_type and issubclass(exc_type, Exception):
            self.mark_failure(exc)  # pragma: no cover - defensive


class GroqKeyPool:
    """Selects the least-used Groq API key while respecting concurrency and backoff."""

    MAX_BACKOFF_SECONDS = 30

    def __init__(self, statuses: Iterable[GroqKeyStatus]):
        statuses = list(statuses)
        if not statuses:
            raise ValueError("At least one Groq API key must be provided")
        self._statuses = {status.key: status for status in statuses}
        self._order = {status.key: idx for idx, status in enumerate(statuses)}
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        api_keys: Optional[List[str]] = None,
        concurrency_map: Optional[Dict[str, int]] = None,
        default_concurrency: Optional[int] = None,
    ) -> "GroqKeyPool":
        """Build the pool from environment variables."""

        if api_keys is None:
            api_keys = _load_key_list()
        if concurrency_map is None:
            concurrency_map = _load_concurrency_map()
        if default_concurrency is None:
            default_concurrency = _load_default_concurrency()
        statuses = [
            GroqKeyStatus(key=key, max_concurrency=concurrency_map.get(key, default_concurrency))
            for key in api_keys
        ]
        return cls(statuses)

    def acquire(self, wait: bool = False, max_wait: float = 30.0) -> GroqKeyLease:
        """Reserve the least-used key that is currently available.
        
        Args:
            wait: If True and no keys are available, wait until one becomes available
            max_wait: Maximum seconds to wait (capped at 30.0)
        """
        max_wait = min(max_wait, 30.0)
        
        # First try without waiting
        with self._lock:
            try:
                status = self._select_next()
                status.inflight += 1
                return GroqKeyLease(self, status)
            except RuntimeError:
                if not wait:
                    raise
        
        # Need to wait - release lock and wait
        start = time.monotonic()
        while time.monotonic() - start < max_wait:
            # Check which key will be available soonest
            with self._lock:
                now = time.monotonic()
                candidates = [
                    (status, max(status.backoff_until - now, 0))
                    for status in self._statuses.values()
                    if status.inflight < status.max_concurrency
                ]
                if not candidates:
                    # All keys at capacity, wait a bit and retry
                    wait_time = 0.1
                else:
                    # Sort by backoff remaining, then by usage
                    candidates.sort(key=lambda x: (x[1], x[0].total_uses, x[0].inflight))
                    status, backoff_remaining = candidates[0]
                    if backoff_remaining <= 0 and status.is_available(now):
                        # Key is available now
                        status.inflight += 1
                        return GroqKeyLease(self, status)
                    # Wait for backoff or a short delay
                    wait_time = min(backoff_remaining + 0.1, max_wait - (now - start), 0.5)
            
            # Release lock while sleeping
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Try to acquire again
            with self._lock:
                try:
                    status = self._select_next()
                    status.inflight += 1
                    return GroqKeyLease(self, status)
                except RuntimeError:
                    # Still no keys available, continue waiting
                    if time.monotonic() - start >= max_wait:
                        # Final attempt
                        status = self._select_next()
                        status.inflight += 1
                        return GroqKeyLease(self, status)
                    continue

    @property
    def keys(self) -> List[str]:
        """Return the ordered list of managed keys."""

        with self._lock:
            return list(self._statuses.keys())

    def summary(self) -> List[Dict[str, object]]:
        """Return a snapshot of each key's status for logging/debugging."""

        with self._lock:
            return [status.snapshot() for status in self._statuses.values()]

    def _select_next(self) -> GroqKeyStatus:
        now = time.monotonic()
        eligible = [status for status in self._statuses.values() if status.is_available(now)]
        if not eligible:
            raise RuntimeError("No Groq API keys are currently available (all throttled or at capacity)")
        eligible.sort(key=lambda s: (s.total_uses, s.inflight, self._order[s.key]))
        return eligible[0]

    def _release(self, status: GroqKeyStatus, success: bool, error: Optional[Exception], rate_limited: bool) -> None:
        with self._lock:
            if status.inflight > 0:
                status.inflight -= 1
            if success:
                status.total_uses += 1
                status.consecutive_rate_limits = 0
                status.last_error = None
                status.backoff_until = 0.0
            else:
                status.last_error = str(error) if error else None
                if rate_limited:
                    status.consecutive_rate_limits += 1
                    delay = min(2 ** status.consecutive_rate_limits, self.MAX_BACKOFF_SECONDS)
                    status.backoff_until = time.monotonic() + delay
                else:
                    status.consecutive_rate_limits = 0


def _load_key_list() -> List[str]:
    """Load Groq API keys from environment variables with fallback."""

    raw_list = os.getenv("GROQ_API_KEYS")
    if raw_list:
        try:
            parsed = json.loads(raw_list)
        except json.JSONDecodeError as err:  # pragma: no cover - defensive
            raise ValueError("GROQ_API_KEYS must be valid JSON") from err
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("GROQ_API_KEYS must be a JSON list of strings")
        keys = [item.strip() for item in parsed if item.strip()]
        if not keys:
            raise ValueError("GROQ_API_KEYS cannot be empty")
        return keys

    single = os.getenv("GROQ_API_KEY")
    if single:
        return [single.strip()]
    raise RuntimeError("Set GROQ_API_KEYS (JSON list) or GROQ_API_KEY in the environment")


def _load_concurrency_map() -> Dict[str, int]:
    """Parse per-key concurrency overrides from env."""

    raw_map = os.getenv("GROQ_API_KEY_CONCURRENCY")
    if not raw_map:
        return {}
    try:
        parsed = json.loads(raw_map)
    except json.JSONDecodeError as err:  # pragma: no cover - defensive
        raise ValueError("GROQ_API_KEY_CONCURRENCY must be valid JSON") from err
    if not isinstance(parsed, dict):
        raise ValueError("GROQ_API_KEY_CONCURRENCY must be a JSON object")
    result: Dict[str, int] = {}
    for key, value in parsed.items():
        try:
            result[key] = int(value)
        except (TypeError, ValueError) as err:  # pragma: no cover - defensive
            raise ValueError("Concurrency limits must be integers") from err
        if result[key] <= 0:
            raise ValueError("Concurrency limits must be positive integers")
    return result


def _load_default_concurrency() -> int:
    """Read the default concurrency fallback from env."""

    raw_default = os.getenv("DEFAULT_MAX_CONCURRENCY", "1")
    try:
        value = int(raw_default)
    except ValueError as err:  # pragma: no cover - defensive
        raise ValueError("DEFAULT_MAX_CONCURRENCY must be an integer") from err
    if value <= 0:
        raise ValueError("DEFAULT_MAX_CONCURRENCY must be greater than zero")
    return value

