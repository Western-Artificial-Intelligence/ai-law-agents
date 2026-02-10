"""JSONL serialization utilities for TrialLog artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence

from .events import TrialLog, UtteranceLog, ObjectionRuling
from .config import Phase, Role
from .schema import validate_trial_log

_DEFAULT_VALIDATE = os.getenv("BAILIFF_VALIDATE_LOGS", "1").lower() not in {"0", "false", "no"}
_FILE_LOCK_GUARD = Lock()
_FILE_LOCKS: Dict[str, Lock] = {}


def _should_validate(flag: Optional[bool]) -> bool:
    return _DEFAULT_VALIDATE if flag is None else flag


def _lock_for_path(path: Path) -> Lock:
    key = str(path.resolve())
    with _FILE_LOCK_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _FILE_LOCKS[key] = lock
        return lock


@dataclass
class RunManifestEntry:
    """Record describing one paired run emitted by the batch driver."""

    run_id: str
    case_identifier: str
    model_identifier: str
    backend: str
    cue_name: str
    cue_control: str
    cue_treatment: str
    control_seed: int
    treatment_seed: int
    block_key: Optional[str]
    is_placebo: bool
    prompt_hash: str
    prompt_hash_control: Optional[str] = None
    prompt_hash_treatment: Optional[str] = None
    params: Dict[str, object] = field(default_factory=dict)
    trial_ids: Sequence[str] = field(default_factory=tuple)
    log_path: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None
    retries: int = 0


class RunManifest:
    """Append-only manifest of executed paired runs (supports resume)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self._lock = Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                run_id = data.get("run_id")
                if isinstance(run_id, str):
                    self._seen.add(run_id)

    def has_run(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._seen

    def append(self, entry: RunManifestEntry) -> None:
        with self._lock:
            if entry.run_id in self._seen:
                return
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), default=_encode, ensure_ascii=False) + "\n")
            self._seen.add(entry.run_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)


def compute_prompt_hash(*components: str) -> str:
    """Return a deterministic hash across multiple prompt components."""

    h = hashlib.sha256()
    for comp in components:
        h.update(comp.encode("utf-8"))
    return h.hexdigest()


def _format_datetime_rfc3339(dt: datetime) -> str:
    """Format a datetime object as RFC 3339 compliant string."""
    if dt.tzinfo is None:
        # Naive datetime - assume UTC and append 'Z' for RFC 3339
        return dt.isoformat() + "Z"
    # Timezone-aware datetime - use isoformat and replace +00:00 with Z for UTC
    formatted = dt.isoformat()
    if formatted.endswith("+00:00"):
        return formatted[:-6] + "Z"
    return formatted


def _encode(obj):  # type: ignore[override]
    if isinstance(obj, datetime):
        return _format_datetime_rfc3339(obj)
    if isinstance(obj, (Role, Phase, ObjectionRuling)):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _convert_datetimes_recursive(obj: object) -> object:
    """Recursively convert datetime objects in nested structures to RFC 3339 strings."""
    if isinstance(obj, datetime):
        return _format_datetime_rfc3339(obj)
    elif isinstance(obj, dict):
        return {key: _convert_datetimes_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_datetimes_recursive(item) for item in obj)
    else:
        return obj


def write_jsonl(logs: Iterable[TrialLog], path: Path, *, validate: Optional[bool] = None) -> None:
    validate_flag = _should_validate(validate)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_for_path(path)
    with lock:
        with path.open("w", encoding="utf-8") as f:
            for log in logs:
                rec = asdict(log)
                # Recursively convert datetime objects to RFC 3339 strings
                rec = _convert_datetimes_recursive(rec)
                if validate_flag:
                    validate_trial_log(rec)
                f.write(json.dumps(rec, default=_encode, ensure_ascii=False) + "\n")


def append_jsonl(logs: Iterable[TrialLog], path: Path, *, validate: Optional[bool] = None) -> None:
    """Append logs to an existing JSONL file (creating it if missing)."""

    validate_flag = _should_validate(validate)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_for_path(path)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            for log in logs:
                rec = asdict(log)
                # Recursively convert datetime objects to RFC 3339 strings
                rec = _convert_datetimes_recursive(rec)
                if validate_flag:
                    validate_trial_log(rec)
                f.write(json.dumps(rec, default=_encode, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    """Return raw dicts (not re-hydrated dataclasses) for analysis."""

    objs: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            objs.append(json.loads(line))
    return objs
