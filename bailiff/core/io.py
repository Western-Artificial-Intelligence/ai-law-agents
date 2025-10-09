"""JSONL serialization utilities for TrialLog artifacts."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from .events import TrialLog, UtteranceLog, ObjectionRuling
from .config import Phase, Role


def _encode(obj):  # type: ignore[override]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (Role, Phase, ObjectionRuling)):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_jsonl(logs: Iterable[TrialLog], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for log in logs:
            rec = asdict(log)
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

