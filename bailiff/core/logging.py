"""Factories and utilities for structured trial logs."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from .config import TrialConfig
from .events import TrialLog


def default_log_factory(config: TrialConfig) -> TrialLog:
    """Instantiate a trial log with metadata populated from the config."""

    started = datetime.utcnow()
    return TrialLog(
        trial_id=f"{config.case_template.stem}-{config.cue.name}-{config.seed}",
        case_identifier=config.case_template.stem,
        model_identifier=config.model_identifier,
        cue_assignment=config.cue.name,
        seed=config.seed,
        started_at=started,
        completed_at=None,
    )


def mark_completed(log: TrialLog) -> None:
    """Set the completion timestamp if not already done."""

    if log.completed_at is None:
        log.completed_at = datetime.utcnow()


LogFactory = Callable[[TrialConfig], TrialLog]
