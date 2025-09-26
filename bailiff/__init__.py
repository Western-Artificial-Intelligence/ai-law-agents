"""Top-level package exports for the B.A.I.L.I.F.F. research scaffold."""

from .core.session import TrialSession
from .core.config import CueToggle, TrialConfig
from .orchestration.pipeline import TrialPipeline

__all__ = [
    "CueToggle",
    "TrialConfig",
    "TrialSession",
    "TrialPipeline",
]
