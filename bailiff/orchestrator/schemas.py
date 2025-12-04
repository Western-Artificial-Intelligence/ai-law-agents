from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Backend(str, Enum):
    ECHO = "echo"
    GROQ = "groq"
    GEMINI = "gemini"
    LOCAL = "local"

class PilotTrialRequest(BaseModel):
    """Request model for running a pilot trial, mirroring CLI arguments."""
    case: Optional[str] = Field(None, description="Path to the case template file")
    config: Optional[str] = Field(None, description="Path to YAML config file")
    seed: int = Field(42, description="Base random seed")
    backend: Backend = Field(Backend.ECHO, description="LLM backend to use")
    model: Optional[str] = Field(None, description="Model identifier for backend")
    out: Optional[str] = Field(None, description="Optional JSONL output path")
    manifest: Optional[str] = Field(None, description="Optional manifest path to append run metadata")
    placebos: List[str] = Field(default_factory=list, description="Placebo cue keys to schedule")
    timeout_seconds: float = Field(30.0, description="Backend timeout in seconds")
    max_retries: int = Field(2, description="Maximum number of backend retries")
    backoff_seconds: float = Field(1.0, description="Initial backoff between retries")
    backoff_multiplier: float = Field(2.0, description="Multiplicative backoff factor")
    rate_limit_seconds: float = Field(0.0, description="Sleep between calls to respect rate limits")
    backend_params: Dict[str, object] = Field(default_factory=dict, description="Backend parameter overrides")

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(BaseModel):
    id: str
    status: JobStatus
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    request: PilotTrialRequest
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
