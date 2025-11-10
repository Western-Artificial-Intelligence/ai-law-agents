"""Helpers for loading and validating JSON Schemas."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Dict

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "0.2"
_SCHEMA_NAME = "trial_log.schema.json"


@lru_cache(maxsize=1)
def _load_trial_log_schema() -> Dict[str, Any]:
    data = resources.files("bailiff.schemas").joinpath(_SCHEMA_NAME).read_text(encoding="utf-8")
    return json.loads(data)


@lru_cache(maxsize=1)
def _trial_log_validator() -> Draft202012Validator:
    return Draft202012Validator(_load_trial_log_schema())


def validate_trial_log(record: Dict[str, Any]) -> None:
    """Validate a TrialLog dictionary against the JSON Schema."""

    _trial_log_validator().validate(record)
