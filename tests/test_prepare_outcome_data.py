"""Tests for outcome CSV preparation helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("scripts/prepare_outcome_data.py")
    spec = importlib.util.spec_from_file_location("prepare_outcome_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_pairs_keeps_exactly_one_control_and_treatment():
    mod = _load_module()
    rows = [
        {"pair_id": "p1", "cue_condition": "control", "trial_id": "c-old"},
        {"pair_id": "p1", "cue_condition": "control", "trial_id": "c-new"},  # duplicate control
        {"pair_id": "p1", "cue_condition": "treatment", "trial_id": "t1"},
        {"pair_id": "p2", "cue_condition": "control", "trial_id": "c2"},  # incomplete pair
        {"pair_id": "p3", "cue_condition": "treatment", "trial_id": "t3"},  # incomplete pair
    ]

    out = mod.normalize_pairs(rows)

    assert len(out) == 2
    p1 = [r for r in out if r["pair_id"] == "p1"]
    assert len(p1) == 2
    assert {r["cue_condition"] for r in p1} == {"control", "treatment"}
    # Latest duplicate for a condition should win.
    assert any(r["trial_id"] == "c-new" for r in p1)
