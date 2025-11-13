"""Unit tests covering budgets, blinding, YAML loader, tagging, IO, and metrics."""
from __future__ import annotations

from pathlib import Path

import pytest

from bailiff.core.config import AgentBudget, Phase, Role
from bailiff.core.events import ObjectionRuling
from bailiff.core.io import (
    RunManifest,
    RunManifestEntry,
    append_jsonl,
    compute_prompt_hash,
    read_jsonl,
)
from bailiff.core.tokenizer import Tokenizer
from bailiff.datasets import templates as template_loader
from bailiff.metrics.outcome import PairedOutcome, flip_rate, mcnemar_log_odds
from bailiff.metrics.procedural import ShareRecord, aggregate_share
from tests.helpers import make_session, make_utterance


def test_token_budget_truncates_and_counts() -> None:
    session, _ = make_session()
    role = Role.PROSECUTION
    session._tokenizer = Tokenizer()
    session._tokens_used = {role: 0}
    session.config.agent_budgets[role] = AgentBudget(max_bytes=1000, max_tokens=3)

    truncated, tokens = session._apply_token_budget(role, "one two three four five")

    assert tokens == 3
    assert truncated.strip().split() == ["one", "two", "three"]


def test_judge_prompt_blinds_cue_values() -> None:
    session, _ = make_session()
    session.config.judge_blinding = True
    session.config.cue_value = "Alex Johnson"
    session._case_text = "Facts mention Alex Johnson and witnesses cite details."

    prompt = session._build_prompt(Role.JUDGE, Phase.OPENING)

    assert "Cue:" not in prompt
    assert "Alex Johnson" not in prompt


def test_load_case_templates_validates_payload(tmp_path: Path) -> None:
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    good = valid_dir / "valid.yaml"
    good.write_text(
        """identifier: demo
summary: Demo case
charges: ["charge"]
facts: ["{{ cue_value }} fact"]
witnesses:
  prosecution:
    - name: Test
      statement: "{{ cue_value }} statement"
  defense: []
cue_slots:
  slot: "{{ cue_value }}"
""",
        encoding="utf-8",
    )
    templates = template_loader.load_case_templates(valid_dir)
    assert [t.identifier for t in templates] == ["demo"]

    bad_dir = tmp_path / "invalid"
    bad_dir.mkdir()
    bad = bad_dir / "invalid.yaml"
    bad.write_text(
        """identifier: bad
summary: Missing placeholder
charges: ["charge"]
facts: ["fact"]
witnesses:
  prosecution: []
  defense: []
cue_slots:
  slot: "no placeholder"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        template_loader.load_case_templates(bad_dir)


def test_event_tagging_sets_objection_fields() -> None:
    session, _ = make_session()
    utterance = make_utterance("Objection, your honor! Please sustain this interruption.")

    session._apply_event_tagging(utterance)

    assert utterance.objection_raised is True
    assert utterance.objection_ruling == ObjectionRuling.SUSTAINED
    assert utterance.interruption is True


def test_manifest_append_is_idempotent(tmp_path: Path) -> None:
    manifest = RunManifest(tmp_path / "runs.jsonl")
    entry = RunManifestEntry(
        run_id="demo",
        case_identifier="case",
        model_identifier="local",
        backend="local",
        cue_name="cue",
        cue_control="Alex",
        cue_treatment="DeShawn",
        control_seed=1,
        treatment_seed=2,
        block_key="case:model",
        is_placebo=False,
        prompt_hash="hash",
    )

    manifest.append(entry)
    manifest.append(entry)

    assert len(manifest) == 1
    contents = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "trial.jsonl"
    session, log = make_session()
    log.utterances.append(make_utterance("Test content"))

    append_jsonl([log], path, validate=False)

    records = read_jsonl(path)
    assert len(records) == 1
    assert records[0]["trial_id"] == log.trial_id


def test_compute_prompt_hash_is_deterministic() -> None:
    base = compute_prompt_hash("a", "b", "c")
    same = compute_prompt_hash("a", "b", "c")
    different = compute_prompt_hash("a", "c", "b")

    assert base == same
    assert base != different


def test_flip_rate_and_mcnemar() -> None:
    pairs = [
        PairedOutcome(control=0, treatment=1),
        PairedOutcome(control=1, treatment=0),
        PairedOutcome(control=1, treatment=1),
    ]
    assert flip_rate(pairs) == pytest.approx(2 / 3)

    estimate, se = mcnemar_log_odds(pairs[:2])
    assert pytest.approx(estimate, rel=1e-6) == 0.0
    assert pytest.approx(se, rel=1e-6) == 2**0.5


def test_aggregate_share_weights_records() -> None:
    records = [
        ShareRecord(trial_id="1", phase="opening", pros_bytes=100, def_bytes=50),
        ShareRecord(trial_id="2", phase="opening", pros_bytes=20, def_bytes=40),
    ]
    score = aggregate_share(records)
    assert pytest.approx(score, rel=1e-6) == -1 / 7
