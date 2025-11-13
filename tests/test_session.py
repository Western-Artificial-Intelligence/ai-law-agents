"""Tests for TrialSession parsing helpers."""
from datetime import datetime

from bailiff.core.config import Phase, Role
from bailiff.core.events import TrialLog, UtteranceLog
from bailiff.core.session import TrialSession


def test_extract_verdict_fields_from_json_block():
    text = '{"verdict":"guilty","sentence":24}'
    verdict, sentence = TrialSession._extract_verdict_fields(text)
    assert verdict == "guilty"
    assert sentence == "24"


def test_extract_verdict_fields_from_embedded_json():
    text = """
    The court now renders judgment.
    {
        "verdict": "not_guilty",
        "sentence": 0
    }
    Further explanation follows.
    """
    verdict, sentence = TrialSession._extract_verdict_fields(text)
    assert verdict == "not_guilty"
    assert sentence == "0"


def test_extract_verdict_fields_uses_regex_fallback():
    text = "I find the defendant not guilty. The sentence is 12 months of probation."
    verdict, sentence = TrialSession._extract_verdict_fields(text)
    assert verdict == "not_guilty"
    assert sentence == "12"


def test_extract_verdict_fields_handles_missing_sentence_in_json():
    text = '{"verdict": "guilty"} Additional findings and rationale.'
    verdict, sentence = TrialSession._extract_verdict_fields(text)
    assert verdict == "guilty"
    assert sentence is None


def test_parse_and_set_verdict_sentence_sets_log_fields():
    """Integration test: ensure judge JSON is plumbed into TrialLog metadata."""
    session = TrialSession.__new__(TrialSession)
    now = datetime.utcnow()
    log = TrialLog(
        trial_id="trial-001",
        case_identifier="case-001",
        model_identifier="echo",
        cue_name="cue",
        cue_condition="control",
        cue_value="Alex Johnson",
        seed=7,
        started_at=now,
        completed_at=now,
    )
    judge_record = UtteranceLog(
        role=Role.JUDGE,
        phase=Phase.VERDICT,
        content='{"verdict": "guilty", "sentence": 12}',
        byte_count=36,
        token_count=None,
        addressed_to=None,
        timestamp=now,
    )
    log.append(judge_record)
    session._log = log

    session._parse_and_set_verdict_sentence()

    assert log.verdict == "guilty"
    assert log.sentence == "12"
