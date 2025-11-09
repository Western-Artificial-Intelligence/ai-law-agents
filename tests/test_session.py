"""Tests for TrialSession parsing helpers."""
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
