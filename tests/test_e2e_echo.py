"""Integration test: run echo backend end-to-end and validate schema basics."""
from bailiff.cli import e2e_validate_echo


def test_e2e_echo_smoke():
    """Run the validator and ensure it returns two TrialLog objects with expected fields."""
    logs = e2e_validate_echo(seed=42)
    assert isinstance(logs, list) and len(logs) == 2
    # basic extra sanity checks
    for log in logs:
        assert log.model_identifier == "echo"
        assert log.cue_name == "name_ethnicity"
        assert log.cue_condition in ("control", "treatment")
        assert len(log.utterances) > 0
        # first utterance content is non-empty
        assert getattr(log.utterances[0], "content")
