from __future__ import annotations

from pathlib import Path
import tempfile

from bailiff.metrics.tone import (
    FrozenToneClassifier,
    calibrate_frozen_tone_model,
    train_frozen_tone_classifier,
    DEFAULT_TONE_SAMPLES,
)


def test_frozen_tone_classifier_direction():
    """Test that the classifier produces correct directional scores."""
    # Train a model first
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        train_frozen_tone_classifier(model_path=model_path)
        clf = FrozenToneClassifier(model_path=model_path)
        
        positive_score = clf.score("Thank you for your patience, Your Honor.")
        negative_score = clf.score("This is outrageous and completely unacceptable!")
        assert positive_score > 0
        assert negative_score < 0


def test_calibration_pipeline_metrics():
    """Test that calibration pipeline produces valid metrics."""
    result = calibrate_frozen_tone_model(force_retrain=True)
    assert 0 <= result.ece <= 0.5
    assert 0 <= result.kappa <= 1.0
    assert result.accuracy >= 0.6
    assert len(result.raw_scores) == len(result.labels) == len(result.texts)


def test_model_reproducibility():
    """Test that the model produces consistent outputs across loads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        
        # Train model
        train_frozen_tone_classifier(model_path=model_path)
        
        # Load and score
        clf1 = FrozenToneClassifier(model_path=model_path)
        scores1 = clf1.score_many([s.text for s in DEFAULT_TONE_SAMPLES[:5]])
        probs1 = clf1.predict_proba([s.text for s in DEFAULT_TONE_SAMPLES[:5]])
        
        # Load again and score
        clf2 = FrozenToneClassifier(model_path=model_path)
        scores2 = clf2.score_many([s.text for s in DEFAULT_TONE_SAMPLES[:5]])
        probs2 = clf2.predict_proba([s.text for s in DEFAULT_TONE_SAMPLES[:5]])
        
        # Should be identical
        assert scores1 == scores2
        assert probs1 == probs2

