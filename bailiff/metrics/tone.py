"""Lightweight tone scoring, calibration, and reliability utilities.

This module provides minimal, dependency-light helpers:
- naive_lexicon_score: simple polarity score in [-1, 1]
- FrozenToneClassifier: frozen logistic regression classifier with Platt calibration
- calibrate_frozen_tone_model: reproducible pipeline returning metrics/output
- fit_platt / apply_platt: logistic calibration of scores
- cohen_kappa: inter-rater reliability
- expected_calibration_error: ECE across bins
- Frozen tone classifier stored under bailiff/metrics/models/frozen_tone_classifier.pkl
- Delete and rerun calibration to retrain if necessary
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple
import math
import string
import pickle

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


_POS = {"good", "polite", "respect", "calm", "measured", "professional"}
_NEG = {"bad", "rude", "hostile", "aggressive", "angry", "unprofessional"}
_POLITE = {"please", "thanks", "thank", "appreciate", "grateful", "kindly"}
_INTENSE = {"furious", "outraged", "disgraceful", "ridiculous", "unacceptable", "angry"}
_HEDGES = {"perhaps", "maybe", "somewhat", "slightly", "generally", "usually"}
_RESPECT_TITLES = {"sir", "madam", "your", "honor"}


def naive_lexicon_score(text: str) -> float:
    tokens = {t.strip(".,:;!?()[]{}\"\'").lower() for t in text.split()}
    pos = len(tokens & _POS)
    neg = len(tokens & _NEG)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


@dataclass
class PlattParams:
    a: float
    b: float


def fit_platt(scores: Iterable[float], labels: Iterable[int]) -> PlattParams:
    # Simple logistic regression with two parameters via Newton steps
    x = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    a, b = 0.0, 0.0
    for _ in range(25):
        z = a * x + b
        p = 1 / (1 + np.exp(-z))
        # gradient
        g_a = np.sum((p - y) * x)
        g_b = np.sum(p - y)
        # Hessian
        w = p * (1 - p)
        h_aa = np.sum(w * x * x) + 1e-8
        h_bb = np.sum(w) + 1e-8
        h_ab = np.sum(w * x)
        # Newton step on 2x2
        det = h_aa * h_bb - h_ab * h_ab + 1e-12
        da = (-g_a * h_bb + g_b * h_ab) / det
        db = (-g_b * h_aa + g_a * h_ab) / det
        a += da
        b += db
        if abs(da) + abs(db) < 1e-6:
            break
    return PlattParams(a, b)


def apply_platt(scores: Iterable[float], params: PlattParams) -> Tuple[float, ...]:
    x = np.asarray(list(scores), dtype=float)
    z = params.a * x + params.b
    p = 1 / (1 + np.exp(-z))
    return tuple(float(v) for v in p)


def cohen_kappa(a: Iterable[int], b: Iterable[int]) -> float:
    a = np.asarray(list(a), dtype=int)
    b = np.asarray(list(b), dtype=int)
    assert a.size == b.size and a.size > 0
    pa = float(np.mean(a == b))
    p_yes = (np.mean(a == 1) * np.mean(b == 1)) + (np.mean(a == 0) * np.mean(b == 0))
    denom = 1 - p_yes
    return 0.0 if denom == 0 else (pa - p_yes) / denom


def expected_calibration_error(probs: Iterable[float], labels: Iterable[int], bins: int = 10) -> float:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    assert p.size == y.size and p.size > 0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        w = float(np.mean(mask))
        ece += w * abs(conf - acc)
    return ece


def _normalize_tokens(text: str) -> Tuple[str, ...]:
    """Normalize text into tokens for feature extraction."""
    tokens = []
    for raw in text.split():
        token = raw.strip(string.punctuation)
        if not token:
            continue
        token = token.lower()
        tokens.append(token)
    return tuple(tokens)


def extract_features(text: str) -> np.ndarray:
    """Extract feature vector from text for tone classification.
    
    Returns a 10-element feature vector:
    [pos_lexicon_count, neg_lexicon_count, polite_count, intense_count,
     hedge_count, titles_count, exclaim_count, question_count,
     all_caps_count, length_feature]
    """
    tokens = _normalize_tokens(text)
    pos_lexicon = frozenset(_POS | {"professional", "measured", "calm"})
    neg_lexicon = frozenset(_NEG | {"shouting", "disrespect", "threat"})
    
    pos = sum(1 for t in tokens if t in pos_lexicon)
    neg = sum(1 for t in tokens if t in neg_lexicon)
    polite = sum(1 for t in tokens if t in _POLITE)
    intense = sum(1 for t in tokens if t in _INTENSE)
    hedge = sum(1 for t in tokens if t in _HEDGES)
    titles = sum(1 for t in tokens if t in _RESPECT_TITLES)
    exclaim = text.count("!")
    question = text.count("?")
    all_caps = sum(1 for word in text.split() if len(word) > 2 and word.isupper())
    length_feature = min(len(tokens) / 25.0, 1.5)
    
    return np.array([
        float(pos),
        float(neg),
        float(polite),
        float(intense),
        float(hedge),
        float(titles),
        float(exclaim),
        float(question),
        float(all_caps),
        float(length_feature),
    ])


def _get_model_path() -> Path:
    """Get the path where the frozen model should be stored."""
    module_dir = Path(__file__).parent
    models_dir = module_dir / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir / "frozen_tone_classifier.pkl"


@dataclass(frozen=True)
class ToneSample:
    """Single labeled utterance for tone calibration."""

    text: str
    label: int  # 1 = professional/positive tone, 0 = hostile/negative tone

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("ToneSample.label must be 0 or 1")


@dataclass(frozen=True)
class ToneCalibrationResult:
    """Outputs from calibrating and evaluating the frozen classifier."""

    params: PlattParams
    ece: float
    kappa: float
    accuracy: float
    raw_scores: Tuple[float, ...]
    calibrated_probs: Tuple[float, ...]
    labels: Tuple[int, ...]
    texts: Tuple[str, ...]

    def as_table(self) -> Tuple[Tuple[str, float, float, int], ...]:
        """Return convenient table rows of (text, score, prob, label)."""
        return tuple(
            (text, score, prob, label)
            for text, score, prob, label in zip(
                self.texts, self.raw_scores, self.calibrated_probs, self.labels
            )
        )


class FrozenToneClassifier:
    """Frozen logistic regression classifier for courtroom tone heuristics.
    
    The model is trained once on the default samples, saved to disk, and
    loaded on subsequent uses. This ensures consistent, reproducible outputs
    across different runs and machines.
    
    The classifier uses sklearn's LogisticRegression with Platt scaling
    via CalibratedClassifierCV for probabilistic outputs.
    """

    def __init__(self, model_path: Path | None = None):
        """Initialize the frozen classifier, loading from disk if available.
        
        Args:
            model_path: Optional path to saved model. If None, uses default location.
        """
        if model_path is None:
            model_path = _get_model_path()
        self.model_path = Path(model_path)
        self._model: CalibratedClassifierCV | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk, or raise error if not found."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Frozen model not found at {self.model_path}. "
                "Train the model first using train_frozen_tone_classifier()."
            )
        with open(self.model_path, "rb") as f:
            self._model = pickle.load(f)

    def score(self, text: str) -> float:
        """Get raw score (decision function value) for a single text."""
        features = extract_features(text).reshape(1, -1)
        # Get decision function value from the base estimator (before calibration)
        # CalibratedClassifierCV stores the base estimator in calibrated_classifiers_
        if hasattr(self._model, "calibrated_classifiers_"):
            # Get the first calibrated classifier (there's only one with cv='prefit')
            cal_clf = self._model.calibrated_classifiers_[0]
            # _CalibratedClassifier has 'estimator' attribute, not 'base_estimator'
            base = cal_clf.estimator
        elif hasattr(self._model, "base_estimator"):
            base = self._model.base_estimator
        elif hasattr(self._model, "estimator"):
            base = self._model.estimator
        else:
            base = self._model
        
        if hasattr(base, "decision_function"):
            return float(base.decision_function(features)[0])
        # Fallback: use predict_proba and invert sigmoid
        prob = self.predict_proba([text])[0]
        # Inverse sigmoid: log(p / (1-p))
        if prob <= 0.0:
            return -10.0
        if prob >= 1.0:
            return 10.0
        return float(np.log(prob / (1 - prob)))

    def score_many(self, texts: Sequence[str]) -> Tuple[float, ...]:
        """Get raw scores for multiple texts."""
        return tuple(self.score(text) for text in texts)

    def predict_proba(self, texts: Sequence[str]) -> Tuple[float, ...]:
        """Get calibrated probabilities for multiple texts.
        
        Returns probabilities for class 1 (professional/positive tone).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        features = np.array([extract_features(text) for text in texts])
        probs = self._model.predict_proba(features)
        # Return probabilities for class 1
        return tuple(float(p[1]) for p in probs)


DEFAULT_TONE_SAMPLES: Tuple[ToneSample, ...] = (
    ToneSample("Thank you for your consideration, Your Honor.", 1),
    ToneSample("I appreciate the court's patience with these exhibits.", 1),
    ToneSample("Please let me clarify one final point for the record.", 1),
    ToneSample("We respectfully request a brief recess to confer.", 1),
    ToneSample("Your Honor, the defense has been professional throughout.", 1),
    ToneSample("Thank you, sir, for allowing me to respond.", 1),
    ToneSample("I understand the concern and will address it calmly.", 1),
    ToneSample("Perhaps we could review the evidence sequentially.", 1),
    ToneSample("Thanks to the prosecution for producing the documents.", 1),
    ToneSample("The witness was measured and cooperative.", 1),
    ToneSample("This behavior is outrageous and completely unacceptable!", 0),
    ToneSample("The prosecution is being ridiculous and hostile.", 0),
    ToneSample("Your Honor, this is a disgraceful waste of time!", 0),
    ToneSample("I'm furious that we even need to discuss this.", 0),
    ToneSample("They shouted threats and were downright unprofessional.", 0),
    ToneSample("This entire process is a joke, an angry sham!", 0),
    ToneSample("You keep ignoring the facts and it's unacceptable.", 0),
    ToneSample("This witness is outright disrespectful!", 0),
    ToneSample("We are outraged by the defendant's aggressive tone.", 0),
    ToneSample("Stop dodging the question and answer directly!", 0),
    ToneSample("Your Honor, I must admit the timeline is slightly uncertain.", 1),
    ToneSample("The defense maybe misinterpreted the contract terms.", 1),
    ToneSample("Kindly note that we complied with discovery.", 1),
    ToneSample("The officer was angry and rude throughout the arrest.", 0),
)


def train_frozen_tone_classifier(
    samples: Sequence[ToneSample] = DEFAULT_TONE_SAMPLES,
    model_path: Path | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> FrozenToneClassifier:
    """Train and save the frozen tone classifier.
    
    This function trains a logistic regression model on the provided samples,
    applies Platt scaling calibration, and saves the model to disk. The model
    is then frozen and can be loaded on subsequent runs.
    
    Args:
        samples: Labeled samples for training
        model_path: Path to save the model. If None, uses default location.
        random_state: Random seed for reproducibility
        test_size: Fraction of data to use for calibration (rest for training)
    
    Returns:
        The trained and saved FrozenToneClassifier
    """
    if model_path is None:
        model_path = _get_model_path()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract features and labels
    X = np.array([extract_features(sample.text) for sample in samples])
    y = np.array([sample.label for sample in samples])
    
    # Split for training and calibration
    # Use stratification if possible, otherwise skip it for small datasets
    try:
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # If stratification fails (e.g., too few samples per class), skip it
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Train base classifier
    base_clf = LogisticRegression(max_iter=500, random_state=random_state)
    base_clf.fit(X_train, y_train)
    
    # Apply Platt scaling calibration
    cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv="prefit")
    cal_clf.fit(X_cal, y_cal)
    
    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(cal_clf, f)
    
    # Return classifier that loads from the saved model
    return FrozenToneClassifier(model_path=model_path)


def calibrate_frozen_tone_model(
    samples: Sequence[ToneSample] = DEFAULT_TONE_SAMPLES,
    classifier: FrozenToneClassifier | None = None,
    bins: int = 10,
    force_retrain: bool = False,
) -> ToneCalibrationResult:
    """Fit Platt scaling for the frozen classifier and report calibration metrics.
    
    If no classifier is provided and no saved model exists, the model will be
    trained first. If force_retrain is True, the model will be retrained even
    if a saved version exists.
    
    Args:
        samples: Samples to use for evaluation (and training if needed)
        classifier: Optional pre-trained classifier. If None, loads from disk or trains.
        bins: Number of bins for ECE calculation
        force_retrain: If True, retrain the model even if saved version exists
    
    Returns:
        ToneCalibrationResult with calibration metrics and outputs
    """
    if classifier is None:
        model_path = _get_model_path()
        if force_retrain or not model_path.exists():
            classifier = train_frozen_tone_classifier(samples=samples)
        else:
            classifier = FrozenToneClassifier()
    
    texts = tuple(sample.text for sample in samples)
    labels = tuple(sample.label for sample in samples)
    
    # Get raw scores (decision function values)
    raw_scores = classifier.score_many(texts)
    
    # Get calibrated probabilities
    calibrated_probs = classifier.predict_proba(texts)
    
    # Extract Platt parameters from the calibrated model
    # The CalibratedClassifierCV uses Platt scaling internally
    # We approximate the params by fitting to the raw scores and calibrated probs
    # This is for reporting purposes - the actual calibration is in the model
    params = fit_platt(raw_scores, labels)
    
    predictions = tuple(1 if prob >= 0.5 else 0 for prob in calibrated_probs)
    ece = expected_calibration_error(calibrated_probs, labels, bins=bins)
    kappa = cohen_kappa(predictions, labels)
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
    
    return ToneCalibrationResult(
        params=params,
        ece=ece,
        kappa=kappa,
        accuracy=accuracy,
        raw_scores=raw_scores,
        calibrated_probs=calibrated_probs,
        labels=labels,
        texts=texts,
    )
