"""Run the frozen tone classifier calibration pipeline and print summary metrics."""
from __future__ import annotations

from textwrap import shorten

from bailiff.metrics.tone import calibrate_frozen_tone_model


def main() -> int:
    result = calibrate_frozen_tone_model()
    print("Frozen tone classifier calibration")
    print(f"Platt parameters: a={result.params.a:.3f}, b={result.params.b:.3f}")
    print(f"Accuracy: {result.accuracy:.3f}")
    print(f"ECE (10 bins): {result.ece:.3f}")
    print(f"Cohen's kappa: {result.kappa:.3f}")
    print()
    print("Sample outputs (text | raw score | calibrated prob | label)")
    for text, score, prob, label in result.as_table():
        excerpt = shorten(text, width=72, placeholder="…")
        print(f"- {excerpt} | {score:+.3f} | {prob:.3f} | {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

