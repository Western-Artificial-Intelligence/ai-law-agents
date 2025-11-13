"""CLI workflow to estimate measurement-error rates (alpha/beta) and corrected bias."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bailiff.metrics.procedural import measurement_error_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate measurement-error calibration parameters.")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with labeled columns.")
    parser.add_argument("--true-column", default="true_label", help="Column containing ground-truth labels (0/1).")
    parser.add_argument("--pred-column", default="pred_label", help="Column containing predicted labels (0/1).")
    parser.add_argument(
        "--observed-rate",
        type=float,
        required=True,
        help="Observed positive rate from the full dataset before correction.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap repetitions for CIs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap.")
    parser.add_argument("--out", type=Path, help="Optional JSON file to write the summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.labels)
    for col in (args.true_column, args.pred_column):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in {args.labels}")
    y_true = df[args.true_column].astype(int).to_numpy()
    y_pred = df[args.pred_column].astype(int).to_numpy()
    result = measurement_error_calibration(
        y_true,
        y_pred,
        observed_rate=args.observed_rate,
        reps=args.bootstrap,
        seed=args.seed,
    )
    summary = {
        "alpha": result.alpha,
        "beta": result.beta,
        "corrected_rate": result.corrected_rate,
        "alpha_ci": result.alpha_ci,
        "beta_ci": result.beta_ci,
        "corrected_ci": result.corrected_ci,
        "n_labeled": int(len(y_true)),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
