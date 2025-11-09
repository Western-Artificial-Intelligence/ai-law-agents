#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import math
import os
import pandas as pd

from bailiff.analysis.stats import wild_cluster_bootstrap


def mcnemar_log_odds_stat(yc: List[int], yt: List[int]) -> float:
    """Return McNemar log-odds on paired binary outcomes.

    yc, yt are aligned by pair.
    """
    assert len(yc) == len(yt)
    n01 = sum(1 for c, t in zip(yc, yt) if c == 0 and t == 1)
    n10 = sum(1 for c, t in zip(yc, yt) if c == 1 and t == 0)
    if n01 == 0 or n10 == 0:
        # Haldane–Anscombe continuity correction for finite log-odds
        return math.log((n01 + 0.5) / (n10 + 0.5))
    return math.log(n01 / n10)


def build_pairs(csv_path: Path) -> Tuple[List[int], List[int]]:
    df = pd.read_csv(csv_path)
    # Expect two rows per pair_id: control and treatment
    pairs = []
    for pid, g in df.groupby("pair_id"):
        g = g.sort_values("cue_treatment")  # 0=control,1=treatment
        if g.shape[0] < 2:
            continue
        c = int(g.iloc[0]["verdict_bin"])  # control
        t = int(g.iloc[1]["verdict_bin"])  # treatment
        pairs.append((c, t))
    yc = [c for c, _ in pairs]
    yt = [t for _, t in pairs]
    return yc, yt


def main() -> None:
    ap = argparse.ArgumentParser(description="Wild cluster bootstrap for paired outcome effect (McNemar log-odds).")
    ap.add_argument("input", type=Path, help="CSV from prepare_outcome_data.py")
    ap.add_argument("--reps", type=int, default=1000, help="Bootstrap repetitions")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default from BAILIFF_ANALYSIS_SEED/ANALYSIS_SEED env or 123)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON path for results")
    args = ap.parse_args()

    yc, yt = build_pairs(args.input)
    if not yc:
        raise SystemExit("No complete pairs found in input.")

    seed = args.seed if args.seed is not None else int(
        os.getenv("BAILIFF_ANALYSIS_SEED", os.getenv("ANALYSIS_SEED", "123"))
    )
    p5, p95 = wild_cluster_bootstrap(mcnemar_log_odds_stat, yc, yt, reps=args.reps, seed=seed)

    # Point estimate
    est = mcnemar_log_odds_stat(yc, yt)

    # Report as log-odds and OR
    res = {
        "stat": "mcnemar_log_odds",
        "pairs": len(yc),
        "estimate": est,
        "or": float(math.exp(est)) if est is not None else None,
        "ci_logit": [p5, p95],
        "ci_or": [float(math.exp(p5)), float(math.exp(p95))],
        "reps": args.reps,
        "seed": seed,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
