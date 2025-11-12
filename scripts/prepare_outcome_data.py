#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List

from bailiff.core.io import read_jsonl


def verdict_to_bin(v: Any) -> int | None:
    if v is None:
        return None
    s = str(v).lower().strip()
    if s == "guilty":
        return 1
    if s in ("not_guilty", "not guilty"):
        return 0
    return None


def pair_id_for(trial: Dict[str, Any]) -> str:
    case = str(trial.get("case_identifier", "case"))
    seed = int(trial.get("seed", 0))
    cond = trial.get("cue_condition")
    # Treatment is assigned seed+1 in TrialPipeline; pair on control seed
    if cond == "treatment":
        seed = seed - 1
    return f"{case}-{seed}"


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare tidy outcome CSV from trial JSONL logs.")
    p.add_argument("logs", type=Path, help="Path to JSONL logs (from run_pilot_trial.py)")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = p.parse_args()

    recs = read_jsonl(args.logs)
    rows: List[Dict[str, Any]] = []
    for r in recs:
        vb = verdict_to_bin(r.get("verdict"))
        if vb is None:
            continue
        row: Dict[str, Any] = {
            "trial_id": r.get("trial_id"),
            "pair_id": pair_id_for(r),
            "cue_condition": r.get("cue_condition"),
            "cue_treatment": 1 if r.get("cue_condition") == "treatment" else 0,
            "verdict_bin": vb,
            "case_identifier": r.get("case_identifier"),
            "seed": int(r.get("seed", 0)),
            "model_identifier": r.get("model_identifier"),
            "cue_value": r.get("cue_value"),
        }
        rows.append(row)

    # Keep only pairs with both control and treatment
    from collections import Counter

    counts = Counter([row["pair_id"] for row in rows])
    keep = {pid for pid, c in counts.items() if c >= 2}
    rows = [row for row in rows if row["pair_id"] in keep]

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = (
            list(rows[0].keys())
            if rows
            else [
                "trial_id",
                "pair_id",
                "cue_condition",
                "cue_treatment",
                "verdict_bin",
                "case_identifier",
                "seed",
                "model_identifier",
                "cue_value",
            ]
        )
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
