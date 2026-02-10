#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from bailiff.core.io import read_jsonl


def verdict_to_bin(v: Any) -> int | None:
    if v is None:
        return None
    s = str(v).lower().strip()
    if "not guilty" in s or "not_guilty" in s:
        return 0
    if "guilty" in s:
        return 1
    return None


def pair_id_for(trial: Dict[str, Any]) -> str:
    case = str(trial.get("case_identifier", "case"))
    cue_name = str(trial.get("cue_name", "cue"))
    is_placebo = bool(trial.get("is_placebo", False))
    seed = int(trial.get("seed", 0))
    cond = trial.get("cue_condition")
    # Treatment is assigned seed+1 in TrialPipeline; pair on control seed
    if cond == "treatment":
        seed = seed - 1
    tag = "placebo" if is_placebo else "primary"
    return f"{case}-{cue_name}-{tag}-{seed}"


def normalize_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return exactly one control+treatment row for each complete pair."""

    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row["pair_id"])
        condition = str(row.get("cue_condition", ""))
        if condition not in {"control", "treatment"}:
            continue
        by_pair[pair_id][condition] = row

    normalized_rows: List[Dict[str, Any]] = []
    for cond_map in by_pair.values():
        if set(cond_map.keys()) != {"control", "treatment"}:
            continue
        normalized_rows.append(cond_map["control"])
        normalized_rows.append(cond_map["treatment"])
    return normalized_rows


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
            "cue_name": r.get("cue_name"),
            "is_placebo": bool(r.get("is_placebo", False)),
            "verdict_bin": vb,
            "case_identifier": r.get("case_identifier"),
            "seed": int(r.get("seed", 0)),
            "model_identifier": r.get("model_identifier"),
            "cue_value": r.get("cue_value"),
        }
        rows.append(row)

    # Keep exactly one control and one treatment per pair.
    # If retries produced duplicates for a condition, retain the most recent row.
    rows = normalize_pairs(rows)

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
                "cue_name",
                "is_placebo",
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
