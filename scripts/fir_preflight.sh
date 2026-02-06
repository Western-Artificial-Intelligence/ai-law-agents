#!/bin/bash
# Fast validation before long FIR jobs.
# Usage: bash scripts/fir_preflight.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
mkdir -p runs/preflight

pytest -q

python scripts/run_pilot_trial.py \
  --config configs/pilot.yaml \
  --backend echo \
  --out runs/preflight/trial_logs.jsonl \
  --manifest runs/preflight/manifest.jsonl

python scripts/prepare_outcome_data.py \
  runs/preflight/trial_logs.jsonl \
  --out runs/preflight/outcomes.csv

python scripts/plot_results.py \
  runs/preflight/outcomes.csv \
  --out runs/preflight/plots

echo "FIR preflight checks passed."
