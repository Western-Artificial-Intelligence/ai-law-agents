#!/bin/bash
#SBATCH --job-name=bailiff-batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p runs/slurm

source .venv/bin/activate

CONFIG_PATH="${BAILIFF_CONFIG:-configs/fir_batch.yaml}"
DEFAULT_RUN_TAG="fir_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${BAILIFF_RUN_TAG:-$DEFAULT_RUN_TAG}"
LOG_PATH="${BAILIFF_LOG_PATH:-runs/${RUN_TAG}_logs.jsonl}"
MANIFEST_PATH="${BAILIFF_MANIFEST_PATH:-runs/${RUN_TAG}_manifest.jsonl}"
OUTCOMES_PATH="${BAILIFF_OUTCOMES_PATH:-runs/${RUN_TAG}_outcomes.csv}"
PLOTS_DIR="${BAILIFF_PLOTS_DIR:-runs/${RUN_TAG}_plots}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$MANIFEST_PATH")" "$(dirname "$OUTCOMES_PATH")" "$PLOTS_DIR"

python scripts/run_trial_matrix.py \
  --config "$CONFIG_PATH" \
  --out "$LOG_PATH" \
  --manifest "$MANIFEST_PATH"

if [ ! -s "$LOG_PATH" ]; then
  echo "[ERROR] No trial logs were produced at: $LOG_PATH"
  if [ -f "$MANIFEST_PATH" ]; then
    echo "[INFO] Manifest exists at: $MANIFEST_PATH"
    python - <<PY
import collections
import json
from pathlib import Path

manifest = Path(r"""$MANIFEST_PATH""")
rows = []
with manifest.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

if not rows:
    print("[INFO] Manifest has no entries.")
else:
    statuses = collections.Counter(str(r.get("status", "unknown")) for r in rows)
    errors = collections.Counter(str(r.get("error", "")).strip() for r in rows if r.get("error"))
    print(f"[INFO] Manifest entries: {len(rows)}")
    print(f"[INFO] Status counts: {dict(statuses)}")
    if errors:
        print("[INFO] Top errors:")
        for msg, count in errors.most_common(5):
            print(f"  - ({count}) {msg}")
PY
  else
    echo "[INFO] Manifest was not created."
  fi
  exit 1
fi

python scripts/prepare_outcome_data.py \
  "$LOG_PATH" \
  --out "$OUTCOMES_PATH"

if [ "${BAILIFF_SKIP_PLOTS:-0}" = "1" ]; then
  echo "Skipping plots (BAILIFF_SKIP_PLOTS=1)."
else
  python scripts/plot_results.py \
    "$OUTCOMES_PATH" \
    --out "$PLOTS_DIR"
fi

echo "Batch run complete."
echo "Logs: $LOG_PATH"
echo "Manifest: $MANIFEST_PATH"
echo "Outcomes CSV: $OUTCOMES_PATH"
if [ "${BAILIFF_SKIP_PLOTS:-0}" = "1" ]; then
  echo "Plots: skipped"
else
  echo "Plots: $PLOTS_DIR"
fi
