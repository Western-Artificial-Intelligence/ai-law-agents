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
RUN_TAG="fir_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="runs/${RUN_TAG}_logs.jsonl"
MANIFEST_PATH="runs/${RUN_TAG}_manifest.jsonl"
OUTCOMES_PATH="runs/${RUN_TAG}_outcomes.csv"
PLOTS_DIR="runs/${RUN_TAG}_plots"

python scripts/run_trial_matrix.py \
  --config "$CONFIG_PATH" \
  --out "$LOG_PATH" \
  --manifest "$MANIFEST_PATH"

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
