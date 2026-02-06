#!/bin/bash
#SBATCH --job-name=bailiff-batch
#SBATCH --account=def-<your_alliance_pi>
#SBATCH --gres=gpu:1
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

python scripts/plot_results.py \
  "$OUTCOMES_PATH" \
  --out "$PLOTS_DIR"

echo "Batch run complete."
echo "Logs: $LOG_PATH"
echo "Manifest: $MANIFEST_PATH"
echo "Outcomes CSV: $OUTCOMES_PATH"
echo "Plots: $PLOTS_DIR"
