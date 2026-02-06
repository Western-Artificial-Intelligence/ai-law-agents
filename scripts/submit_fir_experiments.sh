#!/bin/bash
# Submit FIR experiment job.
# Usage:
#   bash scripts/submit_fir_experiments.sh
#   BAILIFF_CONFIG=configs/fir_batch.yaml bash scripts/submit_fir_experiments.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p runs/slurm
JOB_ID="$(sbatch --export=ALL,BAILIFF_CONFIG="${BAILIFF_CONFIG:-configs/fir_batch.yaml}" scripts/run_fir_experiments.sbatch | awk '{print $4}')"
echo "Submitted job: $JOB_ID"
squeue -j "$JOB_ID" || true
