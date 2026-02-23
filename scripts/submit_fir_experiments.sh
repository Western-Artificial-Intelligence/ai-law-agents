#!/bin/bash
# Submit FIR experiment job.
# Usage:
#   bash scripts/submit_fir_experiments.sh
#   BAILIFF_CONFIG=configs/fir_batch.yaml bash scripts/submit_fir_experiments.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p runs/slurm
SBATCH_ARGS=(--export=ALL,BAILIFF_CONFIG="${BAILIFF_CONFIG:-configs/fir_batch.yaml}")

if [ -n "${SBATCH_ACCOUNT:-}" ]; then
  SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
fi
if [ -n "${SBATCH_PARTITION:-}" ]; then
  SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
fi
if [ -n "${SBATCH_TIME:-}" ]; then
  SBATCH_ARGS+=(--time="${SBATCH_TIME}")
fi
if [ -n "${SBATCH_CPUS_PER_TASK:-}" ]; then
  SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS_PER_TASK}")
fi
if [ -n "${SBATCH_MEM:-}" ]; then
  SBATCH_ARGS+=(--mem="${SBATCH_MEM}")
fi
if [ -n "${SBATCH_GRES:-}" ]; then
  SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
fi
if [ -n "${SBATCH_EXCLUDE:-}" ]; then
  SBATCH_ARGS+=(--exclude="${SBATCH_EXCLUDE}")
fi
if [ -n "${SBATCH_CONSTRAINT:-}" ]; then
  SBATCH_ARGS+=(--constraint="${SBATCH_CONSTRAINT}")
fi
if [ -n "${SBATCH_QOS:-}" ]; then
  SBATCH_ARGS+=(--qos="${SBATCH_QOS}")
fi

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" scripts/run_fir_experiments.sbatch | awk '{print $4}')"
echo "Submitted job: $JOB_ID"
squeue -j "$JOB_ID" || true
