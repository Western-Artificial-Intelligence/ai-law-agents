#!/bin/bash
# Submit the 3-model SOTA add-on local GPU panel on Nibi.
# Usage:
#   bash scripts/submit_nibi_sota_addons.sh
#   SBATCH_ACCOUNT=def-pviswana bash scripts/submit_nibi_sota_addons.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
  "configs/nibi_local_qwen3_30b_a3b_reduced.yaml"
  "configs/nibi_local_deepseek_r1_distill_qwen32b_reduced.yaml"
  "configs/nibi_local_qwen25_72b_reduced.yaml"
)

if [ -z "${SBATCH_ACCOUNT:-}" ]; then
  SBATCH_ACCOUNT="$(id -Gn | tr ' ' '\n' | grep -E '^(def|rrg)-' | head -n1 || true)"
fi
if [ -z "${SBATCH_ACCOUNT:-}" ]; then
  echo "Could not infer SBATCH_ACCOUNT. Set SBATCH_ACCOUNT explicitly."
  exit 1
fi

export SBATCH_PARTITION="${SBATCH_PARTITION:-gpubase_bygpu_b3}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:h100:1}"
export SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-8}"
export SBATCH_MEM="${SBATCH_MEM:-56G}"
export SBATCH_TIME="${SBATCH_TIME:-24:00:00}"

echo "Submitting SOTA add-on jobs with:"
echo "  SBATCH_ACCOUNT=$SBATCH_ACCOUNT"
echo "  SBATCH_PARTITION=$SBATCH_PARTITION"
echo "  SBATCH_GRES=$SBATCH_GRES"
echo "  SBATCH_CPUS_PER_TASK=$SBATCH_CPUS_PER_TASK"
echo "  SBATCH_MEM=$SBATCH_MEM"
echo "  SBATCH_TIME=$SBATCH_TIME"

for cfg in "${CONFIGS[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "Missing config: $cfg"
    exit 1
  fi
  tag="$(basename "$cfg" .yaml)"
  echo ""
  echo "==> Submitting $cfg (tag=$tag)"
  PYTHONUNBUFFERED=1 \
    BAILIFF_CONFIG="$cfg" \
    BAILIFF_RUN_TAG="$tag" \
    BAILIFF_LOG_PATH="runs/${tag}_logs.jsonl" \
    BAILIFF_MANIFEST_PATH="runs/${tag}_manifest.jsonl" \
    BAILIFF_OUTCOMES_PATH="runs/${tag}_outcomes.csv" \
    BAILIFF_PLOTS_DIR="runs/${tag}_plots" \
    SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
    bash scripts/submit_fir_experiments.sh
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER | grep bailiff-batch"
