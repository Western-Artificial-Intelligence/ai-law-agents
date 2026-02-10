#!/bin/bash
# Submit research-backed multi-cue local GPU runs on Nibi.
# Usage:
#   bash scripts/submit_nibi_research_panel.sh
#   SBATCH_ACCOUNT=def-pviswana bash scripts/submit_nibi_research_panel.sh
#   bash scripts/submit_nibi_research_panel.sh configs/nibi_local_qwen25_7b_research.yaml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    "configs/nibi_local_llama3_8b_research.yaml"
    "configs/nibi_local_qwen25_7b_research.yaml"
    "configs/nibi_local_mistral7b_research.yaml"
    "configs/nibi_local_phi3mini_research.yaml"
    "configs/nibi_local_qwen25_14b_research.yaml"
    "configs/nibi_local_mixtral8x7b_research.yaml"
    "configs/nibi_local_qwen3_30b_a3b_research.yaml"
    "configs/nibi_local_deepseek_r1_distill_qwen32b_research.yaml"
    "configs/nibi_local_qwen25_72b_research.yaml"
  )
fi

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

echo "Submitting ${#CONFIGS[@]} research panel jobs with:"
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
