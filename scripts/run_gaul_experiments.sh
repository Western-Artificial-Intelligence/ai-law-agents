#!/bin/bash
# Script to run experiments on GAUL
# Usage: ./scripts/run_gaul_experiments.sh

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Activate environment
source .venv/bin/activate

# Create runs directory
mkdir -p runs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="runs/gaul_batch_logs_${TIMESTAMP}.jsonl"
MANIFEST_FILE="runs/gaul_batch_manifest.jsonl"

echo "🚀 Starting GAUL Experiments..."
echo "   Config: configs/gaul_batch.yaml"
echo "   Output: $LOG_FILE"
echo "   Manifest: $MANIFEST_FILE"

# Run with nohup to keep running if disconnected
# We use the 'local' backend configuration from gaul_batch.yaml
nohup python -u scripts/run_trial_matrix.py \
    --config configs/gaul_batch.yaml \
    --out "$LOG_FILE" \
    --manifest "$MANIFEST_FILE" \
    > "runs/execution_${TIMESTAMP}.log" 2>&1 &

PID=$!
echo "✅ Job started in background with PID: $PID"
echo "   To monitor progress: tail -f runs/execution_${TIMESTAMP}.log"
