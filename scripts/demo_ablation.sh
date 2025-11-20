#!/bin/bash
# Demo script for the B.A.I.L.I.F.F. ablation harness
# This script runs a quick ablation study with the echo backend

set -e

echo "=========================================="
echo "B.A.I.L.I.F.F. Ablation Harness Demo"
echo "=========================================="
echo ""

# Create output directory
mkdir -p runs/demo_ablation

echo "Step 1: Running ablation study with echo backend..."
python3 scripts/run_ablation.py \
    --config configs/ablation_example.yaml \
    --backend echo \
    --out runs/demo_ablation/logs.jsonl \
    --comparison-csv runs/demo_ablation/comparison.csv \
    --comparison-md runs/demo_ablation/comparison.md

echo ""
echo "Step 2: Displaying results..."
echo ""
cat runs/demo_ablation/comparison.md

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Trial logs: runs/demo_ablation/logs.jsonl"
echo "  - CSV table: runs/demo_ablation/comparison.csv"
echo "  - Markdown report: runs/demo_ablation/comparison.md"
echo ""
echo "Next steps:"
echo "  1. Try with a real LLM backend:"
echo "     python3 scripts/run_ablation.py --config configs/ablation_example.yaml --backend groq --model llama3-8b-8192"
echo ""
echo "  2. Create your own ablation config:"
echo "     cp configs/ablation_example.yaml configs/my_ablation.yaml"
echo "     # Edit configs/my_ablation.yaml with your variations"
echo ""
echo "  3. Read the full guide:"
echo "     cat docs/ABLATION_GUIDE.md"
