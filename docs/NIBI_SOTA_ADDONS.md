# Nibi SOTA Add-ons (3 Models)

These add-ons extend the core panel with larger/newer open-weight models.

## Configs

- `configs/nibi_local_qwen3_30b_a3b_reduced.yaml`
- `configs/nibi_local_deepseek_r1_distill_qwen32b_reduced.yaml`
- `configs/nibi_local_qwen25_72b_reduced.yaml`

These are reduced-seed runs to keep wall-clock practical.

## Launch

```bash
cd ~/ai-law-agents
source .venv/bin/activate

SBATCH_ACCOUNT=def-pviswana \
SBATCH_PARTITION=gpubase_bygpu_b3 \
SBATCH_GRES='gpu:h100:1' \
SBATCH_CPUS_PER_TASK=8 \
SBATCH_MEM=56G \
SBATCH_TIME=24:00:00 \
bash scripts/submit_nibi_sota_addons.sh
```

## Notes

- `Qwen/Qwen3-30B-A3B-Instruct-2507` may require a recent `transformers` build. If it fails to load, upgrade in the venv.
- `Qwen/Qwen2.5-72B-Instruct` is the most memory-sensitive of the set; keep 4-bit quantization enabled.
- Submit these after core jobs are stable to avoid queue saturation.
