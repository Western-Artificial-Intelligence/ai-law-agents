# Nibi Research Panel (Multi-Cue)

This panel runs all local model families with a larger, research-backed cue matrix.

## Configs

- `configs/nibi_local_llama3_8b_research.yaml`
- `configs/nibi_local_qwen25_7b_research.yaml`
- `configs/nibi_local_mistral7b_research.yaml`
- `configs/nibi_local_phi3mini_research.yaml`
- `configs/nibi_local_qwen25_14b_research.yaml`
- `configs/nibi_local_mixtral8x7b_research.yaml`
- `configs/nibi_local_qwen3_30b_a3b_research.yaml`
- `configs/nibi_local_deepseek_r1_distill_qwen32b_research.yaml`
- `configs/nibi_local_qwen25_72b_research.yaml`

## Cue Volume

Each case uses:

- `5` primary cues
- `2` placebo cues
- total `7` cue toggles per case

With `6` case templates, this is `42` paired assignments per seed.

Seed tiers used:

- small models: `12` seeds (`504` pairs/model)
- medium models: `8` seeds (`336` pairs/model)
- large models: `3` seeds (`126` pairs/model)

## Launch All

```bash
cd ~/ai-law-agents
source .venv/bin/activate

SBATCH_ACCOUNT=def-pviswana \
SBATCH_PARTITION=gpubase_bygpu_b3 \
SBATCH_GRES='gpu:h100:1' \
SBATCH_CPUS_PER_TASK=8 \
SBATCH_MEM=56G \
SBATCH_TIME=24:00:00 \
bash scripts/submit_nibi_research_panel.sh
```

## Launch Subset

```bash
bash scripts/submit_nibi_research_panel.sh \
  configs/nibi_local_llama3_8b_research.yaml \
  configs/nibi_local_qwen25_7b_research.yaml
```

## Monitor

```bash
squeue -u "$USER" | grep bailiff-batch
```

Per-job logs:

```bash
tail -f runs/slurm/bailiff-batch-<JOBID>.out
tail -f runs/slurm/bailiff-batch-<JOBID>.err
```

Per-run artifacts (tagged by config stem):

- `runs/<tag>_manifest.jsonl`
- `runs/<tag>_logs.jsonl`
- `runs/<tag>_outcomes.csv`
- `runs/<tag>_plots/`

## Analysis Reminder

Use the same artifact parser for each run:

```bash
python scripts/prepare_outcome_data.py runs/<tag>_logs.jsonl --out runs/<tag>_outcomes.csv
python scripts/plot_results.py runs/<tag>_outcomes.csv --out runs/<tag>_plots
python scripts/outcome_wild_bootstrap.py runs/<tag>_outcomes.csv --reps 5000 --seed 42 --out runs/<tag>_wild_bootstrap.json
```

For cue definitions and references, see `docs/RESEARCH_CUE_MATRIX.md`.
