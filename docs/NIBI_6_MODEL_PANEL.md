# Nibi 6-Model Local GPU Panel

This panel is designed for paper-scale breadth with one job per model on one H100 each.

## Configs

Full-seed tier (`30` seeds, `360` paired assignments per model):

- `configs/nibi_local_llama3_8b_full.yaml`
- `configs/nibi_local_qwen25_7b_full.yaml`
- `configs/nibi_local_mistral7b_full.yaml`
- `configs/nibi_local_phi3mini_full.yaml`

Reduced-seed tier (`10` seeds, `120` paired assignments per model):

- `configs/nibi_local_qwen25_14b_reduced.yaml`
- `configs/nibi_local_mixtral8x7b_reduced.yaml`

Total panel volume:

- `1680` paired assignments
- `3360` individual trials

## Launch

From `~/ai-law-agents` on Nibi:

```bash
source .venv/bin/activate
bash scripts/submit_nibi_local_panel.sh
```

Optional overrides:

```bash
SBATCH_ACCOUNT=def-pviswana \
SBATCH_PARTITION=gpubase_bygpu_b3 \
SBATCH_GRES='gpu:h100:1' \
SBATCH_TIME=24:00:00 \
bash scripts/submit_nibi_local_panel.sh
```

## Monitoring

```bash
squeue -u "$USER" | grep bailiff-batch
```

For a specific job id:

```bash
tail -f runs/slurm/bailiff-batch-<JOBID>.out
tail -f runs/slurm/bailiff-batch-<JOBID>.err
```

## Runtime expectations

Based on observed local throughput near `~25` pairs/hour on `1x H100`:

- Full-tier model: about `14-16` hours
- Reduced-tier model: about `5-7` hours

With enough free GPUs for all 6 concurrent jobs, wall-clock is usually bounded by the slowest full-tier model plus queue delay.
