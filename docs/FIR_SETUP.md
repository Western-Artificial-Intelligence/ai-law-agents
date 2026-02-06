# Running B.A.I.L.I.F.F. on FIR (Compute Alliance)

This guide is the production path for paper runs on a Slurm-based FIR/Alliance cluster.

## 1) Clone and setup

```bash
git clone <your-repo-url> ai-law-agents
cd ai-law-agents
bash scripts/setup_fir.sh
```

Notes:
- If your cluster uses different modules, set `FIR_MODULES` before setup.
- If you only plan Groq/Gemini API runs, set `INSTALL_LOCAL_BACKEND=0`.

## 2) Configure credentials

Create `.env` in repo root if you are using API backends:

```bash
cat <<'EOF' > .env
GROQ_API_KEYS='["key1","key2"]'
GROQ_API_KEY_CONCURRENCY='{"key1":1,"key2":1}'
GOOGLE_API_KEY='your-google-key'
EOF
chmod 600 .env
```

## 3) Preflight before long jobs

```bash
bash scripts/fir_preflight.sh
```

This runs:
- unit tests (`pytest -q`)
- one pilot echo run
- outcome CSV generation
- plot generation

## 4) Choose batch config

Default paper config is `configs/fir_batch.yaml`.

Key defaults:
- 6 cases
- 30 seeds
- 1 primary cue + 1 placebo per case
- local Llama-3-8B-instruct (4-bit) with `concurrency: 1`

## 5) Submit Slurm job

Edit account/allocation in `scripts/run_fir_experiments.sbatch`:
- `#SBATCH --account=def-<your_alliance_pi>`

Then submit:

```bash
bash scripts/submit_fir_experiments.sh
```

Or submit directly with a custom config:

```bash
BAILIFF_CONFIG=configs/fir_batch.yaml sbatch scripts/run_fir_experiments.sbatch
```

## 6) Monitor and retrieve outputs

```bash
squeue -u $USER
tail -f runs/slurm/bailiff-batch-<jobid>.out
```

Outputs are written under `runs/` with a job-tagged prefix:
- `*_logs.jsonl`
- `*_manifest.jsonl`
- `*_outcomes.csv`
- `*_plots/`

## 7) Paper analysis commands

From a completed logs file:

```bash
python scripts/prepare_outcome_data.py runs/<tag>_logs.jsonl --out runs/<tag>_outcomes.csv
python scripts/plot_results.py runs/<tag>_outcomes.csv --out runs/<tag>_plots
python scripts/outcome_wild_bootstrap.py runs/<tag>_outcomes.csv --reps 2000 --seed 123 --out runs/<tag>_bootstrap.json
```

Optional R analyses (if R packages installed):

```bash
Rscript scripts/outcome_glmm.R --input=runs/<tag>_outcomes.csv --out=runs/<tag>_glmm.json
Rscript scripts/outcome_gee_satt.R --input=runs/<tag>_outcomes.csv --out=runs/<tag>_gee.json
```

## 8) Runtime planning

Rough estimates for `configs/fir_batch.yaml`:
- Local 8B model, 1 GPU, 360 pairs: typically `8-24` hours.
- Add one more model in config: multiply runtime by roughly 2.
- API backend runtime depends mostly on provider limits and key concurrency.

