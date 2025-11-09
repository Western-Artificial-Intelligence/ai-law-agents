# Outcome models: GLMM + Satterthwaite + Wild Cluster Bootstrap

This page documents runnable scripts to estimate outcome effects and small-sample uncertainty.

- Binary outcome: conviction (`verdict_bin` 1=guilty, 0=not-guilty)
- Treatment: `cue_treatment` (1=treatment, 0=control)
- Cluster: matched pair (`pair_id`)

## Requirements

- Python: repo installed (bailiff), pandas
- R: lme4, geepack, clubSandwich, readr, jsonlite

Install R packages (example):

```r
install.packages(c("lme4","geepack","clubSandwich","readr","jsonlite"))
```

## 1) Generate logs (example)

```bash
# Echo backend (no API keys), writes trial_logs.jsonl
python scripts/run_pilot_trial.py --config configs/pilot.yaml --backend echo --out trial_logs.jsonl
```

## 2) Prepare tidy CSV

```bash
python scripts/prepare_outcome_data.py trial_logs.jsonl --out outcomes.csv
```

Columns: trial_id, pair_id, cue_condition, cue_treatment, verdict_bin, case_identifier, seed, model_identifier, cue_value

## 3) Logistic GLMM (random intercept for pair)

```bash
Rscript scripts/outcome_glmm.R --input=outcomes.csv --out=glmm.json --ci=profile --level=0.95
```

Outputs (JSON):
- model: glmer_logit
- beta (log-odds), se_wald, or (exp(beta))
- ci_logit, ci_or (profile-likelihood if available; falls back to Wald)

Note: Kenward–Roger is defined for linear mixed models, not logistic GLMMs. We provide profile-likelihood CIs for GLMM.

## 4) GEE with CR2 and Satterthwaite correction (small-sample)

```bash
Rscript scripts/outcome_gee_satt.R --input=outcomes.csv --out=gee.json --level=0.95
```

- Model: logit GEE, exchangeable working correlation within pair
- Variance: CR2 cluster-robust (clubSandwich)
- Inference: Satterthwaite df for coefficient on `cue_treatment`
- Outputs: beta, se_cr2, df_satt, or, ci_logit, ci_or

## 5) Wild cluster bootstrap for paired McNemar log-odds

```bash
python scripts/outcome_wild_bootstrap.py outcomes.csv --reps 2000 --seed 123 --out bootstrap.json
```

- Statistic: McNemar log-odds on paired outcomes
- Bootstrap: wild bootstrap with Rademacher signs at the pair level (design-based)
- Outputs: estimate (log-odds), OR, ci_logit [p5,p95], ci_or [p5,p95]
 - Seed: if `--seed` is omitted, defaults to `BAILIFF_ANALYSIS_SEED` or `ANALYSIS_SEED` env var (else 123)
 - Continuity correction: applies Haldane–Anscombe when a discordant cell count is zero to keep a finite log-odds statistic

## Notes

- Use GLMM for model-based effect with random intercepts at pair-level.
- Use GEE+CR2+Satterthwaite for small-sample adjusted inference in logistic regression clustered by pair.
- Use wild bootstrap to obtain design-robust intervals for the paired effect statistic.
- All steps are script-driven and require no notebooks.
