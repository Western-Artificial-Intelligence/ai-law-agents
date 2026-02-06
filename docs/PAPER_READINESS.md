# Paper Readiness Checklist

This file maps manuscript claims to executable artifacts in this repository.

## Status: codebase readiness

- `pytest -q` passes (42 tests).
- Ablation harness, pilot runner, and batch runner are API-consistent.
- Local backend 4-bit quantization config is now wired in both pilot and batch paths.
- `configs/pilot.yaml` now points to a valid case path.
- Merge-conflict markers removed from user docs.

## RQ coverage

1. **Outcome bias (RQ1)**
- Run matrix: `python scripts/run_trial_matrix.py --config configs/fir_batch.yaml ...`
- Outcome extraction: `python scripts/prepare_outcome_data.py ...`
- Statistics: `scripts/outcome_glmm.R`, `scripts/outcome_gee_satt.R`, `scripts/outcome_wild_bootstrap.py`

2. **Procedural bias (RQ2)**
- Procedural fields are in each `TrialLog` utterance:
  - `interruption`
  - `objection_raised`
  - `objection_ruling`
  - `byte_count`
  - `token_count`
- Metric utilities: `bailiff/metrics/procedural.py`
- Measurement calibration CLI: `python scripts/run_measurement_calibration.py ...`

3. **Procedure-outcome link (RQ3)**
- Join outcome and procedural metrics via `trial_id`, `case_identifier`, `seed`, `cue_condition`.
- Fit downstream models in R/Python from exported CSV/JSONL.

## Manuscript alignment updates already made

- Removed `PLACEHOLDER` markers in `paper.tex`.
- Replaced unresolved artifact placeholders with reproducible artifact wording.
- Kept analysis language consistent with implemented finite-sample methods.

## Remaining data-dependent items (cannot be finalized before full run)

- Final numeric tables/figures for:
  - ablation diagnostics
  - classifier calibration curves
  - interruption detector calibration
- Final confidence intervals and p-values for full sample.

These are generated after the FIR batch run finishes.

## One-command run order for paper artifacts

1. `bash scripts/setup_fir.sh`
2. `bash scripts/fir_preflight.sh`
3. `bash scripts/submit_fir_experiments.sh`
4. Post-run:
   - `python scripts/prepare_outcome_data.py ...`
   - `python scripts/plot_results.py ...`
   - `python scripts/outcome_wild_bootstrap.py ...`
   - Optional R scripts for GLMM/GEE.
