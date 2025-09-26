# Notebooks

Place exploratory analysis notebooks here. Suggested starters:

- `01_paired_outcomes.ipynb`: load trial logs, construct `PairedOutcome` records, compute McNemar odds ratios.
- `02_procedural_metrics.ipynb`: derive byte share delta, interruption corrections, and tone gaps.
- `03_randomization_inference.ipynb`: prototype permutation testing aligned with block design.

Ensure notebooks read from exported JSONL logs produced by `TrialPipeline` executions.
