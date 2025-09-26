# B.A.I.L.I.F.F. Scaffold

This repository hosts the early-stage implementation for **Bias Analysis in Interactive Legal Intelligence & Fairness Framework (B.A.I.L.I.F.F.)**. The focus of this scaffold is to provide well-documented entry points for:

- spinning up multi-agent legal role simulations with controllable demographic cues;
- capturing structured event logs aligned with the paper's estimands;
- preparing analysis-ready extracts for outcome and procedural bias metrics.

## Layout

- `bailiff/core`: Simulation primitives (state machine, events, trial configuration).
- `bailiff/agents`: Agent interface definitions and prompt scaffolds.
- `bailiff/datasets`: Case templates, cue catalogs, negative control definitions.
- `bailiff/metrics`: Outcome/procedural metric calculators and measurement corrections.
- `bailiff/orchestration`: Pipelines for running paired trials, randomization, and logging.
- `scripts/`: CLI entry points for generating trial plans and executing pilot runs.

Each module includes TODOs for expanding toward the full methodology in the manuscript.

## Getting Started

1. `python -m venv .venv && .venv\Scripts\activate`
2. `pip install -e .[analysis,agent]`
3. Populate `config/` (to be added) with API credentials and byte budget policies.
4. Run `python scripts/run_pilot_trial.py --config configs/pilot.yaml` after filling in placeholders.

Refer to inline docstrings for details on the logging schema and estimand alignment.

