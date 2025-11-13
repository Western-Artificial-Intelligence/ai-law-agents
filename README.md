# B.A.I.L.I.F.F. — Bias Analysis in Interactive Legal Intelligence & Fairness Framework

This repository implements a reproducible harness for auditing fairness in interactive, role‑governed legal mini‑trials powered by LLMs. It supports paired counterfactual trials, structured logs, and analysis‑ready metrics for both outcomes (e.g., conviction) and procedure (e.g., byte share, objections, interruptions, tone).

## Features
- Multi‑agent trial simulation with roles: judge, prosecution, defense
- Paired cue toggling (control/treatment) with case×model blocked randomization plus placebo tagging in logs
- Budgets/guards: per-role byte & token caps, per-phase message caps, judge blinding
- Structured logs with event tags (objections, interruptions, safety)
- Metrics: paired McNemar log‑odds, flip rate, byte share, measurement‑error correction, basic tone utilities
- Extensible backends: Echo (offline), Groq, Gemini; open‑source adapters are easy to add
- Batch driver with resumable manifests for running K×L×N matrices
- Versioned JSON Schema validation for TrialLog output (toggle via `BAILIFF_VALIDATE_LOGS=0`)
- Configurable backend hardening (timeouts, retries, rate-limit sleeps) with metadata captured in logs

## Quickstart
1. Create a virtual environment and install:
   - `python -m venv .venv && .venv\Scripts\activate` (Windows) or `source .venv/bin/activate`
   - `pip install -e .[analysis,agent]`
2. (Optional) Set API keys: `GROQ_API_KEY`, `GOOGLE_API_KEY`
3. Run a pilot pair and write logs:
   - Echo: `python scripts/run_pilot_trial.py --config configs/pilot.yaml --backend echo --out trial_logs.jsonl`
   - Groq: `python scripts/run_pilot_trial.py --config configs/pilot.yaml --backend groq --model llama3-8b-8192 --out trial_logs.jsonl`
   - Gemini: `python scripts/run_pilot_trial.py --config configs/pilot.yaml --backend gemini --model gemini-1.5-flash --out trial_logs.jsonl`
   - Add `--placebo <key>` (e.g., `name_placebo`) to schedule additional negative-control pairs if you are not using the sample YAML.
   - Provide `--manifest runs/pilot_manifest.jsonl` to co-save per-run metadata with prompt hashes.
4. Run a batch across cases/models: `python scripts/run_trial_matrix.py --config configs/batch.yaml --out runs/batch_logs.jsonl --manifest runs/batch_manifest.jsonl`

## Verdict JSON Contract
During the VERDICT phase the judge agent must begin its response with a JSON object containing a `verdict` key (and optionally `sentence`). Narrative rationale can follow on subsequent lines, but the leading JSON block is required so `_parse_and_set_verdict_sentence()` can populate `TrialLog.verdict`/`sentence`.

## Repository Layout
- `bailiff/core`: State machine, config, logging, session engine, JSONL I/O
- `bailiff/agents`: Agent abstractions, prompts, optional Groq/Gemini backends
- `bailiff/datasets`: Case templates and cue catalogs
- `bailiff/orchestration`: Randomization and pipelines for paired trials
- `bailiff/schemas`: JSON Schemas (TrialLog) used for validation
- `bailiff/metrics`: Outcome and procedural metrics/utilities
- `bailiff/analysis`: Lightweight statistical helpers
- `scripts/`: CLI entry points (pilot runner)
- `docs/`: User guide and API reference

## Learn More
- Design overview and diagrams: `DESIGN.md`
- User guide (install, run, add case/cue/backend, analysis): `docs/USER_GUIDE.md`
- API reference (core modules): `docs/API.md`
- Measurement-error calibration CLI: `scripts/run_measurement_calibration.py`
- Outcome scripts (GLMM, GEE+Satterthwaite, wild cluster bootstrap): `docs/OUTCOME_SCRIPTS.md`

## FAQ
- How do I add a new case? Create a YAML under `bailiff/datasets/cases/` with `summary`, `facts`, `witnesses`, and `cue_slots`, then run `load_case_templates()` to validate it. See the user guide.
- How do I add a cue? Extend `cue_catalog()` in `bailiff/datasets/templates.py`.
- How do I analyze results? Export JSONL from the runner and follow the analysis examples in `docs/USER_GUIDE.md`.
