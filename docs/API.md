# B.A.I.L.I.F.F. API Reference (Core Modules)

This is a concise reference of the primary classes and functions. Import paths are shown; docstrings in code add detail.

## Core
- `bailiff.core.config`
  - `Role`, `Phase` — Enums of roles/phases
  - `AgentBudget(max_bytes, max_tokens=None, max_turns=None)`
  - `PhaseBudget(phase, max_messages=2, allow_interruptions=False)`
  - `CueToggle(name, control_value, treatment_value, metadata={})`
  - `TrialConfig(case_template, cue, model_identifier, backend_name=None, model_parameters=None, seed, agent_budgets, phase_budgets, ..., cue_condition=None, cue_value=None, block_key=None, is_placebo=False, judge_blinding=False)`
  - `DEFAULT_PHASE_ORDER: list[Phase]`
- `bailiff.core.events`
  - `ObjectionRuling` — Enum
  - `EventTag(name, value=None)`
  - `UtteranceLog(role, phase, content, byte_count, token_count, addressed_to, timestamp, interruption=False, objection_raised=False, objection_ruling=None, safety_triggered=False, tags=[])`
  - `TrialLog(trial_id, case_identifier, model_identifier, backend_name, model_parameters, cue_name, cue_condition, cue_value, block_key, is_placebo, seed, started_at, completed_at, utterances=[], verdict=None, sentence=None, schema_version)`
- `bailiff.core.logging`
  - `default_log_factory(config) -> TrialLog`
  - `mark_completed(log) -> None`
- `bailiff.core.session`
  - `TrialSession(config, responders, log_factory, policy_hooks=None)`
    - `.run() -> TrialLog` — executes phases, enforces budgets, tags events, parses verdict/sentence
- `bailiff.core.io`
  - `write_jsonl(logs, path) -> None`
  - `append_jsonl(logs, path) -> None`
  - `read_jsonl(path) -> list[dict]`
  - `RunManifestEntry(...)`
  - `RunManifest(path)`
  - `compute_prompt_hash(*components) -> str`
- `bailiff.core.schema`
  - `SCHEMA_VERSION`
  - `validate_trial_log(record) -> None`

## Orchestration
- `bailiff.orchestration.randomization`
  - `PairAssignment(seed, control_value, treatment_value, cue_name=None, block_key=None, ...)`
  - `RandomizationBlock(case_identifier, model_identifier, cue_name, values, seeds, is_placebo=False)`
  - `block_identifier(case_identifier, model_identifier) -> str`
  - `blocked_permutations(values, seeds, *, cue_name=None, block_key=None, ...) -> Iterator[PairAssignment]`
  - `blockwise_permutations(blocks) -> Iterator[PairAssignment]`
- `bailiff.orchestration.blocks`
  - `resolve_placebos(keys) -> list[CueToggle]`
  - `build_blocks(case_identifier, model_identifier, cues, seeds, placebo_names) -> list[RandomizationBlock]`
- `bailiff.orchestration.pipeline`
  - `TrialPlan(config, cue_value)`
  - `PairPlan(control: TrialPlan, treatment: TrialPlan)`
  - `TrialPipeline(agents, log_factory=default_log_factory)`
    - `.build_session(config) -> TrialSession`
    - `.run_pair(plan) -> list[TrialLog]`
    - `.assign_pairs(base_config, assignments) -> Iterator[PairPlan]`
    - `.assign_blocked_pairs(block_configs, assignments) -> Iterator[PairPlan]`
    - `.assign_blocked_pairs(block_configs, assignments) -> Iterator[PairPlan]`

## Agents
- `bailiff.agents.base`
  - `RetryPolicy(max_retries=2, initial_backoff=1.0, backoff_multiplier=2.0, timeout_seconds=30.0, rate_limit_seconds=0.0)`
  - `BackendTimeoutError`
  - `AgentBackend` — callable protocol `__call__(prompt, **kwargs) -> str`
  - `AgentSpec(role, system_prompt, backend, default_params=None, retry_policy=RetryPolicy())`
    - `.to_responder() -> Callable[[Role, Phase, str], str]`
  - `build_responder_map(specs) -> dict[Role, Callable]`
- `bailiff.agents.prompts`
  - `prompt_for(role) -> str`
- `bailiff.agents.backends`
  - `GroqBackend(model, api_key=None)` — requires `groq` package
  - `GeminiBackend(model, api_key=None)` — requires `google-generativeai`

## Datasets
- `bailiff.datasets.templates`
  - `CaseTemplate(identifier, description, template_path)`
  - `default_cases(root) -> list[CaseTemplate]`
  - `load_case_templates(root) -> list[CaseTemplate]`
  - `cue_catalog() -> dict[str, CueToggle]`
  - `placebo_catalog() -> Iterable[CueToggle]`

## Metrics
- `bailiff.metrics.outcome`
  - `PairedOutcome(control: int, treatment: int)` — `.flip()`
  - `mcnemar_log_odds(pairs) -> (estimate, se)`
  - `flip_rate(pairs) -> float`
  - `normalized_sentence(imposed, minimum, maximum) -> float`
  - `summarize_outcomes(df) -> pd.DataFrame`
- `bailiff.metrics.procedural`
  - `ShareRecord(trial_id, phase, pros_bytes, def_bytes)` — `.total`, `.delta()`
  - `aggregate_share(records) -> float`
  - `correct_measurement(mean_observed, alpha, beta) -> float`
  - `estimate_misclassification(y_true, y_pred) -> (alpha, beta)`
  - `summarize_objections(df) -> pd.DataFrame`
  - `tone_gap(df) -> (control_mean, treatment_mean)`
- `bailiff.metrics.tone`
  - `naive_lexicon_score(text) -> float`
  - `fit_platt(scores, labels) -> PlattParams`
  - `apply_platt(scores, params) -> tuple[float, ...]`
  - `cohen_kappa(a, b) -> float`
  - `expected_calibration_error(probs, labels, bins=10) -> float`

## Analysis Helpers
- `bailiff.analysis.stats`
  - `benjamini_hochberg(pvals, alpha=0.05) -> (rejections, adj_pvals)`
  - `tost_log_odds(est, se, delta=0.1) -> (equivalent, p_lower, p_upper)`
  - `randomization_inference(stat_fn, y_control, y_treat, reps=1000, seed=123) -> float`
  - `wild_cluster_bootstrap(stat_fn, y_control, y_treat, reps=1000, seed=123) -> (p5, p95)`
