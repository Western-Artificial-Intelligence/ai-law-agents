"""Batch runner for executing multi-case/model cue pairs with manifest output."""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from bailiff.agents.base import AgentSpec, RetryPolicy
from bailiff.agents.groq_pool import GroqKeyPool
from bailiff.agents.prompts import prompt_for
from bailiff.core.config import AgentBudget, CueToggle, Phase, PhaseBudget, Role, TrialConfig
from bailiff.core.io import RunManifest, RunManifestEntry, append_jsonl, compute_prompt_hash
from bailiff.core.logging import default_log_factory
from bailiff.datasets.templates import cue_catalog
from bailiff.orchestration.blocks import build_blocks, resolve_placebos
from bailiff.orchestration.pipeline import TrialPipeline
from bailiff.orchestration.randomization import block_identifier, blockwise_permutations


class BackendUnavailable(RuntimeError):
    """Raised when a requested backend is missing optional deps."""


GROQ_LOG_EVERY_COUNT = 50
GROQ_LOG_EVERY_SECONDS = 300.0


def _as_bool(value: object, *, field_name: str) -> bool:
    """Coerce common bool-like values into bool."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y", "on"}:
            return True
        if norm in {"0", "false", "no", "n", "off"}:
            return False
    raise BackendUnavailable(f"Invalid boolean for {field_name}: {value!r}")


def _summarize_groq_pool(summary: List[Dict[str, object]]) -> Dict[str, object]:
    inflight_total = sum(int(item.get("inflight", 0)) for item in summary)
    keys_in_use = sum(1 for item in summary if int(item.get("inflight", 0)) > 0)
    rate_limit_events = sum(int(item.get("consecutive_rate_limits", 0)) for item in summary)
    backoff_active = sum(1 for item in summary if float(item.get("backoff_remaining", 0)) > 0)
    max_backoff = max((float(item.get("backoff_remaining", 0)) for item in summary), default=0.0)
    total_uses = sum(int(item.get("total_uses", 0)) for item in summary)
    return {
        "keys": len(summary),
        "keys_in_use": keys_in_use,
        "inflight": inflight_total,
        "rate_limit_events": rate_limit_events,
        "backoff_active": backoff_active,
        "max_backoff_seconds": round(max_backoff, 2),
        "total_uses": total_uses,
    }


def _log_groq_pool(pool: GroqKeyPool, label: str, completed_pairs: Optional[int] = None) -> None:
    summary = pool.summary()
    totals = _summarize_groq_pool(summary)
    if completed_pairs is not None:
        totals["completed_pairs"] = completed_pairs
    print(f"[GROQ] Pool {label} totals={totals}")
    print(f"[GROQ] Pool {label} keys={json.dumps(summary, sort_keys=True)}")


class GroqPoolLogger:
    def __init__(self, log_every_count: int, log_every_seconds: float) -> None:
        self._log_every_count = log_every_count
        self._log_every_seconds = log_every_seconds
        self._lock = threading.Lock()
        self._last_log_count = 0
        self._last_log_time = time.monotonic()
        self._total_completed = 0
        self._stop_event = threading.Event()
        self._timer_thread: Optional[threading.Thread] = None

    def log_start(self, pool: GroqKeyPool) -> None:
        with self._lock:
            self._last_log_time = time.monotonic()
            total = self._total_completed
        _log_groq_pool(pool, "start", completed_pairs=total)

    def log_end(self, pool: GroqKeyPool) -> None:
        total = self._get_total()
        _log_groq_pool(pool, "end", completed_pairs=total)

    def start_timer(self, pool: GroqKeyPool) -> None:
        if self._timer_thread is not None:
            return
        self._stop_event.clear()
        self._timer_thread = threading.Thread(target=self._run_timer, args=(pool,), daemon=True)
        self._timer_thread.start()

    def stop_timer(self) -> None:
        self._stop_event.set()
        if self._timer_thread is not None:
            self._timer_thread.join()
            self._timer_thread = None

    def record_completed(self, delta: int, pool: GroqKeyPool) -> None:
        now = time.monotonic()
        should_log = False
        total = 0
        with self._lock:
            self._total_completed += delta
            total = self._total_completed
            if (
                total - self._last_log_count >= self._log_every_count
                or now - self._last_log_time >= self._log_every_seconds
            ):
                self._last_log_count = total
                self._last_log_time = now
                should_log = True
        if should_log:
            _log_groq_pool(pool, "progress", completed_pairs=total)

    def _get_total(self) -> int:
        with self._lock:
            return self._total_completed

    def _run_timer(self, pool: GroqKeyPool) -> None:
        while not self._stop_event.wait(self._log_every_seconds):
            self._log_if_due(pool)

    def _log_if_due(self, pool: GroqKeyPool) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_log_time < self._log_every_seconds:
                return
            self._last_log_time = now
            total = self._total_completed
        _log_groq_pool(pool, "progress", completed_pairs=total)


@dataclass
class ModelSpec:
    backend: str
    model_identifier: str
    params: Dict[str, object]
    retry_policy: RetryPolicy


@dataclass
class CaseSpec:
    template: Path
    cue: CueToggle
    placebo_toggles: List[CueToggle]
    judge_blinding: bool
    notes: Optional[str]


@dataclass
class BatchJob:
    case: CaseSpec
    model: ModelSpec
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of paired trials and write a manifest.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config describing cases/models/seeds.")
    parser.add_argument("--out", type=Path, help="Path to JSONL logs (overrides config).")
    parser.add_argument("--manifest", type=Path, help="Path to manifest JSONL (overrides config).")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_budgets(cfg: dict) -> Dict[Role, AgentBudget]:
    defaults = {
        Role.JUDGE: AgentBudget(max_bytes=1500),
        Role.PROSECUTION: AgentBudget(max_bytes=1800),
        Role.DEFENSE: AgentBudget(max_bytes=1800),
    }
    agent_cfg = cfg.get("agent_budgets", {})
    for role, key in ((Role.JUDGE, "judge"), (Role.PROSECUTION, "prosecution"), (Role.DEFENSE, "defense")):
        role_cfg = agent_cfg.get(key, {})
        defaults[role] = AgentBudget(
            max_bytes=int(role_cfg.get("max_bytes", defaults[role].max_bytes)),
            max_tokens=role_cfg.get("max_tokens"),
            max_turns=role_cfg.get("max_turns"),
        )
    return defaults


def build_phase_budgets(cfg: dict) -> List[PhaseBudget]:
    entries = cfg.get("phase_budgets")
    if not entries:
        return [PhaseBudget(phase=phase) for phase in Phase]
    budgets: List[PhaseBudget] = []
    for item in entries:
        if isinstance(item, dict):
            budgets.append(
                PhaseBudget(
                    phase=Phase(item["phase"]),
                    max_messages=int(item.get("max_messages", 2)),
                    allow_interruptions=bool(item.get("allow_interruptions", False)),
                )
            )
        else:
            budgets.append(PhaseBudget(phase=Phase(str(item))))
    return budgets


def build_model_specs(cfg: dict) -> List[ModelSpec]:
    models = []
    default_policy = cfg.get("backend_policy", {}) or {}
    for entry in cfg.get("models", []):
        backend = entry.get("backend", "echo")
        model_identifier = entry.get("model") or backend
        params = dict(entry.get("params", {}) or {})
        policy_cfg = entry.get("backend_policy", default_policy) or {}
        retry_policy = RetryPolicy(
            max_retries=int(policy_cfg.get("max_retries", 2)),
            initial_backoff=float(policy_cfg.get("backoff_seconds", 1.0)),
            backoff_multiplier=float(policy_cfg.get("backoff_multiplier", 2.0)),
            timeout_seconds=float(policy_cfg.get("timeout_seconds", 30.0)),
            rate_limit_seconds=float(policy_cfg.get("rate_limit_seconds", 0.0)),
        )
        models.append(
            ModelSpec(
                backend=backend,
                model_identifier=model_identifier,
                params=params,
                retry_policy=retry_policy,
            )
        )
    if not models:
        models.append(
            ModelSpec(
                backend="echo",
                model_identifier="echo",
                params={},
                retry_policy=RetryPolicy(),
            )
        )
    return models


def build_case_specs(cfg: dict) -> List[CaseSpec]:
    cases = []
    catalog = cue_catalog()
    default_cue_key = cfg.get("cue", "name_ethnicity")
    default_placebos = cfg.get("placebos", [])
    for entry in cfg.get("cases", []):
        cue_key = entry.get("cue", default_cue_key)
        cue = catalog.get(cue_key)
        if cue is None:
            raise KeyError(f"Unknown cue key: {cue_key}")
        template = Path(entry["template"]).resolve()
        placebo_keys = entry.get("placebos", default_placebos)
        placebo_toggles = resolve_placebos(placebo_keys)
        cases.append(
            CaseSpec(
                template=template,
                cue=cue,
                placebo_toggles=placebo_toggles,
                judge_blinding=bool(entry.get("judge_blinding", cfg.get("judge_blinding", False))),
                notes=entry.get("notes"),
            )
        )
    if not cases:
        raise ValueError("Provide at least one case entry under 'cases'.")
    return cases


def load_backend(
    backend: str,
    model: str,
    params: Optional[Dict[str, object]] = None,
    groq_pool: Optional[GroqKeyPool] = None,
):
    runtime_params = params or {}
    if backend == "echo":
        class EchoBackend:
            def __call__(self, prompt: str, **_: object) -> str:
                return f"[ECHO]\n{prompt}"

        return EchoBackend()
    if backend == "groq":
        try:
            from bailiff.agents.backends import GroqBackend  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailable(f"Groq backend unavailable: {exc}") from exc
        backend_impl = GroqBackend(model=model)
        if groq_pool is not None:
            setattr(backend_impl, "_pool", groq_pool)
        return backend_impl
    if backend == "gemini":
        try:
            from bailiff.agents.backends import GeminiBackend  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailable(f"Gemini backend unavailable: {exc}") from exc
        return GeminiBackend(model=model)
    if backend == "local":
        try:
            from bailiff.agents.backends_local import LlamaCppBackend, LocalTransformersBackend  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailable(f"Local backend unavailable: {exc}") from exc
        provider = str(runtime_params.pop("provider", "transformers")).lower()
        if provider == "llama_cpp":
            model_path = str(runtime_params.pop("model_path", model))
            if not model_path:
                raise BackendUnavailable("model_path is required for llama_cpp local backend.")
            n_ctx = int(runtime_params.pop("n_ctx", 2048))
            n_threads_value = runtime_params.pop("n_threads", None)
            n_threads = int(n_threads_value) if n_threads_value is not None else None
            return LlamaCppBackend(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        model_name = str(runtime_params.pop("model_name", model))
        if not model_name:
            raise BackendUnavailable("model_name is required for transformers local backend.")
        device = runtime_params.pop("device", None)
        load_in_4bit = _as_bool(runtime_params.pop("load_in_4bit", False), field_name="load_in_4bit")
        return LocalTransformersBackend(
            model_name_or_path=model_name,
            device=device,
            load_in_4bit=load_in_4bit,
        )
    raise BackendUnavailable(f"Unsupported backend choice: {backend}")


def build_pipeline(model: ModelSpec, groq_pool: Optional[GroqKeyPool]) -> TrialPipeline:
    call_params = dict(model.params)
    backend_impl = load_backend(model.backend, model.model_identifier, call_params, groq_pool)
    agents = {
        role: AgentSpec(
            role=role,
            system_prompt=prompt_for(role),
            backend=backend_impl,
            default_params=call_params,
            retry_policy=model.retry_policy,
        )
        for role in Role
    }
    return TrialPipeline(agents=agents, log_factory=default_log_factory)


def compute_run_id(case_identifier: str, model_identifier: str, cue_name: str, seed: int, backend: str) -> str:
    token = f"{case_identifier}|{model_identifier}|{cue_name}|{seed}|{backend}"
    return compute_prompt_hash(token)


def case_text_for_hash(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.name


def _prompt_hash_for_log(log) -> str:
    return compute_prompt_hash(
        log.trial_id,
        log.case_identifier,
        log.cue_name,
        log.cue_value or "",
        log.model_identifier,
        log.backend_name or "",
        log.cue_condition or "",
    )


def execute_job(
    job: BatchJob,
    budgets: Dict[Role, AgentBudget],
    phase_budgets: List[PhaseBudget],
    out_path: Path,
    manifest: RunManifest,
    max_retries: int,
    backoff_seconds: float,
    groq_pool: Optional[GroqKeyPool],
    groq_logger: Optional[GroqPoolLogger],
) -> int:
    case_identifier = job.case.template.stem
    block_key = block_identifier(case_identifier, job.model.model_identifier)
    base_config = TrialConfig(
        case_template=job.case.template,
        cue=job.case.cue,
        model_identifier=job.model.model_identifier,
        backend_name=job.model.backend,
        model_parameters=dict(job.model.params),
        seed=job.seed,
        agent_budgets=budgets,
        phase_budgets=phase_budgets,
        negative_controls=tuple(job.case.placebo_toggles),
        judge_blinding=job.case.judge_blinding,
        block_key=block_key,
        notes=job.case.notes,
    )
    pipeline = build_pipeline(job.model, groq_pool)
    placebo_names = [toggle.name for toggle in job.case.placebo_toggles]
    cues_for_blocks: List[CueToggle] = [job.case.cue, *job.case.placebo_toggles]
    assignments = list(
        blockwise_permutations(
            build_blocks(
                case_identifier,
                job.model.model_identifier,
                cues_for_blocks,
                seeds=[job.seed],
                placebo_names=placebo_names,
            )
        )
    )
    case_blob = case_text_for_hash(job.case.template)
    completed = 0
    for assignment in assignments:
        cue_name = assignment.cue_name or job.case.cue.name
        run_id = compute_run_id(case_identifier, job.model.model_identifier, cue_name, assignment.seed, job.model.backend)
        if manifest.has_run(run_id):
            continue
        attempt = 0
        while True:
            try:
                plan_iter = pipeline.assign_pairs(base_config, [assignment])
                plan = next(plan_iter)
                logs = pipeline.run_pair(plan)
                control_hash = _prompt_hash_for_log(logs[0])
                treatment_hash = _prompt_hash_for_log(logs[1])
                pair_hash = compute_prompt_hash(control_hash, treatment_hash, case_blob)
                append_jsonl(logs, out_path)
                manifest.append(
                    RunManifestEntry(
                        run_id=run_id,
                        case_identifier=case_identifier,
                        model_identifier=job.model.model_identifier,
                        backend=job.model.backend,
                        cue_name=cue_name,
                        cue_control=assignment.control_value,
                        cue_treatment=assignment.treatment_value,
                        control_seed=assignment.seed,
                        treatment_seed=assignment.seed + 1,
                        block_key=assignment.block_key,
                        is_placebo=assignment.is_placebo,
                        prompt_hash=pair_hash,
                        prompt_hash_control=control_hash,
                        prompt_hash_treatment=treatment_hash,
                        params=job.model.params,
                        trial_ids=tuple(log.trial_id for log in logs),
                        log_path=str(out_path),
                        retries=attempt,
                    )
                )
                completed += 1
                if job.model.backend == "groq" and groq_logger and groq_pool:
                    groq_logger.record_completed(1, groq_pool)
                break
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    manifest.append(
                        RunManifestEntry(
                            run_id=run_id,
                            case_identifier=case_identifier,
                            model_identifier=job.model.model_identifier,
                            backend=job.model.backend,
                            cue_name=cue_name,
                            cue_control=assignment.control_value,
                            cue_treatment=assignment.treatment_value,
                            control_seed=assignment.seed,
                            treatment_seed=assignment.seed + 1,
                            block_key=assignment.block_key,
                            is_placebo=assignment.is_placebo,
                            prompt_hash="failed",
                            prompt_hash_control=None,
                            prompt_hash_treatment=None,
                            params=job.model.params,
                            trial_ids=(),
                            log_path=str(out_path),
                            status="failed",
                            retries=attempt,
                        )
                    )
                    break
                time.sleep(backoff_seconds * attempt)
    return completed


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_path = Path(args.out or cfg.get("out_logs") or "runs/batch_logs.jsonl").resolve()
    manifest_path = Path(args.manifest or cfg.get("manifest") or out_path.with_suffix(".manifest.jsonl")).resolve()
    budgets = build_budgets(cfg)
    phase_budgets = build_phase_budgets(cfg)
    cases = build_case_specs(cfg)
    models = build_model_specs(cfg)
    seeds = [int(s) for s in cfg.get("seeds", [cfg.get("seed", 42)])]
    concurrency = int(cfg.get("concurrency", 1))
    max_retries = int(cfg.get("max_retries", 2))
    backoff_seconds = float(cfg.get("backoff_seconds", 2.0))

    groq_pool = None
    groq_logger = None
    if any(model.backend == "groq" for model in models):
        groq_pool = GroqKeyPool.from_env()
        groq_logger = GroqPoolLogger(GROQ_LOG_EVERY_COUNT, GROQ_LOG_EVERY_SECONDS)
        groq_logger.log_start(groq_pool)
        groq_logger.start_timer(groq_pool)

    jobs: List[BatchJob] = []
    for case in cases:
        for model in models:
            for seed in seeds:
                jobs.append(BatchJob(case=case, model=model, seed=seed))

    manifest = RunManifest(manifest_path)
    print(f"Starting batch: {len(jobs)} jobs, output={out_path}, manifest={manifest_path}")
    completed = 0
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        future_map = {
            executor.submit(
                execute_job,
                job,
                budgets,
                phase_budgets,
                out_path,
                manifest,
                max_retries,
                backoff_seconds,
                groq_pool,
                groq_logger,
            ): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                completed += future.result()
            except BackendUnavailable as exc:
                raise SystemExit(str(exc)) from exc
            except Exception as exc:
                print(f"[WARN] Job failed for {job.case.template.name} ({job.model.model_identifier}): {exc}")

    if groq_logger and groq_pool:
        groq_logger.stop_timer()
        groq_logger.log_end(groq_pool)

    print(f"Completed {completed} paired assignments; manifest now has {len(manifest)} entries.")


if __name__ == "__main__":
    main()
