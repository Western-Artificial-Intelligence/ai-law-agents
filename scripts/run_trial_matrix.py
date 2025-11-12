"""Batch runner for executing multi-case/model cue pairs with manifest output."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from bailiff.agents.base import AgentSpec
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


@dataclass
class ModelSpec:
    backend: str
    model_identifier: str
    params: Dict[str, object]


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
    for entry in cfg.get("models", []):
        backend = entry.get("backend", "echo")
        model_identifier = entry.get("model") or backend
        params = entry.get("params", {})
        models.append(ModelSpec(backend=backend, model_identifier=model_identifier, params=params))
    if not models:
        models.append(ModelSpec(backend="echo", model_identifier="echo", params={}))
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


def load_backend(backend: str, model: str):
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
        return GroqBackend(model=model)
    if backend == "gemini":
        try:
            from bailiff.agents.backends import GeminiBackend  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailable(f"Gemini backend unavailable: {exc}") from exc
        return GeminiBackend(model=model)
    raise BackendUnavailable(f"Unsupported backend choice: {backend}")


def build_pipeline(model: ModelSpec) -> TrialPipeline:
    backend_impl = load_backend(model.backend, model.model_identifier)
    agents = {
        role: AgentSpec(role=role, system_prompt=prompt_for(role), backend=backend_impl, default_params=model.params)
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


def execute_job(
    job: BatchJob,
    budgets: Dict[Role, AgentBudget],
    phase_budgets: List[PhaseBudget],
    out_path: Path,
    manifest: RunManifest,
    max_retries: int,
    backoff_seconds: float,
) -> int:
    case_identifier = job.case.template.stem
    block_key = block_identifier(case_identifier, job.model.model_identifier)
    base_config = TrialConfig(
        case_template=job.case.template,
        cue=job.case.cue,
        model_identifier=job.model.model_identifier,
        seed=job.seed,
        agent_budgets=budgets,
        phase_budgets=phase_budgets,
        negative_controls=tuple(job.case.placebo_toggles),
        judge_blinding=job.case.judge_blinding,
        block_key=block_key,
        notes=job.case.notes,
    )
    pipeline = build_pipeline(job.model)
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
                        prompt_hash=compute_prompt_hash(
                            case_blob,
                            assignment.control_value,
                            assignment.treatment_value,
                            job.model.model_identifier,
                            job.model.backend,
                        ),
                        params=job.model.params,
                        trial_ids=tuple(log.trial_id for log in logs),
                        log_path=str(out_path),
                        retries=attempt,
                    )
                )
                completed += 1
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

    print(f"Completed {completed} paired assignments; manifest now has {len(manifest)} entries.")


if __name__ == "__main__":
    main()
