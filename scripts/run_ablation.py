"""CLI entry point for running ablation studies on trial configurations.

An ablation study systematically varies specific configuration parameters
(phase budgets, prompts, blinding modes, etc.) to isolate their effect on
trial outcomes and procedural metrics.

Usage:
    python scripts/run_ablation.py --config configs/ablation_example.yaml \\
        --out runs/ablation_results.jsonl \\
        --comparison-csv runs/ablation_comparison.csv \\
        --comparison-md runs/ablation_comparison.md

The ablation config YAML defines:
  - base_config: Path to a baseline pilot.yaml
  - ablations: List of named variation sweeps, each with multiple configs
  - repetitions: Number of trial pairs to run per variation (default: 5)
  - output_format: 'csv', 'markdown', or 'both' (default: 'both')
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from bailiff.agents.base import AgentSpec, RetryPolicy
from bailiff.agents.prompts import prompt_for
from bailiff.core.config import AgentBudget, CueToggle, Phase, PhaseBudget, Role, TrialConfig
from bailiff.core.events import TrialLog
from bailiff.core.io import append_jsonl, compute_prompt_hash
from bailiff.datasets.templates import cue_catalog, load_case_templates
from bailiff.orchestration.pipeline import TrialPipeline
from bailiff.orchestration.randomization import blockwise_permutations, RandomizationBlock
from bailiff.metrics.outcome import flip_rate, PairedOutcome
from bailiff.metrics.procedural import ShareRecord, aggregate_share

load_dotenv()


@dataclass
class AblationVariation:
    """Represents a single configuration variation within an ablation sweep."""
    
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    def apply_to_config(self, base_config: TrialConfig) -> TrialConfig:
        """Apply this variation's overrides to a base config."""
        # Create a shallow copy and apply modifications
        config_dict = {
            "case_template": base_config.case_template,
            "cue": base_config.cue,
            "model_identifier": base_config.model_identifier,
            "seed": base_config.seed,
            "agent_budgets": dict(base_config.agent_budgets),
            "phase_budgets": list(base_config.phase_budgets),
            "backend_name": base_config.backend_name,
            "model_parameters": dict(base_config.model_parameters),
            "negative_controls": base_config.negative_controls,
            "judge_blinding": base_config.judge_blinding,
            "strict_blinding": base_config.strict_blinding,
            "enforce_role_phase_policy": base_config.enforce_role_phase_policy,
            "notes": base_config.notes,
        }
        
        # Apply overrides
        for key, value in self.overrides.items():
            if key == "phase_budgets":
                # Handle phase budget updates specially
                phase_budgets = list(base_config.phase_budgets)
                for phase_name, budget_overrides in value.items():
                    phase_enum = Phase(phase_name)
                    # Find and update existing phase budget
                    found = False
                    for i, pb in enumerate(phase_budgets):
                        if pb.phase == phase_enum:
                            phase_budgets[i] = PhaseBudget(
                                phase=phase_enum,
                                max_messages=budget_overrides.get("max_messages", pb.max_messages),
                                allow_interruptions=budget_overrides.get("allow_interruptions", pb.allow_interruptions),
                            )
                            found = True
                            break
                    if not found:
                        # Add new phase budget
                        phase_budgets.append(PhaseBudget(
                            phase=phase_enum,
                            max_messages=budget_overrides.get("max_messages", 2),
                            allow_interruptions=budget_overrides.get("allow_interruptions", False),
                        ))
                config_dict["phase_budgets"] = phase_budgets
            elif key == "agent_budgets":
                # Handle agent budget updates
                agent_budgets = dict(base_config.agent_budgets)
                for role_name, budget_overrides in value.items():
                    role_enum = Role(role_name)
                    existing = agent_budgets.get(role_enum, AgentBudget(max_bytes=1500))
                    agent_budgets[role_enum] = AgentBudget(
                        max_bytes=budget_overrides.get("max_bytes", existing.max_bytes),
                        max_tokens=budget_overrides.get("max_tokens", existing.max_tokens),
                        max_turns=budget_overrides.get("max_turns", existing.max_turns),
                    )
                config_dict["agent_budgets"] = agent_budgets
            elif key in ("judge_blinding", "strict_blinding", "enforce_role_phase_policy"):
                # Boolean flags
                config_dict[key] = bool(value)
            elif key == "notes":
                # String fields
                config_dict[key] = str(value) if value else None
            else:
                # For any other fields, set directly if they exist in TrialConfig
                if key in config_dict:
                    config_dict[key] = value
        
        return TrialConfig(**config_dict)


@dataclass
class AblationSweep:
    """Represents a named collection of configuration variations to test."""
    
    name: str
    variations: List[AblationVariation] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class AblationConfig:
    """Top-level configuration for an ablation study."""
    
    base_config_path: Path
    sweeps: List[AblationSweep] = field(default_factory=list)
    repetitions: int = 5
    output_format: str = "both"  # 'csv', 'markdown', or 'both'


def parse_ablation_yaml(config_path: Path) -> AblationConfig:
    """Parse an ablation configuration YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    base_config_path = Path(data["base_config"])
    repetitions = data.get("repetitions", 5)
    output_format = data.get("output_format", "both")
    
    sweeps = []
    for sweep_data in data.get("ablations", []):
        variations = []
        for var_data in sweep_data.get("variations", []):
            variations.append(AblationVariation(
                name=var_data["name"],
                overrides=var_data.get("overrides", {}),
            ))
        sweeps.append(AblationSweep(
            name=sweep_data["name"],
            variations=variations,
            description=sweep_data.get("description"),
        ))
    
    return AblationConfig(
        base_config_path=base_config_path,
        sweeps=sweeps,
        repetitions=repetitions,
        output_format=output_format,
    )


def load_base_config(config_path: Path) -> TrialConfig:
    """Load the base trial configuration from YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    # Parse case template
    case_template_rel = data.get("case_template", "cases/traffic.yaml")
    if not case_template_rel.startswith("bailiff/"):
        case_template_rel = f"bailiff/datasets/{case_template_rel}"
    case_template = Path(case_template_rel)
    
    # Parse cue
    cue_key = data.get("cue", "name_ethnicity")
    cue_dict = cue_catalog().get(cue_key)
    if not cue_dict:
        raise ValueError(f"Unknown cue key: {cue_key}")
    cue = CueToggle(**cue_dict)
    
    # Parse budgets
    agent_budgets = {}
    for role_name, budget_data in data.get("agent_budgets", {}).items():
        agent_budgets[Role(role_name)] = AgentBudget(
            max_bytes=budget_data.get("max_bytes", 1500),
            max_tokens=budget_data.get("max_tokens"),
            max_turns=budget_data.get("max_turns"),
        )
    
    phase_budgets = []
    for pb_data in data.get("phase_budgets", []):
        phase_budgets.append(PhaseBudget(
            phase=Phase(pb_data["phase"]),
            max_messages=pb_data.get("max_messages", 2),
            allow_interruptions=pb_data.get("allow_interruptions", False),
        ))
    
    # Backend parameters
    backend_params = data.get("backend_params", {})
    
    return TrialConfig(
        case_template=case_template,
        cue=cue,
        model_identifier=data.get("model_identifier", "llama-3.1-8b-instant"),
        seed=data.get("seed", 42),
        agent_budgets=agent_budgets,
        phase_budgets=phase_budgets,
        backend_name=data.get("backend", "echo"),
        model_parameters=backend_params,
        judge_blinding=data.get("judge_blinding", False),
        strict_blinding=data.get("strict_blinding", False),
        enforce_role_phase_policy=data.get("enforce_role_phase_policy", True),
    )


def run_ablation_trial(
    pipeline: TrialPipeline,
    config: TrialConfig,
    seed_offset: int,
) -> tuple[TrialLog, TrialLog]:
    """Run a single paired trial with the given configuration."""
    # Update seed for this repetition
    varied_config = TrialConfig(
        case_template=config.case_template,
        cue=config.cue,
        model_identifier=config.model_identifier,
        seed=config.seed + seed_offset,
        agent_budgets=config.agent_budgets,
        phase_budgets=config.phase_budgets,
        backend_name=config.backend_name,
        model_parameters=config.model_parameters,
        negative_controls=config.negative_controls,
        judge_blinding=config.judge_blinding,
        strict_blinding=config.strict_blinding,
        enforce_role_phase_policy=config.enforce_role_phase_policy,
        notes=config.notes,
    )
    
    # Create randomization block and get assignments
    block = RandomizationBlock(
        case_identifier=config.case_template.stem,
        model_identifier=config.model_identifier,
        cue_toggles=[config.cue],
        placebo_toggles=[],
    )
    
    assignments = list(blockwise_permutations([block], n_replicates=1, base_seed=varied_config.seed))
    
    # Generate and run pair
    pairs = list(pipeline.assign_pairs(varied_config, assignments))
    if not pairs:
        raise RuntimeError("No trial pairs generated")
    
    logs = pipeline.run_pair(pairs[0])
    return logs[0], logs[1]


def extract_metrics(control_log: TrialLog, treatment_log: TrialLog) -> Dict[str, Any]:
    """Extract comparison metrics from a paired trial."""
    metrics = {}
    
    # Verdict outcomes
    control_verdict = 1 if control_log.verdict == "guilty" else 0
    treatment_verdict = 1 if treatment_log.verdict == "guilty" else 0
    metrics["control_verdict"] = control_log.verdict or "unknown"
    metrics["treatment_verdict"] = treatment_log.verdict or "unknown"
    metrics["verdict_flip"] = control_verdict != treatment_verdict
    
    # Sentence comparison
    if control_log.sentence is not None and treatment_log.sentence is not None:
        metrics["control_sentence"] = control_log.sentence
        metrics["treatment_sentence"] = treatment_log.sentence
        metrics["sentence_delta"] = treatment_log.sentence - control_log.sentence
    else:
        metrics["control_sentence"] = None
        metrics["treatment_sentence"] = None
        metrics["sentence_delta"] = None
    
    # Token counts
    control_tokens = sum(u.token_count or 0 for u in control_log.utterances)
    treatment_tokens = sum(u.token_count or 0 for u in treatment_log.utterances)
    metrics["control_total_tokens"] = control_tokens
    metrics["treatment_total_tokens"] = treatment_tokens
    metrics["token_delta"] = treatment_tokens - control_tokens
    
    # Byte counts
    control_bytes = sum(u.byte_count for u in control_log.utterances)
    treatment_bytes = sum(u.byte_count for u in treatment_log.utterances)
    metrics["control_total_bytes"] = control_bytes
    metrics["treatment_total_bytes"] = treatment_bytes
    metrics["byte_delta"] = treatment_bytes - control_bytes
    
    # Utterance counts by role
    for role in [Role.JUDGE, Role.PROSECUTION, Role.DEFENSE]:
        control_count = sum(1 for u in control_log.utterances if u.role == role.value)
        treatment_count = sum(1 for u in treatment_log.utterances if u.role == role.value)
        metrics[f"control_{role.value}_utterances"] = control_count
        metrics[f"treatment_{role.value}_utterances"] = treatment_count
    
    # Procedural events
    control_objections = sum(1 for u in control_log.utterances if u.objection_raised)
    treatment_objections = sum(1 for u in treatment_log.utterances if u.objection_raised)
    metrics["control_objections"] = control_objections
    metrics["treatment_objections"] = treatment_objections
    
    control_interruptions = sum(1 for u in control_log.utterances if u.interruption)
    treatment_interruptions = sum(1 for u in treatment_log.utterances if u.interruption)
    metrics["control_interruptions"] = control_interruptions
    metrics["treatment_interruptions"] = treatment_interruptions
    
    return metrics


def run_ablation_study(
    ablation_config: AblationConfig,
    pipeline: TrialPipeline,
    output_jsonl: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute all ablation variations and collect results."""
    results = []
    
    # Load base configuration
    base_config = load_base_config(ablation_config.base_config_path)
    
    # Run each sweep
    for sweep in ablation_config.sweeps:
        print(f"\n{'='*60}")
        print(f"Ablation Sweep: {sweep.name}")
        if sweep.description:
            print(f"Description: {sweep.description}")
        print(f"{'='*60}\n")
        
        for variation in tqdm(sweep.variations, desc=f"Variations in {sweep.name}"):
            # Apply variation to base config
            varied_config = variation.apply_to_config(base_config)
            varied_config.notes = f"{sweep.name}::{variation.name}"
            
            # Run multiple repetitions
            for rep in range(ablation_config.repetitions):
                try:
                    control_log, treatment_log = run_ablation_trial(
                        pipeline, varied_config, seed_offset=rep
                    )
                    
                    # Save logs if output path provided
                    if output_jsonl:
                        append_jsonl(output_jsonl, [control_log, treatment_log])
                    
                    # Extract metrics
                    metrics = extract_metrics(control_log, treatment_log)
                    
                    # Record result
                    results.append({
                        "sweep": sweep.name,
                        "variation": variation.name,
                        "repetition": rep,
                        "trial_id_control": control_log.trial_id,
                        "trial_id_treatment": treatment_log.trial_id,
                        **metrics,
                    })
                
                except Exception as e:
                    print(f"Error in {sweep.name}::{variation.name} rep {rep}: {e}")
                    continue
    
    return pd.DataFrame(results)


def generate_comparison_tables(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate summary comparison tables from ablation results."""
    
    # Aggregate by sweep and variation
    summary = results_df.groupby(["sweep", "variation"]).agg({
        "verdict_flip": ["mean", "sum"],
        "sentence_delta": ["mean", "std"],
        "token_delta": ["mean", "std"],
        "byte_delta": ["mean", "std"],
        "control_objections": "mean",
        "treatment_objections": "mean",
        "control_interruptions": "mean",
        "treatment_interruptions": "mean",
        "repetition": "count",
    }).reset_index()
    
    # Flatten column names
    summary.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in summary.columns.values]
    summary.rename(columns={"repetition_count": "n_trials"}, inplace=True)
    
    # Create a detailed comparison table
    detail = results_df.groupby(["sweep", "variation"]).agg({
        "control_verdict": lambda x: (x == "guilty").sum(),
        "treatment_verdict": lambda x: (x == "guilty").sum(),
        "verdict_flip": "sum",
        "sentence_delta": ["mean", "std", "min", "max"],
        "token_delta": ["mean", "std"],
        "control_total_tokens": "mean",
        "treatment_total_tokens": "mean",
    }).reset_index()
    
    detail.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in detail.columns.values]
    
    return summary, detail


def save_markdown_table(df: pd.DataFrame, output_path: Path, title: str):
    """Save a DataFrame as a formatted Markdown table."""
    with open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def main():
    """Main entry point for ablation CLI."""
    parser = argparse.ArgumentParser(
        description="Run ablation studies on trial configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to ablation configuration YAML",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSONL output path for all trial logs",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        help="Path to save comparison table as CSV",
    )
    parser.add_argument(
        "--comparison-md",
        type=Path,
        help="Path to save comparison table as Markdown",
    )
    parser.add_argument(
        "--backend",
        default="echo",
        help="Backend to use for trials (default: echo)",
    )
    parser.add_argument(
        "--model",
        help="Model identifier for backend",
    )
    
    args = parser.parse_args()
    
    # Parse ablation config
    print(f"Loading ablation configuration from {args.config}")
    ablation_config = parse_ablation_yaml(args.config)
    
    # Set up pipeline with Echo backend
    from scripts.run_pilot_trial import EchoBackend, _load_backend, Backend
    
    backend_choice = Backend(args.backend)
    backend = EchoBackend() if backend_choice == Backend.ECHO else _load_backend(
        backend_choice, args.model or "llama3-8b-8192", {}, {}
    )
    
    retry_policy = RetryPolicy(
        timeout_seconds=30.0,
        max_retries=2,
        backoff_seconds=1.0,
        backoff_multiplier=2.0,
        rate_limit_seconds=0.0,
    )
    
    agents = {
        role: AgentSpec(
            role=role,
            system_prompt=prompt_for(role),
            backend=backend,
            retry_policy=retry_policy,
        )
        for role in [Role.JUDGE, Role.PROSECUTION, Role.DEFENSE]
    }
    
    pipeline = TrialPipeline(agents=agents)
    
    # Run ablation study
    print(f"\nRunning ablation study with {ablation_config.repetitions} repetitions per variation")
    results_df = run_ablation_study(ablation_config, pipeline, output_jsonl=args.out)
    
    if results_df.empty:
        print("No results collected. Exiting.")
        return 1
    
    # Generate comparison tables
    print("\nGenerating comparison tables...")
    summary, detail = generate_comparison_tables(results_df)
    
    # Save outputs
    if args.comparison_csv or ablation_config.output_format in ("csv", "both"):
        csv_path = args.comparison_csv or Path("ablation_comparison.csv")
        summary.to_csv(csv_path, index=False)
        print(f"Saved summary table to {csv_path}")
        
        detail_path = csv_path.with_stem(f"{csv_path.stem}_detail")
        detail.to_csv(detail_path, index=False)
        print(f"Saved detail table to {detail_path}")
    
    if args.comparison_md or ablation_config.output_format in ("markdown", "both"):
        md_path = args.comparison_md or Path("ablation_comparison.md")
        with open(md_path, "w") as f:
            f.write("# Ablation Study Results\n\n")
            f.write("## Summary Table\n\n")
            f.write(summary.to_markdown(index=False))
            f.write("\n\n## Detailed Comparison\n\n")
            f.write(detail.to_markdown(index=False))
            f.write("\n")
        print(f"Saved comparison tables to {md_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
