"""High-level orchestration utilities with token budget integration."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from bailiff.agents.base import AgentSpec, build_responder_map
from bailiff.core.config import CueToggle, Person, Role, TrialConfig
from bailiff.core.logging import LogFactory, default_log_factory
from bailiff.core.session import TrialSession
from bailiff.core.token_budget import check_run_allowed
from bailiff.metrics.outcome import PairedOutcome
from bailiff.orchestration.randomization import PairAssignment, block_identifier


@dataclass
class TrialPlan:
    """Defines a single trial run within a paired design."""

    config: TrialConfig
    cue_value: str


@dataclass
class PairPlan:
    """Holds two matched trial plans differing only in cue assignment."""

    control: TrialPlan
    treatment: TrialPlan

    def to_outcome(self, control_verdict: int, treatment_verdict: int) -> PairedOutcome:
        """Convert recorded verdicts into a paired outcome object."""

        return PairedOutcome(control=control_verdict, treatment=treatment_verdict)


@dataclass
class TrialPipeline:
    """Convenience wrapper for generating and executing trial pairs."""

    agents: Dict[Role, AgentSpec]
    log_factory: LogFactory = default_log_factory
    enforce_budget: bool = True

    def build_session(self, config: TrialConfig) -> TrialSession:
        """Construct a TrialSession with responders derived from agent specs."""

        responders = build_responder_map(self.agents)
        return TrialSession(config=config, responders=responders, log_factory=self.log_factory)

    def run_pair(self, plan: PairPlan) -> List:
        """Execute a control/treatment pair and return their logs."""
        
        # Check token budget before running if enforcement is enabled
        if self.enforce_budget:
            # Generate a unique run ID for this pair
            run_id = str(uuid.uuid4())
            
            # Estimate total token usage for the pair
            total_tokens_estimate = self._estimate_pair_token_usage(plan)
            
            # Check if this run would exceed budget limits
            model_identifier = plan.control.config.model_identifier or "default"
            allowed, reason = check_run_allowed(
                run_id=run_id,
                model_identifier=model_identifier,
                api_key_id="pipeline",
                estimated_tokens=total_tokens_estimate
            )
            
            if not allowed:
                raise RuntimeError(f"Token budget exceeded: {reason}")

        control_session = self.build_session(plan.control.config)
        treatment_session = self.build_session(plan.treatment.config)
        control_log = control_session.run()
        treatment_log = treatment_session.run()
        return [control_log, treatment_log]
    
    def _estimate_pair_token_usage(self, plan: PairPlan) -> int:
        """Estimate the total token usage for a trial pair based on budgets."""
        total_estimate = 0
        
        # Estimate based on agent budgets for both control and treatment
        for trial_plan in [plan.control, plan.treatment]:
            config = trial_plan.config
            for role, budget in config.agent_budgets.items():
                # If token budget is specified, use it; otherwise estimate from bytes
                if budget.max_tokens:
                    role_tokens = budget.max_tokens
                else:
                    # Estimate tokens based on byte count (roughly 4 bytes per token)
                    role_tokens = budget.max_bytes // 4
                
                # Account for both prompt and completion tokens
                # Assume prompts are about the same size as completions
                total_estimate += role_tokens * 2
        
        return total_estimate

    def assign_pairs(self, base_config: TrialConfig, assignments: Iterable[PairAssignment]) -> Iterator[PairPlan]:
        """Generate pair plans by applying randomization assignments."""

        block_key = base_config.block_key or block_identifier(
            base_config.case_template.stem, base_config.model_identifier
        )
        yield from self.assign_blocked_pairs({block_key: base_config}, assignments)

    def assign_blocked_pairs(
        self,
        block_configs: Mapping[str, TrialConfig],
        assignments: Iterable[PairAssignment],
    ) -> Iterator[PairPlan]:
        """Generate plans across multiple case × model blocks."""

        block_map: Dict[str, TrialConfig] = dict(block_configs)
        if not block_map:
            return
        cue_maps: Dict[str, Dict[str, CueToggle]] = {
            key: self._cue_lookup(cfg) for key, cfg in block_map.items()
        }
        default_key = next(iter(block_map))
        for assignment in assignments:
            block_key = assignment.block_key or default_key
            base = block_map.get(block_key)
            if base is None:
                raise KeyError(f"No TrialConfig registered for block '{block_key}'.")
            cue_name = assignment.cue_name or base.cue.name
            cue_template = cue_maps[block_key].get(cue_name)
            if cue_template is None:
                raise KeyError(f"Cue '{cue_name}' is not configured for block '{block_key}'.")
            # Helper to resolve Person objects from string values
            def _resolve_person(name: str, template: CueToggle) -> Person:
                if name == template.control_person.name:
                    return template.control_person
                if name == template.treatment_person.name:
                    return template.treatment_person
                # Fallback: create a temporary Person if name doesn't match (e.g. for robust handling)
                return Person(name=name)

            control_person = _resolve_person(assignment.control_value, cue_template)
            treatment_person = _resolve_person(assignment.treatment_value, cue_template)

            control_cue = CueToggle(
                name=cue_template.name,
                control_person=control_person,
                treatment_person=treatment_person,
                metadata=cue_template.metadata,
            )
            treatment_cue = CueToggle(
                name=cue_template.name,
                control_person=treatment_person,
                treatment_person=control_person,
                metadata=cue_template.metadata,
            )
            control_config = self._clone_config(
                base,
                control_cue,
                cue_condition="control",
                cue_value=assignment.control_value,
                assignment=assignment,
                is_treatment=False,
            )
            treatment_config = self._clone_config(
                base,
                treatment_cue,
                cue_condition="treatment",
                cue_value=assignment.treatment_value,
                assignment=assignment,
                is_treatment=True,
            )
            yield PairPlan(
                control=TrialPlan(config=control_config, cue_value=assignment.control_value),
                treatment=TrialPlan(config=treatment_config, cue_value=assignment.treatment_value),
            )

    def _cue_lookup(self, config: TrialConfig) -> Dict[str, CueToggle]:
        """Return a cue lookup including negative controls for a config."""

        lookup: Dict[str, CueToggle] = {config.cue.name: config.cue}
        for placebo in config.negative_controls:
            lookup[placebo.name] = placebo
        return lookup

    def _clone_config(
        self,
        base: TrialConfig,
        cue: CueToggle,
        *,
        cue_condition: str,
        cue_value: str,
        assignment: PairAssignment,
        is_treatment: bool,
    ) -> TrialConfig:
        """Copy a TrialConfig while swapping cue assignment metadata."""

        block_key = assignment.block_key or base.block_key or block_identifier(
            base.case_template.stem, base.model_identifier
        )
        seed = assignment.seed + (1 if is_treatment else 0)
        return TrialConfig(
            case_template=base.case_template,
            cue=cue,
            model_identifier=base.model_identifier,
            backend_name=base.backend_name,
            model_parameters=base.model_parameters,
            seed=seed,
            agent_budgets=base.agent_budgets,
            phase_budgets=base.phase_budgets,
            negative_controls=base.negative_controls,
            cue_condition=cue_condition,
            cue_value=cue_value,
            judge_blinding=base.judge_blinding,
            strict_blinding=base.strict_blinding,
            enforce_role_phase_policy=base.enforce_role_phase_policy,
            notes=base.notes,
            block_key=block_key,
            is_placebo=assignment.is_placebo,
        )
