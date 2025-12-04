"""High-level orchestration utilities with token budget integration."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from bailiff.agents.base import AgentSpec, build_responder_map
from bailiff.core.config import CueToggle, Role, TrialConfig
from bailiff.core.logging import LogFactory, default_log_factory
from bailiff.core.session import TrialSession
from bailiff.core.token_budget_simplified import check_run_allowed
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

    def assign_pairs(self, base: TrialConfig, assignments: Iterable[PairAssignment]) -> Iterator[PairPlan]:
        """Generate trial plans from cue assignments."""

        for assignment in assignments:
            control_config = base.copy()
            treatment_config = base.copy()
            # Apply common metadata from assignment
            for config in [control_config, treatment_config]:
                config.cue = assignment.cue
                config.is_placebo = assignment.is_placebo
                config.block_key = assignment.block_key
                config.seed = assignment.seed
            # Apply condition-specific metadata
            control_config.cue_condition = "control"
            control_config.cue_value = assignment.cue.control_value
            treatment_config.cue_condition = "treatment"
            treatment_config.cue_value = assignment.cue.treatment_value
            # Yield the paired plan
            yield PairPlan(
                control=TrialPlan(config=control_config, cue_value=control_config.cue_value or ""),
                treatment=TrialPlan(config=treatment_config, cue_value=treatment_config.cue_value or ""),
            )
