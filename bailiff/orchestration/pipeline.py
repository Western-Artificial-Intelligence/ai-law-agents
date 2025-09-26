"""High-level orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence

from bailiff.agents.base import AgentSpec, build_responder_map
from bailiff.core.config import CueToggle, Role, TrialConfig
from bailiff.core.logging import LogFactory, default_log_factory
from bailiff.core.session import TrialSession
from bailiff.metrics.outcome import PairedOutcome
from bailiff.orchestration.randomization import PairAssignment


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

    def build_session(self, config: TrialConfig) -> TrialSession:
        """Construct a TrialSession with responders derived from agent specs."""

        responders = build_responder_map(self.agents)
        return TrialSession(config=config, responders=responders, log_factory=self.log_factory)

    def run_pair(self, plan: PairPlan) -> List:
        """Execute a control/treatment pair and return their logs."""

        sessions = [self.build_session(plan.control.config), self.build_session(plan.treatment.config)]
        logs = [session.run() for session in sessions]
        return logs

    def assign_pairs(self, base_config: TrialConfig, assignments: Iterable[PairAssignment]) -> Iterator[PairPlan]:
        """Generate pair plans by applying randomization assignments."""

        for assignment in assignments:
            control_cue = CueToggle(
                name=base_config.cue.name,
                control_value=assignment.control_value,
                treatment_value=assignment.treatment_value,
                metadata=base_config.cue.metadata,
            )
            treatment_cue = CueToggle(
                name=base_config.cue.name,
                control_value=assignment.treatment_value,
                treatment_value=assignment.control_value,
                metadata=base_config.cue.metadata,
            )
            control_config = TrialConfig(
                case_template=base_config.case_template,
                cue=control_cue,
                model_identifier=base_config.model_identifier,
                seed=assignment.seed,
                agent_budgets=base_config.agent_budgets,
                phase_budgets=base_config.phase_budgets,
                negative_controls=base_config.negative_controls,
                notes=base_config.notes,
            )
            treatment_config = TrialConfig(
                case_template=base_config.case_template,
                cue=treatment_cue,
                model_identifier=base_config.model_identifier,
                seed=assignment.seed + 1,
                agent_budgets=base_config.agent_budgets,
                phase_budgets=base_config.phase_budgets,
                negative_controls=base_config.negative_controls,
                notes=base_config.notes,
            )
            yield PairPlan(
                control=TrialPlan(config=control_config, cue_value=assignment.control_value),
                treatment=TrialPlan(config=treatment_config, cue_value=assignment.treatment_value),
            )
