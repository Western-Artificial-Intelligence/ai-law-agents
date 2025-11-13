"""High-level orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

from bailiff.agents.base import AgentSpec, build_responder_map
from bailiff.core.config import CueToggle, Role, TrialConfig
from bailiff.core.logging import LogFactory, default_log_factory
from bailiff.core.session import TrialSession
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
            control_cue = CueToggle(
                name=cue_template.name,
                control_value=assignment.control_value,
                treatment_value=assignment.treatment_value,
                metadata=cue_template.metadata,
            )
            treatment_cue = CueToggle(
                name=cue_template.name,
                control_value=assignment.treatment_value,
                treatment_value=assignment.control_value,
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
