"""Session orchestration for multi-agent trials."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, Optional, Tuple

from .config import DEFAULT_PHASE_ORDER, Phase, Role, TrialConfig
from .events import TrialLog, UtteranceLog
from .logging import mark_completed
from pathlib import Path
import re
import yaml

AgentResponder = Callable[[Role, Phase, str], str]


@dataclass
class TrialSession:
    """Coordinates role agents to execute a single trial under constraints."""

    config: TrialConfig
    responders: Dict[Role, AgentResponder]
    log_factory: Callable[[TrialConfig], TrialLog]
    policy_hooks: Optional[Dict[str, Callable[[TrialLog], None]]] = None
    _log: Optional[TrialLog] = field(default=None, init=False, repr=False)
    _bytes_used: Dict[Role, int] = field(default_factory=dict, init=False, repr=False)
    _case_text: Optional[str] = field(default=None, init=False, repr=False)

    def run(self) -> TrialLog:
        """Execute the state machine and return the populated trial log."""

        self._log = self.log_factory(self.config)
        self._bytes_used = {role: 0 for role in Role}
        self._case_text = self._load_and_render_case()
        for phase in DEFAULT_PHASE_ORDER:
            self._run_phase(phase)
        if self.policy_hooks:
            for hook in self.policy_hooks.values():
                hook(self._log)
        mark_completed(self._log)
        return self._log

    def _run_phase(self, phase: Phase) -> None:
        """Run a phase by iterating over active roles."""

        active_roles = self._roles_for_phase(phase)
        pb = self.config.phase_budget_for(phase)
        for role in active_roles:
            max_msgs = max(1, pb.max_messages)
            for _ in range(max_msgs):
                content = self._emit(role, phase)
                # Enforce byte budget per role by truncation
                content = self._apply_byte_budget(role, content)
                record = self._build_record(role, phase, content)
                self._apply_event_tagging(record)
                self._log.append(record)  # type: ignore[arg-type]
            # Verdict/sentence parsing after phase for judge
            if role is Role.JUDGE and phase is Phase.VERDICT:
                self._parse_and_set_verdict_sentence()

    def _roles_for_phase(self, phase: Phase) -> Iterable[Role]:
        """Return roles expected to contribute during a phase."""

        if phase in (Phase.OPENING, Phase.CLOSING):
            return (Role.PROSECUTION, Role.DEFENSE)
        if phase in (Phase.DIRECT, Phase.REDIRECT):
            return (Role.PROSECUTION,)
        if phase is Phase.CROSS:
            return (Role.DEFENSE,)
        return (Role.JUDGE,)

    def _emit(self, role: Role, phase: Phase) -> str:
        """Invoke a responder while respecting budgets (enforced later)."""

        responder = self.responders[role]
        prompt = self._build_prompt(role, phase)
        return responder(role, phase, prompt)

    def _build_prompt(self, role: Role, phase: Phase) -> str:
        """Construct the shared prompt context passed to an agent responder."""

        cue_line = ""
        if not (self.config.judge_blinding and role is Role.JUDGE):
            cue_line = f"\nCue: {self.config.cue.name} = {self.config.cue_value}"
        case_text = self._case_text or str(self.config.case_template)
        if self.config.judge_blinding and role is Role.JUDGE and (self.config.cue_value or ""):
            cv = self.config.cue_value or ""
            case_text = case_text.replace(cv, "[REDACTED]")
        return (
            f"Case:\n{case_text}{cue_line}\n"
            f"Phase: {phase.value}\n"
            f"Role: {role.value}"
        )

    def _build_record(self, role: Role, phase: Phase, content: str) -> UtteranceLog:
        """Create a minimal log entry for downstream metric extraction."""

        rec = UtteranceLog(
            role=role,
            phase=phase,
            content=content,
            byte_count=len(content.encode("utf-8")),
            token_count=None,
            addressed_to=None,
            timestamp=datetime.utcnow(),
        )
        return rec

    # --- Helpers ---
    def _apply_byte_budget(self, role: Role, content: str) -> str:
        budget = self.config.budget_for(role)
        current = self._bytes_used.get(role, 0)
        remaining = max(0, budget.max_bytes - current)
        encoded = content.encode("utf-8")
        if len(encoded) > remaining:
            clipped = encoded[:remaining]
            content = clipped.decode("utf-8", errors="ignore")
        used = len(content.encode("utf-8"))
        self._bytes_used[role] = current + used
        return content

    def _load_and_render_case(self) -> str:
        path = Path(self.config.case_template)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return str(path)
        cue_val = self.config.cue_value or ""
        summary = str(data.get("summary", "")).replace("{{ cue_value }}", cue_val)
        facts = data.get("facts", []) or []
        facts_text = "\n".join(f"- {str(f).replace('{{ cue_value }}', cue_val)}" for f in facts)
        charges = data.get("charges", []) or []
        charges_text = ", ".join(map(str, charges))
        # witnesses optional
        wit = data.get("witnesses", {}) or {}
        prose = []
        for side in ("prosecution", "defense"):
            for w in wit.get(side, []) or []:
                name = str(w.get("name", "Witness"))
                stmt = str(w.get("statement", "")).replace("{{ cue_value }}", cue_val)
                prose.append(f"{side.title()} witness {name}: {stmt}")
        witnesses_text = "\n".join(prose)
        return (
            f"Summary: {summary}\n"
            f"Charges: {charges_text}\n"
            f"Facts:\n{facts_text}\n"
            f"Witnesses:\n{witnesses_text}"
        )

    _OBJECTION_RE = re.compile(r"\b(objection)\b", re.IGNORECASE)
    _SUSTAIN_RE = re.compile(r"\b(sustain|sustained)\b", re.IGNORECASE)
    _OVERRULE_RE = re.compile(r"\b(overrule|overruled)\b", re.IGNORECASE)
    _INTERRUPT_RE = re.compile(r"\b(interrupt|interruption)\b", re.IGNORECASE)

    def _apply_event_tagging(self, record: UtteranceLog) -> None:
        text = record.content
        if self._OBJECTION_RE.search(text):
            record.objection_raised = True
            if self._SUSTAIN_RE.search(text):
                from .events import ObjectionRuling
                record.objection_ruling = ObjectionRuling.SUSTAINED
            elif self._OVERRULE_RE.search(text):
                from .events import ObjectionRuling
                record.objection_ruling = ObjectionRuling.OVERRULED
        if self._INTERRUPT_RE.search(text):
            record.interruption = True

    def _parse_and_set_verdict_sentence(self) -> None:
        assert self._log is not None
        # naive parse from latest judge utterance in VERDICT phase
        judge_utts = [u for u in self._log.utterances if u.role is Role.JUDGE and u.phase is Phase.VERDICT]
        if not judge_utts:
            return
        text = judge_utts[-1].content.lower()
        verdict = None
        if "not guilty" in text:
            verdict = "not_guilty"
        elif "guilty" in text:
            verdict = "guilty"
        self._log.verdict = verdict
        # sentence extraction placeholder: looks for "sentence" and a number
        m = re.search(r"sentence[^0-9]*([0-9]+)", text)
        if m:
            self._log.sentence = m.group(1)
