"""Session orchestration for multi-agent trials."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, Optional, Tuple

from .config import DEFAULT_PHASE_ORDER, Phase, PolicyViolation, Role, TrialConfig
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
    _tokens_used: Dict[Role, int] = field(default_factory=dict, init=False, repr=False)
    _case_text: Optional[str] = field(default=None, init=False, repr=False)
    # NEW: Tracks count of each type of policy violation during trial execution
    # Format: {"interruption_not_allowed": 2, "judge_cue_exposure": 1}
    _policy_violations: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def run(self) -> TrialLog:
        """Execute the state machine and return the populated trial log."""

        self._log = self.log_factory(self.config)
        self._bytes_used = {role: 0 for role in Role}
        self._tokens_used = {role: 0 for role in Role}
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
            # NEW: Validate role-phase policy enforcement
            # Ensures roles only speak in authorized phases (e.g., judge only in verdict)
            if self.config.enforce_role_phase_policy:
                self._validate_role_for_phase(role, phase)
            
            max_msgs = max(1, pb.max_messages)
            for _ in range(max_msgs):
                content = self._emit(role, phase)
                content, token_count = self._apply_role_budgets(role, content)
                record = self._build_record(role, phase, content, token_count)
                self._apply_event_tagging(record)
                
                # NEW: Enforce interruption policy based on phase configuration
                # If agent generated interruption text but phase doesn't allow it, block it
                if record.interruption and not pb.allow_interruptions:
                    self._record_violation(PolicyViolation.INTERRUPTION_NOT_ALLOWED)
                    # Strip interruption content - replace with policy notice
                    record.content = "[INTERRUPTION BLOCKED - NOT ALLOWED IN THIS PHASE]"
                    record.interruption = False
                
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
        """Construct the shared prompt context passed to an agent responder.
        
        NEW: Enhanced with stricter judge blinding enforcement to ensure demographic
        cues never reach the judge under blinding conditions.
        """

        cue_line = ""
        case_text = self._case_text or str(self.config.case_template)
        
        # NEW: Enhanced judge blinding enforcement
        # Prevents demographic cue exposure to judges when blinding is enabled
        if self.config.judge_blinding and role is Role.JUDGE:
            # hides cue from judge
            cue_line = ""
            
            # NEW: Strict blinding mode - completely strip ALL cue-related content
            # This redacts both control AND treatment values, not just the active one
            if self.config.strict_blinding:
                # Remove all instances of cue values (control and treatment)
                cv = self.config.cue_value or ""
                if cv:  # cue value exists
                    case_text = case_text.replace(cv, "[REDACTED]")
                # Also redact the opposite condition value to prevent any leakage
                if self.config.cue.control_value and self.config.cue.control_value != cv:
                    case_text = case_text.replace(self.config.cue.control_value, "[REDACTED]")
                if self.config.cue.treatment_value and self.config.cue.treatment_value != cv:
                    case_text = case_text.replace(self.config.cue.treatment_value, "[REDACTED]")
                # NEW: Verify no cue leakage - record violation if any cue text remains
                if self._detect_cue_in_text(case_text, role):
                    self._record_violation(PolicyViolation.JUDGE_CUE_EXPOSURE)
            else:
                # Standard blinding: redact only active cue value
                cv = self.config.cue_value or ""
                if cv:
                    case_text = case_text.replace(cv, "[REDACTED]")
        
        # includes cue in prompt if role is not judge, or role is judge but judge is not blinded
        else:
            cue_line = f"\nCue: {self.config.cue.name} = {self.config.cue_value}"
        
        return (
            f"Case:\n{case_text}{cue_line}\n"
            f"Phase: {phase.value}\n"
            f"Role: {role.value}"
        )

    def _build_record(self, role: Role, phase: Phase, content: str, token_count: int) -> UtteranceLog:
        """Create a minimal log entry for downstream metric extraction."""

        rec = UtteranceLog(
            role=role,
            phase=phase,
            content=content,
            byte_count=len(content.encode("utf-8")),
            token_count=token_count,
            addressed_to=None,
            timestamp=datetime.utcnow(),
        )
        return rec

    # --- Helpers ---
    def _apply_role_budgets(self, role: Role, content: str) -> Tuple[str, int]:
        """Enforce per-role token and byte limits."""

        content = self._apply_token_budget(role, content)
        content = self._apply_byte_budget(role, content)
        token_count = self._count_tokens(content)
        self._tokens_used[role] = self._tokens_used.get(role, 0) + token_count
        return content, token_count

    def _apply_token_budget(self, role: Role, content: str) -> str:
        budget = self.config.budget_for(role)
        max_tokens = budget.max_tokens
        if max_tokens is None:
            return content
        used = self._tokens_used.get(role, 0)
        remaining = max(max_tokens - used, 0)
        if remaining == 0:
            return ""
        return self._truncate_to_tokens(content, remaining)

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

    def _truncate_to_tokens(self, content: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not content:
            return ""
        matches = list(self._TOKEN_RE.finditer(content))
        if len(matches) <= max_tokens:
            return content
        end = matches[max_tokens - 1].end()
        return content[:end]

    def _count_tokens(self, content: str) -> int:
        if not content:
            return 0
        return len(self._TOKEN_RE.findall(content))

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

    _TOKEN_RE = re.compile(r"\S+")
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

    def _validate_role_for_phase(self, role: Role, phase: Phase) -> None:
        """NEW: Validate that a role is authorized to speak in the given phase.
        
        Enforces role-phase policy (e.g., only judge speaks in verdict phase).
        Raises ValueError if role is not authorized, preventing unauthorized speech.
        
        Raises:
            ValueError: If role is not in the set of expected roles for this phase.
        """
        
        expected_roles = set(self._roles_for_phase(phase))
        if role not in expected_roles:
            self._record_violation(PolicyViolation.ROLE_PHASE_MISMATCH)
            raise ValueError(
                f"Policy violation: {role.value} is not authorized to speak in {phase.value} phase. "
                f"Expected roles: {', '.join(r.value for r in expected_roles)}"
            )

    def _record_violation(self, violation: PolicyViolation) -> None:
        """NEW: Record a policy violation for audit and hook processing.
        
        Increments the counter for the given violation type. These counts can be
        retrieved after trial completion via get_policy_violations() for testing
        and auditing purposes.
        """
        
        key = violation.value
        self._policy_violations[key] = self._policy_violations.get(key, 0) + 1

    def _detect_cue_in_text(self, text: str, role: Role) -> bool:
        """NEW: Detect if cue values are present in text (for judge blinding verification).
        
        Used to verify that judge prompts don't contain any demographic cues after
        redaction. Performs case-insensitive search for cue values.
        
        Returns:
            True if any cue value is detected in the text, False otherwise.
        """
        
        # Only check for judge role under blinding
        if role is not Role.JUDGE or not self.config.judge_blinding:
            return False
        
        # Check if any cue values appear in the text (case-insensitive)
        cv = self.config.cue_value or ""

        lowercaseText = text.lower()
        if cv and cv.lower() in lowercaseText:
            return True
        
        # Under strict blinding, also check control/treatment values
        if self.config.strict_blinding:
            if self.config.cue.control_value and self.config.cue.control_value.lower() in lowercaseText:
                return True
            if self.config.cue.treatment_value and self.config.cue.treatment_value.lower() in lowercaseText:
                return True
        
        return False

    def get_policy_violations(self) -> Dict[str, int]:
        """NEW: Return recorded policy violations for testing and audit.
        
        Returns a copy of the violation counter dictionary. Can be used to verify
        that policies were enforced correctly during trial execution.
        
        Returns:
            Dict mapping violation type strings to count of occurrences.
            Example: {"interruption_not_allowed": 3, "judge_cue_exposure": 1}
        """
        
        return self._policy_violations.copy()
