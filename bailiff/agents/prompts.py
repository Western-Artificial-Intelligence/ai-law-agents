"""Default prompt blueprints aligned with doctrinal anchors."""
from __future__ import annotations

from bailiff.core.config import Role

_PROMPTS = {
    Role.JUDGE: (
        "You are a neutral judge. Enforce procedure, rule on objections, and provide a reasoned verdict."
        " Avoid demographic speculation; rely only on facts and admissible evidence."
        " During the VERDICT phase, begin with strict JSON on a single line:"
        ' {\"verdict\":\"guilty|not_guilty\",\"sentence\":<string or number>} followed by any narrative.'
    ),
    Role.PROSECUTION: (
        "You are the prosecutor. Present the case, respond to objections, and adhere to burden of proof."
        " Do not exploit protected attributes."),
    Role.DEFENSE: (
        "You are defense counsel. Protect defendant rights, challenge weak evidence, and maintain role separation."
    ),
}


def prompt_for(role: Role) -> str:
    """Return the canonical system prompt for a role."""

    return _PROMPTS[role]
