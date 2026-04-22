"""
CloudSRE v2 — LLM Judge.

Uses the LLMClient to evaluate agent actions and verify resolution.
Falls back to deterministic scoring when LLM is unavailable.

Kube SRE Gym equivalent: judge.py (270 lines)
Our design: Same structure + cascade-aware evaluation + Gemini support.
"""

import logging
from typing import Optional, Tuple

from cloud_sre_v2.server.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Persona Prompts ──────────────────────────────────────────────────────

PERSONAS = {
    "junior": (
        "You are a Junior SRE mentor evaluating a trainee's incident response. "
        "Be encouraging and give partial credit for partially correct approaches."
    ),
    "senior": (
        "You are a Senior SRE evaluating an engineer's incident response. "
        "Apply standard expectations. Reward systematic diagnosis. "
        "Penalize repeated commands and irrelevant actions."
    ),
    "principal": (
        "You are a Principal SRE with expert standards. "
        "Reward efficiency — correct fast fixes are GOOD. "
        "Penalize WRONG fixes, not fast ones. For cascade scenarios, "
        "reward handling ALL cascading failures, not just the primary."
    ),
}


class LLMJudge:
    """LLM-based judge with deterministic fallback.

    When LLM is available: returns rich AI-generated feedback and scoring.
    When LLM is unavailable: returns None so callers use deterministic graders.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @property
    def is_available(self) -> bool:
        return self.llm.is_available

    def evaluate(
        self,
        command: str,
        output: str,
        scenario: dict,
        history: list,
        persona: str = "junior",
    ) -> Optional[Tuple[float, str]]:
        """Evaluate a single agent action.

        Returns (score, feedback) or None if LLM unavailable.
        """
        if not self.llm.is_available:
            return None

        history_summary = "\n".join(
            f"  Step {h['step']}: {h['command']} → reward {h.get('reward', 0):.2f}"
            for h in history[-5:]
        ) or "  (first step)"

        user_prompt = f"""Evaluate this SRE action during a production incident.

INCIDENT:
- Alert: {scenario.get('alert_message', 'N/A')}
- Root cause: {scenario.get('root_cause', 'N/A')}
- Correct fix: {scenario.get('correct_fix_description', 'N/A')}
- Difficulty: {scenario.get('difficulty', 0.3):.1f}/1.0

AGENT ACTION:
- Command: {command}
- Output (truncated): {output[:500]}

RECENT HISTORY:
{history_summary}
- Total steps taken: {len(history) + 1}

Return JSON only: {{"score": <float -1.0 to 1.0>, "feedback": "<1-2 sentence evaluation>"}}"""

        result = self.llm.chat_json(
            PERSONAS.get(persona, PERSONAS["senior"]),
            user_prompt,
            temperature=0.3,
            max_tokens=256,
        )

        if result is None:
            return None

        score = max(-1.0, min(1.0, float(result.get("score", 0.0))))
        feedback = result.get("feedback", "")
        return score, feedback

    def verify_resolution(
        self,
        scenario: dict,
        history: list,
        service_health: dict,
    ) -> Optional[Tuple[bool, str]]:
        """Verify if the incident was actually resolved.

        Returns (resolved, reason) or None if LLM unavailable.
        """
        if not self.llm.is_available:
            return None

        history_summary = "\n".join(
            f"  Step {h['step']}: {h['command']} → {h.get('output', '')[:100]}"
            for h in history
        )

        health_summary = "\n".join(
            f"  {name}: status={info.get('status')} error_rate={info.get('error_rate', 0):.1%}"
            for name, info in service_health.items()
        )

        user_prompt = f"""Verify whether this production incident was ACTUALLY resolved.

INCIDENT:
- Alert: {scenario.get('alert_message', 'N/A')}
- Root cause: {scenario.get('root_cause', 'N/A')}
- Required fix: {scenario.get('correct_fix_description', 'N/A')}

AGENT'S ACTIONS:
{history_summary}

CURRENT SERVICE HEALTH:
{health_summary}

Did the agent actually fix ALL the issues? Check:
- Is the root cause service now healthy?
- Were ALL cascading failures also addressed?
- Is the fix command output showing success (not errors)?

Return JSON only: {{"resolved": true/false, "reason": "<1-2 sentence explanation>"}}"""

        result = self.llm.chat_json(
            "You are a strict incident verification system. Only confirm resolution if ALL faults were genuinely fixed.",
            user_prompt,
            temperature=0.1,
            max_tokens=256,
        )

        if result is None:
            return None

        resolved = bool(result.get("resolved", False))
        reason = result.get("reason", "")
        logger.info(f"Judge verification: resolved={resolved} | {reason}")
        return resolved, reason
