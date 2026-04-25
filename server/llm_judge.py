"""
CloudSRE v2 — LLM Judge.

Scores SRE episodes using HuggingFace Inference API for real
LLM-based evaluation of agent performance.

Used for tiers 3-5 (cascade, multi_cascade, adversarial) where
deterministic grading alone isn't enough to evaluate complex
multi-step incident response.

Architecture:
    Deterministic grader (fast, free) provides base score
    LLM judge (Qwen2.5-72B) provides nuanced workflow evaluation
    Final score = 0.4 * deterministic + 0.6 * llm_score (tiers 3-5)
    Tiers 1-2 use deterministic only (speed + cost)
"""

import os
import json
import logging
import asyncio
from typing import Tuple, Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# HuggingFace Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/{model}"
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class LLMJudge:
    """Scores SRE episodes using an LLM for workflow quality evaluation.

    The judge evaluates:
    1. SRE Workflow (0.3): triage → investigate → fix → verify
    2. Root Cause Accuracy (0.3): correct service + correct fault identified
    3. Fix Quality (0.2): appropriate and minimal remediation
    4. Efficiency (0.2): steps taken vs optimal path

    Uses HuggingFace Inference API with configurable model.
    Falls back to deterministic scoring if API is unavailable.
    """

    def __init__(self, model: str = DEFAULT_MODEL, hf_token: Optional[str] = None):
        self.model = model
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.api_url = HF_API_URL.format(model=model)
        self._available = bool(self.hf_token)
        if not self._available:
            logger.warning("LLM Judge: No HF_TOKEN found. Will use deterministic scoring only.")

    def _build_judge_prompt(
        self,
        history: List[dict],
        scenario: dict,
        resolved: bool,
        service_health: Dict[str, dict],
    ) -> str:
        """Build the prompt for the LLM judge."""
        # Format command history
        cmd_lines = []
        for entry in history[-12:]:  # Last 12 steps max
            cmd = entry.get("command", "")
            output_preview = entry.get("output", "")[:150]
            phase = entry.get("phase", "unknown")
            cmd_lines.append(f"  [{phase}] {cmd}\n    -> {output_preview}")

        commands_text = "\n".join(cmd_lines) if cmd_lines else "  (no commands executed)"

        # Format health summary
        health_lines = []
        for svc, info in service_health.items():
            status = info.get("status", "unknown")
            error = info.get("error", "")
            if status != "healthy":
                health_lines.append(f"  {svc}: {status} ({error})")
            else:
                health_lines.append(f"  {svc}: healthy")

        health_text = "\n".join(health_lines[:10])  # Top 10

        target = scenario.get("target_service", "unknown")
        fault = scenario.get("failure_type", "unknown")
        alert = scenario.get("alert_message", "No alert")

        return f"""You are an expert SRE reviewing an incident response. Score the agent's performance.

INCIDENT ALERT:
{alert[:500]}

ROOT CAUSE: {fault} on {target} service

AGENT'S ACTIONS (chronological):
{commands_text}

FINAL SERVICE HEALTH:
{health_text}

INCIDENT RESOLVED: {"YES" if resolved else "NO"}

Score the agent on these criteria (0.0 to 1.0 each):

1. WORKFLOW (weight 0.3): Did the agent follow proper SRE workflow?
   - Triage first (status/healthz checks)
   - Investigate (logs, metrics, database queries)
   - Fix (restart, drain, config change)
   - Verify (health check after fix)

2. ROOT_CAUSE (weight 0.3): Did the agent correctly identify the failing service?
   - Did they investigate the RIGHT service?
   - Did they understand the failure type?

3. FIX_QUALITY (weight 0.2): Was the fix appropriate?
   - Correct remediation action?
   - Minimal blast radius (didn't restart everything blindly)?

4. EFFICIENCY (weight 0.2): How efficient was the response?
   - Reasonable number of steps?
   - No excessive repetition?
   - Direct path to resolution?

Respond in EXACTLY this JSON format (no other text):
{{"workflow": 0.X, "root_cause": 0.X, "fix_quality": 0.X, "efficiency": 0.X, "feedback": "one-sentence summary"}}"""

    async def score_episode(
        self,
        history: List[dict],
        scenario: dict,
        resolved: bool,
        service_health: Dict[str, dict],
    ) -> Tuple[float, str, Dict[str, float]]:
        """Score an episode using the LLM judge.

        Returns:
            (weighted_score, feedback_text, component_scores)
        """
        if not self._available:
            return self._fallback_score(resolved)

        prompt = self._build_judge_prompt(history, scenario, resolved, service_health)

        try:
            result = await self._call_hf_api(prompt)
            scores = self._parse_scores(result)

            weighted = (
                scores["workflow"] * 0.3
                + scores["root_cause"] * 0.3
                + scores["fix_quality"] * 0.2
                + scores["efficiency"] * 0.2
            )

            feedback = scores.get("feedback", "No feedback")
            return weighted, feedback, scores

        except Exception as e:
            logger.warning(f"LLM Judge API error: {e}. Falling back to deterministic.")
            return self._fallback_score(resolved)

    async def _call_hf_api(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.1,
                "return_full_text": False,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
            elif response.status_code == 503:
                # Model loading — retry once after wait
                logger.info("LLM Judge: Model loading, waiting 10s...")
                await asyncio.sleep(10)
                response = await client.post(self.api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "")
                raise Exception(f"HF API still loading: {response.status_code}")
            else:
                raise Exception(f"HF API error: {response.status_code} {response.text[:200]}")

    def _parse_scores(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into scores dict."""
        # Try to find JSON in the response
        try:
            # Look for JSON block
            json_match = response_text
            if "{" in response_text:
                start = response_text.index("{")
                end = response_text.rindex("}") + 1
                json_match = response_text[start:end]
            scores = json.loads(json_match)

            # Validate and clamp scores
            for key in ["workflow", "root_cause", "fix_quality", "efficiency"]:
                if key in scores:
                    scores[key] = max(0.0, min(1.0, float(scores[key])))
                else:
                    scores[key] = 0.5  # Default

            return scores

        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Could not parse LLM judge response: {response_text[:100]}")
            return {
                "workflow": 0.5,
                "root_cause": 0.5,
                "fix_quality": 0.5,
                "efficiency": 0.5,
                "feedback": "Parse error — using default scores",
            }

    def _fallback_score(self, resolved: bool) -> Tuple[float, str, Dict[str, float]]:
        """Deterministic fallback when LLM API is unavailable."""
        base = 0.7 if resolved else 0.2
        scores = {
            "workflow": base,
            "root_cause": base,
            "fix_quality": base,
            "efficiency": base,
            "feedback": "LLM judge unavailable — using deterministic fallback",
        }
        return base, scores["feedback"], scores

    def score_episode_sync(
        self,
        history: List[dict],
        scenario: dict,
        resolved: bool,
        service_health: Dict[str, dict],
    ) -> Tuple[float, str, Dict[str, float]]:
        """Synchronous wrapper for score_episode."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.score_episode(history, scenario, resolved, service_health),
                    )
                    return future.result(timeout=45)
            else:
                return loop.run_until_complete(
                    self.score_episode(history, scenario, resolved, service_health)
                )
        except Exception as e:
            logger.warning(f"Sync LLM judge error: {e}")
            return self._fallback_score(resolved)
