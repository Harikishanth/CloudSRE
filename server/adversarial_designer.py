"""
CloudSRE v2 — Adversarial Scenario Designer.

Uses HuggingFace Inference API to generate novel fault scenarios
that target the agent's tracked weaknesses.

This is the self-improvement loop:
    1. Curriculum controller tracks per-fault-type mastery
    2. Designer generates scenarios targeting weak areas
    3. Agent trains on harder scenarios
    4. Mastery improves → designer targets new weaknesses

State-of-the-art approach: LLM-driven curriculum + weakness tracking,
matching Kube-SRE-Gym's Claude-based adversarial designer but using
open-source models via HF API.
"""

import os
import json
import random
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# All available fault types for scenario generation
FAULT_TYPES = [
    "db_lock", "db_pool", "queue_overflow", "queue_pause", "process_crash",
    "cache_invalidation", "webhook_storm", "latency",
    "index_corruption", "index_lag", "rate_limit_zero", "circuit_breaker_stuck",
    "scheduler_stuck", "duplicate_execution", "disk_full", "data_corruption",
    "scrape_failure", "retention_full", "smtp_down", "email_queue_overflow",
    "billing_desync", "invoice_stuck", "config_poisoned", "config_locked",
    "dns_resolution_failure", "stale_entries", "all_backends_removed", "session_corruption",
]

# Service to region mapping for realistic alert generation
SERVICE_REGIONS = {
    "payment": "us-east-1", "auth": "us-east-1", "billing": "us-east-1",
    "gateway": "us-east-1", "loadbalancer": "us-east-1", "config": "us-east-1",
    "worker": "eu-west-1", "scheduler": "eu-west-1", "search": "eu-west-1",
    "storage": "eu-west-1", "metrics_collector": "eu-west-1",
    "frontend": "ap-south-1", "cache": "ap-south-1", "notification": "ap-south-1",
    "email": "ap-south-1", "dns": "ap-south-1",
}

# Fault type → which services it can target
FAULT_SERVICE_MAP = {
    "db_lock": ["payment", "billing", "auth"],
    "db_pool": ["payment", "billing"],
    "queue_overflow": ["worker"],
    "queue_pause": ["worker"],
    "process_crash": list(SERVICE_REGIONS.keys()),
    "cache_invalidation": ["cache"],
    "webhook_storm": ["notification"],
    "latency": list(SERVICE_REGIONS.keys()),
    "index_corruption": ["search"],
    "index_lag": ["search"],
    "rate_limit_zero": ["gateway"],
    "circuit_breaker_stuck": ["gateway"],
    "scheduler_stuck": ["scheduler"],
    "duplicate_execution": ["scheduler"],
    "disk_full": ["storage"],
    "data_corruption": ["storage"],
    "scrape_failure": ["metrics_collector"],
    "retention_full": ["metrics_collector"],
    "smtp_down": ["email"],
    "email_queue_overflow": ["email"],
    "billing_desync": ["billing"],
    "invoice_stuck": ["billing"],
    "config_poisoned": ["config"],
    "config_locked": ["config"],
    "dns_resolution_failure": ["dns"],
    "stale_entries": ["dns"],
    "all_backends_removed": ["loadbalancer"],
    "session_corruption": ["loadbalancer"],
}

HF_API_URL = "https://api-inference.huggingface.co/models/{model}"
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class PerformanceTracker:
    """Tracks per-fault-type mastery for curriculum-aware scenario generation.

    This is the weakness tracking component of the self-improvement loop.
    """

    def __init__(self):
        # {fault_type: [scores]} — rolling window of last 10 scores
        self._scores: Dict[str, List[float]] = {ft: [] for ft in FAULT_TYPES}
        self._attempts: Dict[str, int] = {ft: 0 for ft in FAULT_TYPES}

    def record(self, fault_type: str, score: float):
        """Record a score for a fault type."""
        if fault_type in self._scores:
            self._scores[fault_type].append(score)
            if len(self._scores[fault_type]) > 10:
                self._scores[fault_type] = self._scores[fault_type][-10:]
            self._attempts[fault_type] = self._attempts.get(fault_type, 0) + 1

    def get_mastery(self, fault_type: str) -> float:
        """Get mastery level (0.0 = never tried, 1.0 = perfect) for a fault type."""
        scores = self._scores.get(fault_type, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_weaknesses(self, top_n: int = 5) -> List[str]:
        """Get the fault types the agent is worst at."""
        masteries = [(ft, self.get_mastery(ft)) for ft in FAULT_TYPES]
        # Sort by mastery (lowest first), break ties by fewest attempts
        masteries.sort(key=lambda x: (x[1], self._attempts.get(x[0], 0)))
        return [ft for ft, _ in masteries[:top_n]]

    def get_strengths(self, top_n: int = 5) -> List[str]:
        """Get the fault types the agent is best at."""
        masteries = [(ft, self.get_mastery(ft)) for ft in FAULT_TYPES
                     if self._attempts.get(ft, 0) > 0]
        masteries.sort(key=lambda x: -x[1])
        return [ft for ft, _ in masteries[:top_n]]

    def get_weights(self, scenario_ids: List[str]) -> List[float]:
        """Get sampling weights — higher weight = agent is weaker at this."""
        weights = []
        for sid in scenario_ids:
            # Extract fault type from scenario ID (e.g., "warmup_db_lock" → "db_lock")
            parts = sid.split("_", 1)
            fault = parts[1] if len(parts) > 1 else sid
            mastery = self.get_mastery(fault)
            # Inverse mastery = higher weight for weak areas
            weights.append(max(0.1, 1.0 - mastery))
        return weights

    def summary(self) -> Dict[str, Any]:
        """Summary for logging."""
        return {
            "total_episodes": sum(self._attempts.values()),
            "weaknesses": self.get_weaknesses(3),
            "strengths": self.get_strengths(3),
            "coverage": sum(1 for ft in FAULT_TYPES if self._attempts.get(ft, 0) > 0),
            "total_fault_types": len(FAULT_TYPES),
        }


class AdversarialDesigner:
    """Generates novel fault scenarios targeting agent weaknesses.

    Two modes:
    1. LLM-based (with HF_TOKEN): Uses Qwen2.5-72B to generate creative scenarios
    2. Programmatic fallback: Uses weakness tracking + random composition

    Both modes are curriculum-aware — they adapt to the agent's performance.
    """

    def __init__(self, model: str = DEFAULT_MODEL, hf_token: Optional[str] = None):
        self.model = model
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.api_url = HF_API_URL.format(model=model)
        self._use_llm = bool(self.hf_token)
        self.tracker = PerformanceTracker()

        if not self._use_llm:
            logger.warning("Adversarial Designer: No HF_TOKEN. Using programmatic mode.")

    def design_scenario(self, difficulty: int = 3) -> Dict[str, Any]:
        """Design a scenario targeting agent weaknesses.

        Args:
            difficulty: 1-5 scale (1=warmup, 5=expert)

        Returns:
            Scenario dict with: target_service, failure_type, alert_message,
            root_cause, correct_fix_description, red_herrings, cascades
        """
        if self._use_llm:
            try:
                return asyncio.run(self._design_with_llm(difficulty))
            except Exception as e:
                logger.warning(f"LLM design failed: {e}. Using programmatic.")

        return self._design_programmatic(difficulty)

    def _design_programmatic(self, difficulty: int) -> Dict[str, Any]:
        """Programmatic scenario generation targeting weaknesses."""
        weaknesses = self.tracker.get_weaknesses(5)
        strengths = self.tracker.get_strengths(3)

        # Primary fault: target a weakness
        if weaknesses:
            primary_fault = random.choice(weaknesses[:3])
        else:
            primary_fault = random.choice(FAULT_TYPES)

        # Pick target service for the fault
        possible_services = FAULT_SERVICE_MAP.get(primary_fault, ["payment"])
        target_service = random.choice(possible_services)
        region = SERVICE_REGIONS.get(target_service, "us-east-1")

        # Build alert message
        alert = self._generate_alert(target_service, primary_fault, region, difficulty)

        scenario = {
            "target_service": target_service,
            "failure_type": primary_fault,
            "difficulty": difficulty * 0.2,
            "alert_message": alert,
            "root_cause": f"{primary_fault} on {target_service} in {region}",
            "correct_fix_description": f"Restart {target_service} or apply targeted fix for {primary_fault}",
            "generated_by": "adversarial_designer_programmatic",
        }

        # Add red herrings for difficulty >= 3
        if difficulty >= 3 and strengths:
            red_herring_fault = random.choice(strengths)
            rh_services = FAULT_SERVICE_MAP.get(red_herring_fault, ["frontend"])
            rh_service = random.choice([s for s in rh_services if s != target_service] or rh_services)
            scenario["red_herrings"] = [{
                "service": rh_service,
                "fault_type": "misleading_signal",
                "message": f"Elevated latency on {rh_service} (red herring)",
            }]

        # Add cascade for difficulty >= 4
        if difficulty >= 4:
            cascade_targets = [s for s in SERVICE_REGIONS.keys() if s != target_service]
            cascade_svc = random.choice(cascade_targets)
            scenario["cascade"] = {
                "trigger_on_fix": True,
                "cascade_fault": "process_crash",
                "cascade_target": cascade_svc,
                "cascade_delay_seconds": 2,
            }

        return scenario

    def _generate_alert(self, service: str, fault: str, region: str, difficulty: int) -> str:
        """Generate a realistic PagerDuty-style alert."""
        severity = {1: "P3", 2: "P2", 3: "P2", 4: "P1", 5: "P0"}[difficulty]
        time_str = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}"

        fault_descriptions = {
            "db_lock": f"RDS connection pool exhausted on {service}.{region}.internal — all writes blocked",
            "process_crash": f"ECS task terminated (OOMKilled) — {service}.{region}.internal not responding",
            "queue_overflow": f"SQS dead letter queue depth critical — worker.eu-west-1.internal backpressure",
            "cache_invalidation": f"ElastiCache hit ratio dropped to 0% — thundering herd on {service}.{region}.internal",
            "index_corruption": f"Search index checksum mismatch on {service}.{region}.internal — queries returning empty",
            "rate_limit_zero": f"Gateway rate limit misconfigured to 0 RPS on {service}.{region}.internal",
            "circuit_breaker_stuck": f"Circuit breaker stuck OPEN on {service}.{region}.internal — all upstream rejected",
            "disk_full": f"EBS volume 100% full on {service}.{region}.internal — writes rejected",
            "smtp_down": f"SMTP upstream unreachable from {service}.{region}.internal — email queue backing up",
            "dns_resolution_failure": f"Route53 resolution failure — NXDOMAIN for {service}.{region}.internal",
            "all_backends_removed": f"ALB has no healthy targets — {service}.{region}.internal returning 503",
            "config_poisoned": f"SSM Parameter Store poisoned values detected on {service}.{region}.internal",
        }

        desc = fault_descriptions.get(fault,
            f"{fault.replace('_', ' ').title()} detected on {service}.{region}.internal")

        alert = (
            f"\U0001f6a8 INCIDENT — {time_str} UTC | Severity: {severity}\n"
            f"Region: {region} | Service: {service}\n"
            f"{desc}"
        )

        if difficulty >= 3:
            alert += f"\nMultiple services may be affected. Check cross-region dependencies."

        return alert

    async def _design_with_llm(self, difficulty: int) -> Dict[str, Any]:
        """Use LLM to generate a creative scenario."""
        import httpx

        weaknesses = self.tracker.get_weaknesses(5)
        strengths = self.tracker.get_strengths(3)

        prompt = f"""You are an adversarial SRE scenario designer. Create a cloud infrastructure incident.

AGENT WEAKNESSES (target these): {weaknesses or 'unknown — first episode'}
AGENT STRENGTHS (use as red herrings): {strengths or 'none yet'}
DIFFICULTY: {difficulty}/5

Available fault types: {FAULT_TYPES}
Available services: {list(SERVICE_REGIONS.keys())}
Service regions: {json.dumps(SERVICE_REGIONS)}

Design a scenario. Output ONLY valid JSON:
{{
  "target_service": "<service>",
  "failure_type": "<fault_type>",
  "alert_message": "<PagerDuty-style alert>",
  "root_cause": "<technical root cause>",
  "correct_fix_description": "<what the agent should do>",
  "red_herrings": [
    {{"service": "<svc>", "fault_type": "misleading_signal", "message": "<misleading log>"}}
  ]
}}"""

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "return_full_text": False,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                text = result[0].get("generated_text", "") if isinstance(result, list) else str(result)

                # Parse JSON from response
                if "{" in text:
                    start = text.index("{")
                    end = text.rindex("}") + 1
                    scenario = json.loads(text[start:end])
                    scenario["generated_by"] = "adversarial_designer_llm"
                    scenario["difficulty"] = difficulty * 0.2

                    # Validate required fields
                    if "target_service" not in scenario or "failure_type" not in scenario:
                        raise ValueError("Missing required fields")

                    return scenario

            raise Exception(f"LLM design failed: {response.status_code}")
