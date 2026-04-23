"""
CloudSRE v2 — Deterministic Graders.

Pure Python grading functions — NO network calls, NO LLM dependency.
These work in sandboxed environments where network is blocked.

Kube SRE Gym relies on LLM for per-step scoring (network required).
When LLM fails, they return 0.0 (no signal). Our graders ALWAYS work.

5 grader tiers:
  1. warmup         — Did you find the right service and fix it?
  2. single_fault   — Did you ignore the red herrings?
  3. cascade        — Did you handle the cascade correctly?
  4. multi_cascade  — Did you fix in priority order?
  5. adversarial    — LLM judge (with deterministic fallback)
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Shared Helpers ────────────────────────────────────────────────────────

def _extract_services_mentioned(commands: List[str]) -> set:
    """Extract which services an agent investigated."""
    services = set()
    service_patterns = {
        "payment": [":8001", "payment", "pay"],
        "auth": [":8002", "auth", "jwt", "token"],
        "worker": [":8003", "worker", "queue"],
        "frontend": [":8004", "frontend", "proxy"],
        "cache": [":8005", "cache"],
        "notification": [":8006", "notification", "webhook"],
    }
    for cmd in commands:
        cmd_lower = cmd.lower()
        for svc, patterns in service_patterns.items():
            if any(p in cmd_lower for p in patterns):
                services.add(svc)
    return services


def _extract_phases(history: List[dict]) -> List[str]:
    """Extract the workflow phases from command history."""
    phases = []
    for h in history:
        phase = h.get("phase", h.get("cmd_type", "unknown"))
        if not phases or phases[-1] != phase:
            phases.append(phase)
    return phases


def _has_diagnostic_before_fix(history: List[dict]) -> bool:
    """Check if agent investigated before attempting a fix."""
    fix_idx = None
    diag_idx = None
    for i, h in enumerate(history):
        cmd_type = h.get("cmd_type", h.get("phase", ""))
        if cmd_type in ("health_check", "logs", "metrics", "database"):
            if diag_idx is None:
                diag_idx = i
        elif cmd_type == "fix":
            if fix_idx is None:
                fix_idx = i

    # Good: diagnostic happened before fix
    if diag_idx is not None and fix_idx is not None:
        return diag_idx < fix_idx
    return diag_idx is not None  # At least some diagnostics


def _check_correct_service_targeted(
    commands: List[str], target_service: str
) -> bool:
    """Check if the agent's fix targeted the correct service."""
    service_indicators = {
        "payment": [":8001", "payment", "restart_service payment"],
        "auth": [":8002", "auth", "restart_service auth"],
        "worker": [":8003", "worker", "restart_service worker", "queue drain"],
        "frontend": [":8004", "frontend", "restart_service frontend"],
        "cache": [":8005", "cache", "restart_service cache"],
        "notification": [":8006", "notification", "restart_service notification", "webhook"],
    }
    indicators = service_indicators.get(target_service, [])

    # Look at fix commands specifically
    fix_commands = [c for c in commands if any(kw in c.lower() for kw in
                    ["restart", "kill", "fix:", "drain", "config"])]

    return any(
        any(ind in cmd.lower() for ind in indicators)
        for cmd in fix_commands
    )


def _count_repeats(commands: List[str]) -> int:
    """Count how many commands were repeated."""
    seen = set()
    repeats = 0
    for cmd in commands:
        if cmd in seen:
            repeats += 1
        seen.add(cmd)
    return repeats


# ── Tier 1: Warmup Grader ────────────────────────────────────────────────

def grade_warmup(
    history: List[dict],
    scenario: dict,
    service_health: Dict[str, dict],
    resolved: bool,
) -> Tuple[float, str, dict]:
    """Grade a warmup episode (single clear failure).

    Scoring (max 1.0):
      +0.40  Resolved the incident
      +0.20  Targeted the correct service
      +0.15  Investigated before fixing (diagnostic first)
      +0.15  Efficiency (fewer steps = better)
      +0.10  No repeated commands

    Returns:
        (score, feedback, details_dict)
    """
    commands = [h.get("command", "") for h in history]
    target = scenario.get("target_service", "payment")
    max_steps = scenario.get("max_steps", 10)
    steps = len(history)

    score = 0.0
    feedback_parts = []
    details = {"tier": "warmup", "steps": steps, "resolved": resolved}

    # Resolution (40%)
    if resolved:
        score += 0.40
        feedback_parts.append("✅ Incident resolved.")
    else:
        feedback_parts.append("❌ Incident not resolved.")

    # Correct service (20%)
    if _check_correct_service_targeted(commands, target):
        score += 0.20
        feedback_parts.append(f"✅ Correctly targeted {target} service.")
        details["correct_service"] = True
    else:
        feedback_parts.append(f"⚠️ Did not target the root cause service ({target}).")
        details["correct_service"] = False

    # Diagnostic before fix (15%)
    if _has_diagnostic_before_fix(history):
        score += 0.15
        feedback_parts.append("✅ Good — investigated before fixing.")
        details["diagnostic_first"] = True
    else:
        feedback_parts.append("⚠️ Jumped to fix without investigation.")
        details["diagnostic_first"] = False

    # Efficiency (15%)
    if steps > 0:
        efficiency = max(0, 1.0 - (steps / max_steps))
        eff_score = 0.15 * efficiency
        score += eff_score
        details["efficiency"] = round(efficiency, 2)
    else:
        details["efficiency"] = 0.0

    # No repeats (10%)
    repeats = _count_repeats(commands)
    if repeats == 0:
        score += 0.10
        details["repeats"] = 0
    else:
        score += max(0, 0.10 - 0.03 * repeats)
        details["repeats"] = repeats
        feedback_parts.append(f"⚠️ {repeats} repeated command(s).")

    score = round(min(1.0, score), 2)
    details["score"] = score
    return score, " ".join(feedback_parts), details


# ── Tier 2: Single Fault Grader ──────────────────────────────────────────

def grade_single_fault(
    history: List[dict],
    scenario: dict,
    service_health: Dict[str, dict],
    resolved: bool,
) -> Tuple[float, str, dict]:
    """Grade a single-fault episode with red herrings.

    Scoring (max 1.0):
      +0.30  Resolved the incident
      +0.20  Targeted the correct service (not the red herring)
      +0.15  Investigated the red herring but didn't try to fix it
      +0.15  Diagnostic-first workflow
      +0.10  Efficiency
      +0.10  Phase progression (triage → investigation → fix)
    """
    commands = [h.get("command", "") for h in history]
    target = scenario.get("target_service", "payment")
    misleading = scenario.get("misleading_signals", {})
    max_steps = scenario.get("max_steps", 15)
    steps = len(history)

    score = 0.0
    feedback_parts = []
    details = {"tier": "single_fault", "steps": steps, "resolved": resolved}

    # Resolution (30%)
    if resolved:
        score += 0.30
        feedback_parts.append("✅ Incident resolved.")
    else:
        feedback_parts.append("❌ Incident not resolved.")

    # Correct service targeted (20%)
    if _check_correct_service_targeted(commands, target):
        score += 0.20
        feedback_parts.append(f"✅ Correctly identified {target} as root cause.")
        details["correct_service"] = True
    else:
        feedback_parts.append(f"⚠️ Root cause service ({target}) not targeted.")
        details["correct_service"] = False

    # Red herring handling (15%)
    if misleading:
        herring_services = set(misleading.keys())
        investigated_services = _extract_services_mentioned(commands)
        fix_commands = [c for c in commands if any(kw in c.lower() for kw in
                        ["restart", "fix:", "kill", "config"])]
        fix_services = _extract_services_mentioned(fix_commands)

        # Good: investigated the herring service (shows thoroughness)
        investigated_herring = bool(herring_services & investigated_services)
        # Bad: tried to FIX the herring service
        fixed_herring = bool(herring_services & fix_services)

        if investigated_herring and not fixed_herring:
            score += 0.15
            feedback_parts.append("✅ Investigated misleading signal but didn't fix it — excellent triage.")
            details["herring_handled"] = "investigated_only"
        elif not investigated_herring:
            score += 0.05  # Partial — didn't waste time but also didn't verify
            feedback_parts.append("⚠️ Didn't investigate misleading signals (could miss real issues).")
            details["herring_handled"] = "ignored"
        elif fixed_herring:
            feedback_parts.append("❌ Tried to fix a red herring service — wasted steps.")
            details["herring_handled"] = "fell_for_it"
    else:
        score += 0.10  # No herrings in this scenario
        details["herring_handled"] = "none_present"

    # Diagnostic first (15%)
    if _has_diagnostic_before_fix(history):
        score += 0.15
        details["diagnostic_first"] = True
    else:
        details["diagnostic_first"] = False

    # Efficiency (10%)
    if steps > 0:
        efficiency = max(0, 1.0 - (steps / max_steps))
        score += 0.10 * efficiency
        details["efficiency"] = round(efficiency, 2)

    # Phase progression (10%)
    phases = _extract_phases(history)
    expected_order = ["triage", "investigation", "fix"]
    correct_phase_count = 0
    for expected in expected_order:
        if expected in phases:
            correct_phase_count += 1
    phase_score = 0.10 * (correct_phase_count / len(expected_order))
    score += phase_score
    details["phases_correct"] = correct_phase_count

    score = round(min(1.0, score), 2)
    details["score"] = score
    return score, " ".join(feedback_parts), details


# ── Tier 3: Cascade Grader ───────────────────────────────────────────────

def grade_cascade(
    history: List[dict],
    scenario: dict,
    service_health: Dict[str, dict],
    resolved: bool,
    cascade_triggered: bool = False,
) -> Tuple[float, str, dict]:
    """Grade a cascade episode.

    Scoring (max 1.0):
      +0.25  Resolved the incident (primary)
      +0.20  Handled cascade correctly (if triggered)
      +0.15  Used controlled drain (not drain_all)
      +0.15  Investigated cascade after it triggered
      +0.15  Diagnostic-first and correct service
      +0.10  Efficiency
    """
    commands = [h.get("command", "") for h in history]
    target = scenario.get("target_service", "payment")
    cascade_rules = scenario.get("cascade_rules", [])
    max_steps = scenario.get("max_steps", 20)
    steps = len(history)

    score = 0.0
    feedback_parts = []
    details = {"tier": "cascade", "steps": steps, "resolved": resolved,
               "cascade_triggered": cascade_triggered}

    # Primary resolution (25%)
    if resolved:
        score += 0.25
        feedback_parts.append("✅ Primary incident resolved.")
    else:
        feedback_parts.append("❌ Primary incident not resolved.")

    # Cascade handling (20%)
    if cascade_triggered:
        # Check if the cascade service is now healthy
        cascade_service = (cascade_rules[0].get("affected_service", "worker")
                          if cascade_rules else "worker")
        cascade_healthy = (
            cascade_service in service_health and
            service_health[cascade_service].get("status") == "healthy"
        )

        if cascade_healthy:
            score += 0.20
            feedback_parts.append("✅ Cascade failure also resolved — excellent!")
            details["cascade_resolved"] = True
        else:
            score += 0.05
            feedback_parts.append(f"⚠️ Cascade triggered but {cascade_service} still unhealthy.")
            details["cascade_resolved"] = False
    else:
        # Cascade never triggered — agent may have prevented it!
        used_controlled_drain = any("drain" in c.lower() and "all" not in c.lower()
                                    for c in commands)
        if used_controlled_drain:
            score += 0.20
            feedback_parts.append("✅ Prevented cascade by using controlled drain — PERFECT!")
            details["cascade_prevented"] = True
        else:
            score += 0.10
            details["cascade_prevented"] = False

    # Controlled drain vs drain_all (15%)
    used_drain_all = any("drain" in c.lower() and "all" in c.lower() for c in commands)
    used_controlled = any(
        "drain" in c.lower() and "all" not in c.lower() and "rate" in c.lower()
        for c in commands
    ) or any("drain" in c.lower() and re.search(r'\d+', c) for c in commands)

    if used_controlled and not used_drain_all:
        score += 0.15
        feedback_parts.append("✅ Used controlled drain rate — avoided thundering herd.")
        details["drain_strategy"] = "controlled"
    elif used_drain_all:
        feedback_parts.append("❌ Used drain_all — caused thundering herd!")
        details["drain_strategy"] = "dangerous_drain_all"
    else:
        score += 0.05  # Didn't drain at all
        details["drain_strategy"] = "none"

    # Post-cascade investigation (15%)
    if cascade_triggered:
        # Check if agent investigated after cascade
        cascade_step = next(
            (i for i, h in enumerate(history) if "CASCADE" in str(h.get("feedback", ""))),
            len(history)
        )
        post_cascade_cmds = commands[cascade_step:]
        post_cascade_diag = any(
            kw in " ".join(post_cascade_cmds).lower()
            for kw in ["healthz", "metrics", "logs", "status", "queue"]
        )
        if post_cascade_diag:
            score += 0.15
            feedback_parts.append("✅ Investigated after cascade — systematic response.")
        else:
            feedback_parts.append("⚠️ Did not investigate after cascade triggered.")

    # Correct service + efficiency (10%)
    if _check_correct_service_targeted(commands, target):
        score += 0.05
    if steps > 0:
        efficiency = max(0, 1.0 - (steps / max_steps))
        score += 0.05 * efficiency

    score = round(min(1.0, score), 2)
    details["score"] = score
    return score, " ".join(feedback_parts), details


# ── Tier 4: Multi-Cascade Grader ─────────────────────────────────────────

def grade_multi_cascade(
    history: List[dict],
    scenario: dict,
    service_health: Dict[str, dict],
    resolved: bool,
    cascade_triggered: bool = False,
) -> Tuple[float, str, dict]:
    """Grade a multi-cascade episode.

    Scoring (max 1.0):
      +0.25  All services healthy
      +0.20  Fixed in correct priority order
      +0.15  Handled all cascades
      +0.15  Correct diagnostic workflow
      +0.10  Controlled drain usage
      +0.10  No command escalation / no wasted deploys
      +0.05  Efficiency
    """
    commands = [h.get("command", "") for h in history]
    max_steps = scenario.get("max_steps", 25)
    steps = len(history)

    score = 0.0
    feedback_parts = []
    details = {"tier": "multi_cascade", "steps": steps, "resolved": resolved}

    # All services healthy (25%)
    all_healthy = all(
        svc.get("status") == "healthy"
        for svc in service_health.values()
    )
    if all_healthy:
        score += 0.25
        feedback_parts.append("✅ All services restored to healthy.")
    elif resolved:
        score += 0.15
        feedback_parts.append("⚠️ Partially resolved — some services still degraded.")
    else:
        feedback_parts.append("❌ Services still failing.")

    # Priority order (20%) — payment (DB) should be fixed before auth
    fix_order = []
    for i, cmd in enumerate(commands):
        cmd_lower = cmd.lower()
        if any(kw in cmd_lower for kw in ["restart", "fix:", "kill"]):
            if "payment" in cmd_lower or ":8001" in cmd_lower:
                fix_order.append("payment")
            elif "auth" in cmd_lower or ":8002" in cmd_lower:
                fix_order.append("auth")
            elif "worker" in cmd_lower or ":8003" in cmd_lower:
                fix_order.append("worker")

    # Expected: payment first, then worker/drain, then auth
    if fix_order:
        if fix_order[0] == "payment":
            score += 0.20
            feedback_parts.append("✅ Correct priority — fixed payment (DB) first.")
            details["priority_order"] = "correct"
        else:
            score += 0.05
            feedback_parts.append(f"⚠️ Wrong priority — fixed {fix_order[0]} first (should be payment).")
            details["priority_order"] = "wrong"
    else:
        details["priority_order"] = "no_fixes"

    # Cascade handling (15%)
    if cascade_triggered:
        cascade_services_healthy = sum(
            1 for name, svc in service_health.items()
            if svc.get("status") == "healthy" and name != "frontend"
        )
        cascade_score = 0.15 * (cascade_services_healthy / max(len(service_health) - 1, 1))
        score += cascade_score
        details["cascade_services_healthy"] = cascade_services_healthy

    # Diagnostic workflow (15%)
    services_investigated = _extract_services_mentioned(
        [c for c in commands if any(kw in c.lower() for kw in
         ["healthz", "logs", "metrics", "status"])]
    )
    investigation_coverage = len(services_investigated) / 4.0  # 4 services
    diag_score = 0.15 * investigation_coverage
    score += diag_score
    details["services_investigated"] = list(services_investigated)

    # Controlled drain (10%)
    used_controlled = any("drain" in c.lower() and "all" not in c.lower()
                         for c in commands)
    if used_controlled:
        score += 0.10
        details["drain_controlled"] = True

    # Efficiency (5%)
    if steps > 0:
        efficiency = max(0, 1.0 - (steps / max_steps))
        score += 0.05 * efficiency
        details["efficiency"] = round(efficiency, 2)

    score = round(min(1.0, score), 2)
    details["score"] = score
    return score, " ".join(feedback_parts), details


# ── Tier 5: Adversarial Grader (deterministic fallback) ──────────────────

def grade_adversarial(
    history: List[dict],
    scenario: dict,
    service_health: Dict[str, dict],
    resolved: bool,
    cascade_triggered: bool = False,
    llm_score: Optional[float] = None,
    llm_feedback: Optional[str] = None,
) -> Tuple[float, str, dict]:
    """Grade an adversarial episode.

    Uses LLM score if available, otherwise falls back to deterministic.
    The deterministic fallback uses grade_multi_cascade as a base.

    Scoring:
      If LLM available: 0.6 * LLM_score + 0.4 * deterministic_score
      If no LLM:        1.0 * deterministic_score
    """
    # Get deterministic score using multi_cascade grader
    det_score, det_feedback, det_details = grade_multi_cascade(
        history, scenario, service_health, resolved, cascade_triggered
    )
    det_details["tier"] = "adversarial"

    if llm_score is not None:
        # Blend LLM and deterministic
        blended = 0.6 * llm_score + 0.4 * det_score
        blended = round(min(1.0, max(0.0, blended)), 2)
        feedback = f"LLM: {llm_feedback or 'N/A'} | Deterministic: {det_feedback}"
        det_details["llm_score"] = llm_score
        det_details["deterministic_score"] = det_score
        det_details["blended_score"] = blended
        det_details["score"] = blended
        return blended, feedback, det_details
    else:
        # Pure deterministic fallback (no network needed)
        det_details["llm_available"] = False
        return det_score, det_feedback, det_details


# ── Grader Registry ──────────────────────────────────────────────────────

GRADERS = {
    "warmup": grade_warmup,
    "single_fault": grade_single_fault,
    "cascade": grade_cascade,
    "multi_cascade": grade_multi_cascade,
    "adversarial": grade_adversarial,
}


def grade_episode(
    task_id: str,
    history: List[dict],
    scenario: dict,
    service_health: Dict[str, dict],
    resolved: bool,
    cascade_triggered: bool = False,
    llm_score: Optional[float] = None,
    llm_feedback: Optional[str] = None,
) -> Tuple[float, str, dict]:
    """Grade an episode by task ID. Returns (score, feedback, details)."""
    grader = GRADERS.get(task_id, grade_warmup)

    if task_id == "adversarial":
        return grader(history, scenario, service_health, resolved,
                     cascade_triggered, llm_score, llm_feedback)
    elif task_id in ("cascade", "multi_cascade"):
        return grader(history, scenario, service_health, resolved, cascade_triggered)
    else:
        return grader(history, scenario, service_health, resolved)
