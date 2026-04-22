# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the CloudSRE v2 Environment.

These define the strict API contract between the agent and the environment.
The agent sends a CloudSREAction (a real SRE command to execute).
The environment returns a CloudSREObservation (the result of running that command
on real microservices, plus service health, metrics, and reward feedback).

Unlike Round 1's IncidentAction (which had predefined tool/target enums),
CloudSRE uses free-form commands that execute on REAL services — curl, sqlite3,
cat, kill, etc. This is harder for the agent but more realistic.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CloudSREAction(Action):
    """What the agent sends — a real SRE command to execute on the service mesh.

    Instead of choosing from a dropdown (query_logs, check_metrics, apply_fix),
    the agent writes actual commands like a real SRE would. These commands
    execute on real running services — real HTTP calls, real SQL queries,
    real process management.

    Examples:
        - "curl http://localhost:8001/healthz"
        - "curl http://localhost:8001/metrics"
        - "cat /var/log/payment/error.log | tail -20"
        - "sqlite3 /data/app.db 'SELECT count(*) FROM payments WHERE status=\"pending\"'"
        - "kill -9 $(pgrep -f payment_service)"
        - "python /app/services/payment_service.py &"
        - "curl -X POST http://localhost:8003/queue/drain?rate=10"
    """

    command: str = Field(
        ...,
        description=(
            "A real SRE command to execute. Can be: "
            "curl (HTTP health/metrics), cat (log reading), "
            "sqlite3 (database inspection), kill/restart (process management), "
            "or POST to service control endpoints (queue drain, config change)."
        ),
    )


class CloudSREObservation(Observation):
    """What the agent sees — real system state from real running services.

    Every field in this observation comes from REAL infrastructure:
    - alert: the PagerDuty-style incident description
    - command_output: actual stdout from the command (real HTTP response, real log, real SQL result)
    - service_health: real /healthz responses from each service
    - phase: detected SRE workflow phase (triage/investigation/fix/verify)
    - feedback: phase-aware hint from the grading system
    """

    # ── Incident Context ──
    alert: str = Field(
        default="",
        description="PagerDuty-style incident alert describing the symptoms.",
    )
    scenario_id: str = Field(
        default="",
        description="Current scenario identifier for grading.",
    )
    task_id: str = Field(
        default="",
        description="Current task tier (warmup/single_fault/cascade/multi_cascade/adversarial).",
    )

    # ── Command Result ──
    command_output: str = Field(
        default="",
        description="Real stdout/stderr from the last executed command (truncated to 2000 chars).",
    )

    # ── Service State (from real /healthz endpoints) ──
    service_health: Dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Real health status of each service. "
            "Example: {'payment': {'status': 'unhealthy', 'latency_ms': 30200, 'error_rate': 0.94}}"
        ),
    )

    # ── Episode Progress ──
    step_number: int = Field(
        default=0,
        description="Current step in the episode (0-indexed).",
    )
    max_steps: int = Field(
        default=15,
        description="Maximum steps allowed for this task tier.",
    )

    # ── SRE Workflow Tracking ──
    phase: str = Field(
        default="triage",
        description="Detected SRE workflow phase: triage, investigation, mitigation, fix, or verification.",
    )
    history: List[str] = Field(
        default_factory=list,
        description="Log of previously executed commands (prevents repeat actions).",
    )

    # ── Feedback ──
    feedback: str = Field(
        default="",
        description="Phase-aware feedback from the grading system. Hints at what a good SRE would do next.",
    )

    # ── Cascade State ──
    cascade_triggered: bool = Field(
        default=False,
        description="Whether a cascading failure has been triggered by the agent's fix.",
    )
    cascade_alert: str = Field(
        default="",
        description="Alert text for the cascade failure (empty if no cascade).",
    )

    # ── Reward Signal ──
    reward: float = Field(
        default=0.0,
        description="Dense per-step reward. Range: -2.0 to +5.0 across the episode.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode is complete (resolved, failed, or timed out).",
    )


class CloudSREState(State):
    """Episode metadata — internal state not shown to the agent.

    This tracks everything the environment needs to grade, manage curriculum,
    and handle cascades. The agent never sees this directly — only the
    Observation is visible to the agent.

    Kube SRE Gym has this too (KubeSreGymState) — it's required by OpenEnv
    for the /state endpoint.
    """

    # ── Scenario Identity ──
    scenario_id: str = ""
    task_id: str = ""
    difficulty: float = 0.2

    # ── Root Cause Ground Truth ──
    root_cause_service: str = ""
    root_cause_description: str = ""
    correct_fix: str = ""

    # ── Episode Progress ──
    is_resolved: bool = False
    cumulative_reward: float = 0.0
    steps_taken: int = 0

    # ── Cascade Tracking ──
    cascade_triggered: bool = False
    cascade_resolved: bool = False
    primary_fix_applied: bool = False

    # ── Curriculum ──
    judge_persona: str = "junior"  # junior / senior / principal
    tier: int = 1  # 1-5
    curriculum_stats: dict = {}

    # ── Phase Tracking ──
    current_phase: str = "triage"
    phases_visited: list = []
    investigated_services: list = []
    fixes_attempted: list = []


# ── Scenario Data Models ────────────────────────────────────────────────────


@dataclass
class CascadeRule:
    """Defines what happens AFTER the agent fixes the primary fault.

    This is the key differentiator from Kube SRE Gym — they don't have this.
    Our cascades are REAL: fixing the DB lock actually causes the payment
    queue to flood, which the agent must then handle.
    """

    trigger_condition: str  # e.g., "agent removes DB lock while queue depth > 100"
    cascade_type: str  # e.g., "thundering_herd"
    affected_service: str  # e.g., "payment"
    description: str  # e.g., "Queued requests flood payment simultaneously"
    agent_must: str  # e.g., "Rate-limit queue drain before removing lock"


@dataclass
class ScenarioSpec:
    """A failure scenario template — what goes wrong and how.

    Matches Kube SRE Gym's ScenarioSpec interface for compatibility,
    but adds cascade_rules and misleading_signals that they don't have.
    """

    # ── Core (matches Kube SRE Gym's ScenarioSpec) ──
    failure_type: str  # e.g., "db_lock", "process_crash", "jwt_corruption"
    target_service: str  # e.g., "payment", "auth", "worker", "frontend"
    params: dict = field(default_factory=dict)  # Fault-specific parameters
    root_cause: str = ""  # Ground truth explanation
    difficulty: float = 0.3
    alert_message: str = ""  # PagerDuty-style alert text
    correct_fix_description: str = ""  # What the agent should do
    expected_diagnostic_path: list = field(default_factory=list)  # Ideal command sequence

    # ── CloudSRE-specific (they don't have these) ──
    misleading_signals: dict = field(default_factory=dict)  # {service: "fake error message"}
    cascade_rules: list = field(default_factory=list)  # List[CascadeRule]
    task_id: str = "single_fault"  # Which task tier this belongs to
    scenario_id: str = ""  # Unique identifier


@dataclass
class IncidentStep:
    """One mutation in a multi-step adversarial incident.

    Matches Kube SRE Gym's IncidentStep exactly.
    """

    action: str  # Command to inject the fault
    effect: str  # What this causes
    order: int  # Injection sequence
    is_root_cause: bool = False


@dataclass
class AdversarialScenarioSpec:
    """Multi-step incident designed by an LLM judge.

    Matches Kube SRE Gym's AdversarialScenarioSpec interface but adds
    cascade_chain for our cascading failure mechanic.
    """

    # ── Fields matching Kube SRE Gym's interface ──
    failure_type: str
    target_service: str
    root_cause: str
    difficulty: float
    alert_message: str
    correct_fix_description: str

    # ── Multi-step incident fields (same as theirs) ──
    name: str = ""
    steps: list = field(default_factory=list)  # List[IncidentStep]
    diagnosis_steps: list = field(default_factory=list)
    fix_steps: list = field(default_factory=list)
    verify_steps: list = field(default_factory=list)
    red_herrings: list = field(default_factory=list)
    expected_observation_hints: list = field(default_factory=list)

    # ── CloudSRE-specific ──
    cascade_chain: list = field(default_factory=list)  # List[CascadeRule]
    expected_cascade_handling: list = field(default_factory=list)  # Steps to handle cascades

    # ── ScenarioSpec compat ──
    params: dict = field(default_factory=dict)
    expected_diagnostic_path: list = field(default_factory=list)
    task_id: str = "adversarial"
    scenario_id: str = ""

