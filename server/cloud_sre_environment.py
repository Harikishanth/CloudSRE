"""
CloudSRE v2 — Core Environment Implementation.

This is the OpenEnv Environment class — the heart of everything.
Agent sends CloudSREAction, environment returns CloudSREObservation.

Kube SRE Gym equivalent: kube_sre_gym_environment.py (480 lines)
Ours adds: cascade tracking, phase detection, dense rewards, and 5 task tiers.

Key differences from theirs:
  1. Real services (not K8s pods) — richer observation data
  2. Cascade mechanic — fix one thing, break another
  3. Phase tracking — triage/investigation/mitigation/fix/verification
  4. Dense per-step rewards — not just resolve/fail binary
  5. 5 task tiers (warmup → adversarial) vs their 2 modes
"""

import json
import os
import logging
import time
import threading
from uuid import uuid4
from typing import Optional, Dict

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import CloudSREAction, CloudSREObservation, CloudSREState
except ImportError:
    from models import CloudSREAction, CloudSREObservation, CloudSREState

from .constants import TASK_CONFIGS
from cloud_sre_v2.services.orchestrator import ServiceOrchestrator

logger = logging.getLogger(__name__)


# ── Adaptive Sampling — Theme #4: Self-Improving Environment ─────────────

class PerformanceTracker:
    """Tracks per-scenario agent performance for adaptive sampling.

    The environment remembers which scenarios the agent struggles with and
    presents them MORE OFTEN — a self-improvement loop that doesn't require
    any changes to the training client.

    Weight formula:  w_i = 1.0 / (success_rate_i + 0.1)
      - Agent aces a scenario (90% success) → weight 1.0   (rare)
      - Agent fails a scenario (10% success) → weight 5.0   (5x more likely)
      - New scenario (no data)               → weight ~1.0  (fair chance)

    The +0.1 floor prevents division by zero AND ensures even mastered
    scenarios still appear occasionally (curriculum doesn't forget them).
    """

    TIER_ORDER = ["warmup", "single_fault", "cascade", "multi_cascade", "adversarial"]

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        # {scenario_id: [bool, bool, ...]}  True = resolved
        self._history: Dict[str, list] = {}
        self._total_episodes = 0

    def record(self, scenario_id: str, resolved: bool):
        """Record an episode outcome for a scenario."""
        if scenario_id not in self._history:
            self._history[scenario_id] = []
        self._history[scenario_id].append(resolved)
        # Sliding window — keep last N attempts
        if len(self._history[scenario_id]) > self._window_size:
            self._history[scenario_id] = self._history[scenario_id][-self._window_size:]
        self._total_episodes += 1

    def success_rate(self, scenario_id: str) -> float:
        """Success rate for a specific scenario (0.0 to 1.0)."""
        history = self._history.get(scenario_id, [])
        if not history:
            return 0.5  # Unknown — assume middle
        return sum(history) / len(history)

    def get_weights(self, scenario_ids: list) -> list:
        """Return sampling weights — inverse of success rate.

        Scenarios the agent fails get HIGHER weight → sampled more often.
        """
        weights = []
        for sid in scenario_ids:
            rate = self.success_rate(sid)
            w = 1.0 / (rate + 0.1)
            weights.append(w)
        return weights

    def should_promote(self, current_tier: str, threshold: float = 0.7, min_attempts: int = 8) -> bool:
        """RLVE §2.2 — Auto-curriculum promotion.

        When agent achieves >= threshold success over >= min_attempts on current
        tier, promote to next tier. Prevents learning stall.
        """
        tier_scenarios = [s for s in self._history if s.startswith(current_tier)]
        if not tier_scenarios:
            return False
        total = sum(len(self._history[s]) for s in tier_scenarios)
        if total < min_attempts:
            return False
        correct = sum(sum(self._history[s]) for s in tier_scenarios)
        return (correct / total) >= threshold

    def get_stats(self) -> Dict[str, dict]:
        """Return performance stats for logging/debugging."""
        stats = {}
        for sid, history in self._history.items():
            rate = sum(history) / len(history) if history else 0.0
            stats[sid] = {
                "attempts": len(history),
                "success_rate": round(rate, 2),
                "weight": round(1.0 / (rate + 0.1), 2),
            }
        return stats


class CloudSREEnvironment(Environment):
    """CloudSRE v2 OpenEnv Environment.

    Agent diagnoses and fixes real microservice incidents with cascading failures.

    Config via env vars:
      TASK_TIER      - "warmup", "single_fault", "cascade", "multi_cascade", "adversarial", "auto"
      LLM_BACKEND    - "gemini" (default), "openai", "anthropic"
      MAX_STEPS      - Override max steps per episode

    RLVE Alignment (arXiv:2511.07317):
      - Rewards normalized to [-1.0, +1.0] (Appendix B)
      - Smooth partial rewards (§B.1)
      - Auto-curriculum promotion (§2.2)
      - Rubric monitoring functions (zero-weight)
      - Execute-with-time-limit guards
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        logger.info("Initializing CloudSREEnvironment...")

        # Orchestrator manages all services and infrastructure
        self.orchestrator = ServiceOrchestrator()
        self.orchestrator.start()

        # Episode state
        self._step_count = 0
        self._max_steps = 15
        self._history = []
        self._current_scenario = None
        self._current_task_id = os.environ.get("TASK_TIER", "warmup")
        self._cascade_triggered = False
        self._cascade_alert = ""
        self._episode_id = ""
        self._current_phase = "triage"

        # State object for /state endpoint
        self._state = CloudSREState(episode_id=str(uuid4()), step_count=0)

        # Episode counter for curriculum
        self._episode_count = 0

        # Adaptive sampling — self-improving scenario selection (Theme #4)
        self._performance_tracker = PerformanceTracker(window_size=20)

        # RLVE: Auto-curriculum — current auto-tier (only used when TASK_TIER=auto)
        self._auto_tier = "warmup"

        logger.info("CloudSREEnvironment initialized successfully")

    def reset(self, task_id: str = None, **kwargs) -> CloudSREObservation:
        """Reset environment for a new episode.

        Fast reset (<100ms) — no K8s pod stabilization needed.
        
        Args:
            task_id: Override the task tier for this episode.
                     If None, uses TASK_TIER env var (default: warmup).
        """
        if task_id is not None:
            self._current_task_id = task_id
        logger.info(f"reset() called with task_id={self._current_task_id}")
        try:
            return self._do_reset()
        except Exception as e:
            logger.error(f"FATAL: reset() failed: {e}", exc_info=True)
            raise

    def _do_reset(self) -> CloudSREObservation:
        # Log abandoned episode
        if self._step_count > 0 and self._history:
            raw_sum = sum(h.get("reward", 0) for h in self._history)
            total_reward = raw_sum / max(self._step_count, 1)
            logger.info(
                f"  === EPISODE ABANDONED | task={self._current_task_id} | "
                f"steps={self._step_count} | total_reward={total_reward:.2f} ==="
            )

        # Step 1: Fast reset of all services + infra
        self.orchestrator.reset()
        logger.info("Infrastructure reset complete (<100ms)")

        # Step 2: Pick scenario — with auto-curriculum promotion (RLVE §2.2)
        task_id = self._current_task_id

        # RLVE auto-curriculum: promote tier when agent masters current one
        if task_id == "auto":
            if self._performance_tracker.should_promote(self._auto_tier):
                idx = PerformanceTracker.TIER_ORDER.index(self._auto_tier)
                if idx < len(PerformanceTracker.TIER_ORDER) - 1:
                    old_tier = self._auto_tier
                    self._auto_tier = PerformanceTracker.TIER_ORDER[idx + 1]
                    logger.info(f"RLVE AUTO-PROMOTE: {old_tier} → {self._auto_tier}")
            effective_tier = self._auto_tier
            task_cfg = TASK_CONFIGS.get(effective_tier, TASK_CONFIGS["warmup"])
        else:
            task_cfg = TASK_CONFIGS.get(task_id, TASK_CONFIGS["warmup"])

        self._max_steps = int(os.environ.get("MAX_STEPS", str(task_cfg["max_steps"])))

        scenario = task_cfg["scenario_picker"](
            self.orchestrator,
            performance_tracker=self._performance_tracker,
        )
        self._current_scenario = scenario

        # Step 3: Inject the fault
        inject_result = self.orchestrator.inject_fault(
            scenario.failure_type,
            scenario.params,
        )
        logger.info(f"Fault injected: {inject_result}")

        # Step 4: Inject misleading signals (red herrings) if configured
        for service, message in scenario.misleading_signals.items():
            self.orchestrator.inject_fault("misleading_signal", {
                "target": service,
                "message": message,
            })

        # Step 5: Arm cascade if present
        if scenario.cascade_rules:
            cascade = scenario.cascade_rules[0]
            self.orchestrator.inject_cascade(
                scenario.failure_type,
                cascade.cascade_type,
                {
                    "target": scenario.target_service,  # PRIMARY service to watch
                    "cascade_params": {"target": cascade.affected_service},
                },
            )

        # Step 6: Initialize episode state
        self._step_count = 0
        self._history = []
        self._cascade_triggered = False
        self._cascade_alert = ""
        self._current_phase = "triage"
        self._episode_id = str(uuid4())
        self._episode_count += 1
        self._cmd_type_counts = {}  # Reset diminishing returns
        self._phases_awarded = set()  # Reset phase progression tracking

        self._state = CloudSREState(
            episode_id=self._episode_id,
            step_count=0,
            scenario_id=scenario.scenario_id,
            task_id=task_id,
            difficulty=scenario.difficulty,
            root_cause_service=scenario.target_service,
            root_cause_description=scenario.root_cause,
            correct_fix=scenario.correct_fix_description,
            tier=task_cfg.get("tier", 1),
            current_phase="triage",
        )

        # Build initial observation
        service_health = self.orchestrator.check_health()

        return CloudSREObservation(
            alert=scenario.alert_message,
            scenario_id=scenario.scenario_id,
            task_id=task_id,
            command_output=(
                f"🔔 PAGERDUTY ALERT 🔔\n\n"
                f"{scenario.alert_message}\n\n"
                f"You are the on-call SRE. Investigate and resolve this incident.\n"
                f"Available commands:\n"
                f"  curl http://localhost:<port>/healthz   — Check service health\n"
                f"  curl http://localhost:<port>/metrics    — View metrics\n"
                f"  cat /var/log/<service>/error.log        — Read error logs\n"
                f"  sqlite3 /data/app.db '<SQL>'           — Query database\n"
                f"  ps aux                                  — List processes\n"
                f"  status                                  — Service overview\n\n"
                f"Services: payment(:8001) auth(:8002) worker(:8003) frontend(:8004) cache(:8005) notification(:8006)\n"
            ),
            service_health=service_health,
            step_number=0,
            max_steps=self._max_steps,
            phase="triage",
            history=[],
            feedback="Incident detected. Start by checking service health and reading error logs.",
            cascade_triggered=False,
            cascade_alert="",
            done=False,
            reward=0.0,
        )

    def step(self, action: CloudSREAction) -> CloudSREObservation:
        """Execute one agent step — run a real command and return real results."""
        self._step_count += 1
        self._state.step_count = self._step_count
        cmd = action.command.strip()

        logger.info(f"  Step {self._step_count}/{self._max_steps}: {cmd}")

        # ── Repeat Detection ──────────────────────────────────────────
        repeat_count = sum(1 for h in self._history if h.get("command") == cmd)
        if repeat_count >= 2:
            output = (
                f"BLOCKED: You already tried this command {repeat_count + 1} times. "
                "Try a different approach."
            )
            reward = -0.5
            feedback = "Command blocked — try something different."
            cmd_type = "unknown"
        else:
            # ── Execute with Time Limit (Daniel/Unsloth workshop) ─────
            output, cmd_type = self._execute_with_time_limit(cmd)

            # ── Phase Detection ───────────────────────────────────────
            self._current_phase = self._detect_phase(cmd_type)
            self._state.current_phase = self._current_phase

            # ── Dense Reward Calculation ──────────────────────────────
            reward, feedback = self._calculate_reward(cmd, output, cmd_type, repeat_count)

        if repeat_count == 1:
            reward -= 0.2
            feedback += " (repeated command — penalty)"

        # RLVE: Clamp reward to [-1.0, +1.0] (Appendix B)
        reward = self._clamp_reward(reward)

        logger.info(f"    → reward={reward:.2f} | phase={self._current_phase} | {feedback[:80]}")

        # ── Check for Cascade Trigger ────────────────────────────────
        cascade_result = self.orchestrator.check_and_trigger_cascade()
        if cascade_result and not self._cascade_triggered:
            self._cascade_triggered = True
            self._cascade_alert = (
                f"🔔 CASCADE ALERT: A secondary failure has been triggered!\n"
                f"{cascade_result}\n"
                f"The system is still degraded. Investigate and resolve the new issue."
            )
            logger.info(f"  CASCADE TRIGGERED: {cascade_result}")
            feedback += " ⚠️ CASCADE: A new failure has been triggered by your fix!"

        # ── Done Check ───────────────────────────────────────────────
        done = False
        all_healthy = self._check_all_resolved()
        cascade_just_fired = (cascade_result is not None)  # Cascade triggered THIS step

        if all_healthy and cmd_type in ("fix", "health_check") and not cascade_just_fired:
            done = True
            # RLVE: Smooth partial rewards (§B.1 + Lewis: Wordle green/yellow)
            efficiency = 1.0 - (self._step_count / self._max_steps)
            reward = 0.5 + 0.5 * (efficiency ** 2)  # Range: [0.5, 1.0]
            feedback = f"🎉 Incident resolved! (efficiency: {efficiency:.1%})"

            if self._cascade_triggered:
                reward = min(1.0, reward + 0.2)  # Cascade bonus, capped at 1.0
                feedback += " + cascade handled!"

        elif self._step_count >= self._max_steps:
            done = True
            # RLVE: Timeout penalty, capped at -1.0
            reward = -1.0
            feedback = "⏰ Timeout — incident remains unresolved."

        # ── Record History ───────────────────────────────────────────
        self._history.append({
            "step": self._step_count,
            "command": cmd,
            "output": output[:200],
            "reward": reward,
            "feedback": feedback,
            "phase": self._current_phase,
            "cmd_type": cmd_type,
        })

        # ── Log Episode Completion ───────────────────────────────────
        if done:
            raw_sum = sum(h.get("reward", 0) for h in self._history)
            total_reward = raw_sum / max(self._step_count, 1)
            resolved = "resolved" in feedback.lower()
            self._state.is_resolved = resolved
            self._state.cumulative_reward = total_reward

            logger.info(
                f"  === EPISODE DONE: {'RESOLVED' if resolved else 'FAILED'} | "
                f"task={self._current_task_id} | scenario={self._current_scenario.scenario_id if self._current_scenario else 'N/A'} | "
                f"steps={self._step_count} | total_reward={total_reward:.2f} ==="
            )

            # Save transcript with rubric monitoring (RLVE)
            self._save_transcript(resolved)
            self._step_count = 0

        # ── Build Observation ────────────────────────────────────────
        service_health = self.orchestrator.check_health()

        return CloudSREObservation(
            alert=self._current_scenario.alert_message if self._current_scenario and not done else "",
            scenario_id=self._current_scenario.scenario_id if self._current_scenario else "",
            task_id=self._current_task_id,
            command_output=output,
            service_health=service_health,
            step_number=self._step_count,
            max_steps=self._max_steps,
            phase=self._current_phase,
            history=[h["command"] for h in self._history],
            feedback=feedback,
            cascade_triggered=self._cascade_triggered,
            cascade_alert=self._cascade_alert if self._cascade_triggered else "",
            done=done,
            reward=reward,
        )

    # ── Phase Detection ──────────────────────────────────────────────────

    def _detect_phase(self, cmd_type: str) -> str:
        """Detect which SRE workflow phase the agent is in based on commands.

        triage → investigation → mitigation → fix → verification
        """
        phase_map = {
            "health_check": "triage" if self._step_count <= 2 else "verification",
            "metrics": "investigation",
            "logs": "investigation",
            "database": "investigation",
            "diagnosis": "investigation",
            "fix": "fix",
            "process": "investigation",
            "unknown": self._current_phase,
        }
        new_phase = phase_map.get(cmd_type, self._current_phase)

        # Phase can only advance, not go backwards (except verification)
        phase_order = ["triage", "investigation", "mitigation", "fix", "verification"]
        current_idx = phase_order.index(self._current_phase) if self._current_phase in phase_order else 0
        new_idx = phase_order.index(new_phase) if new_phase in phase_order else current_idx

        if new_idx >= current_idx or new_phase == "verification":
            return new_phase
        return self._current_phase

    # ── RLVE: Reward Clamping (Appendix B) ────────────────────────────────

    def _clamp_reward(self, reward: float) -> float:
        """RLVE Appendix B: all environments use [-1.0, +1.0] range."""
        return max(-1.0, min(1.0, reward))

    # ── RLVE: Execute with Time Limit (Daniel/Unsloth workshop) ──────────

    def _execute_with_time_limit(self, cmd: str, timeout: float = 10.0) -> tuple:
        """Prevent infinite loops and reward hacking.

        Daniel (Unsloth workshop): guard against commands that hang forever.
        If a command takes >10s, kill it and return error + penalty.
        """
        result = [None, None]
        def _run():
            result[0], result[1] = self.orchestrator.executor.execute(cmd)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return "ERROR: Command timed out after 10s. This may indicate a loop.", "unknown"
        return result[0], result[1]

    # ── Dense Reward Calculation (RLVE-aligned) ──────────────────────────

    def _calculate_reward(
        self, cmd: str, output: str, cmd_type: str, repeat_count: int
    ) -> tuple:
        """Calculate dense per-step reward.

        RLVE-aligned: all per-step rewards are small fractions of [-1, +1].
        Resolution/timeout rewards are handled separately in step().

        Anti-gaming measures:
          - Diminishing returns on command TYPES (not just exact strings)
          - Service name bonus requires prior diagnostic work
          - Phase progression bonus only on first transition
        """
        reward = 0.0
        feedback_parts = []

        # Track command type counts for diminishing returns
        if not hasattr(self, '_cmd_type_counts'):
            self._cmd_type_counts = {}
        type_count = self._cmd_type_counts.get(cmd_type, 0)
        self._cmd_type_counts[cmd_type] = type_count + 1

        # Base reward by command type (RLVE-scaled)
        # Diminishing returns: reward * (1 / (1 + type_count))
        type_rewards = {
            "health_check": 0.05,
            "metrics": 0.08,
            "logs": 0.08,
            "database": 0.10,
            "diagnosis": 0.04,
            "fix": 0.12,
            "unknown": -0.05,
        }
        base = type_rewards.get(cmd_type, 0.0)
        diminishing = 1.0 / (1.0 + type_count)  # 1st=1.0, 2nd=0.5, 3rd=0.33...
        reward += base * diminishing

        # Bonus for investigating the RIGHT service — ONLY after at least
        # 1 diagnostic command (prevents free bonus on first fix command)
        has_done_diagnostics = any(
            h.get("cmd_type") in ("health_check", "logs", "database", "metrics")
            for h in self._history
        )
        if (self._current_scenario and self._current_scenario.target_service in cmd
                and has_done_diagnostics and cmd_type != "diagnosis"):
            reward += 0.06
            feedback_parts.append("Investigating relevant service.")

        # Penalty for errors in output
        if "Error:" in output or "error:" in output or "refused" in output:
            if cmd_type == "health_check":
                reward += 0.04
                feedback_parts.append("Found service error — dig deeper.")
            elif cmd_type == "fix":
                reward -= 0.10
                feedback_parts.append("Fix attempt failed.")

        # Phase progression bonus — only on FIRST transition to each phase
        if not hasattr(self, '_phases_awarded'):
            self._phases_awarded = set()
        new_phase = self._detect_phase(cmd_type)
        if new_phase != self._current_phase and new_phase not in self._phases_awarded:
            reward += 0.04
            feedback_parts.append("Advancing to next SRE workflow phase.")
            self._phases_awarded.add(new_phase)

        # Cascade-aware rewards
        if self._cascade_triggered and self._cascade_alert:
            if cmd_type in ("health_check", "logs", "metrics"):
                reward += 0.04
                feedback_parts.append("Investigating cascade failure.")

        feedback = " ".join(feedback_parts) if feedback_parts else "Command executed."
        return round(reward, 3), feedback

    # ── Resolution Check ─────────────────────────────────────────────────

    def _check_all_resolved(self) -> bool:
        """Check if all services are healthy.

        For cascade scenarios, both primary AND cascade must be resolved.
        The cascade must actually TRIGGER before the episode can end.
        """
        health = self.orchestrator.check_health()
        all_healthy = all(
            svc["status"] == "healthy"
            for svc in health.values()
        )

        # CRITICAL: If scenario has cascade rules but cascade hasn't triggered yet,
        # the episode is NOT resolved — the agent fixed the primary fault but
        # the cascade hasn't fired yet. We must wait for it.
        has_cascade_rules = (
            self._current_scenario
            and hasattr(self._current_scenario, 'cascade_rules')
            and self._current_scenario.cascade_rules
        )

        if has_cascade_rules and not self._cascade_triggered:
            # Primary fault fixed but cascade hasn't triggered yet.
            # Don't end the episode — the cascade should fire soon.
            return False

        # If cascade was triggered, it must also be resolved
        if self._cascade_triggered:
            cascade_service = (
                self._current_scenario.cascade_rules[0].affected_service
                if self._current_scenario and self._current_scenario.cascade_rules
                else None
            )
            if cascade_service and cascade_service in health:
                if health[cascade_service]["status"] != "healthy":
                    return False

        return all_healthy

    # ── Transcript Saving + Rubric Monitoring (RLVE) ─────────────────────

    def _compute_rubrics(self) -> dict:
        """Zero-weight monitoring rewards — detect reward hacking.

        David (Meta workshop): 'average rollout length reveals reward hacking early.'
        Daniel (Unsloth workshop): 'use zero-weight rubric functions to monitor
        without affecting gradients.'
        """
        cmds = [h["command"] for h in self._history]
        phases = [h.get("phase", "unknown") for h in self._history]
        return {
            "rubric_avg_steps": len(self._history),
            "rubric_unique_cmds": len(set(cmds)),
            "rubric_repeat_rate": round(1 - (len(set(cmds)) / max(len(cmds), 1)), 3),
            "rubric_phases_covered": len(set(phases)),
            "rubric_target_investigated": any(
                self._current_scenario.target_service in c for c in cmds
            ) if self._current_scenario else False,
        }

    def _save_transcript(self, resolved: bool):
        """Save episode transcript to JSONL and update adaptive sampling."""
        # Record outcome for adaptive weighted sampling (Theme #4)
        if self._current_scenario:
            self._performance_tracker.record(
                self._current_scenario.scenario_id, resolved
            )
            stats = self._performance_tracker.get_stats()
            logger.info(f"  Adaptive sampling stats: {len(stats)} scenarios tracked")

        # RLVE: Compute rubric metrics for hacking detection
        rubrics = self._compute_rubrics()
        if rubrics["rubric_avg_steps"] < 2 or rubrics["rubric_unique_cmds"] <= 1:
            logger.warning(f"  ⚠️ POTENTIAL REWARD HACKING: {rubrics}")

        try:
            transcript = {
                "episode": self._episode_count,
                "task_id": self._current_task_id,
                "scenario_id": self._current_scenario.scenario_id if self._current_scenario else "",
                "resolved": resolved,
                "steps": self._step_count,
                "total_reward": sum(h.get("reward", 0) for h in self._history) / max(self._step_count, 1),
                "difficulty": self._current_scenario.difficulty if self._current_scenario else 0.0,
                "cascade_triggered": self._cascade_triggered,
                "auto_tier": self._auto_tier if self._current_task_id == "auto" else None,
                "alert": self._current_scenario.alert_message if self._current_scenario else "",
                "root_cause": self._current_scenario.root_cause if self._current_scenario else "",
                "history": self._history,
                "rubrics": rubrics,
                "adaptive_sampling": self._performance_tracker.get_stats(),
            }
            log_path = os.environ.get("EPISODE_LOG", "episode_transcripts.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(transcript) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save transcript: {e}")

    @property
    def state(self) -> CloudSREState:
        return self._state
