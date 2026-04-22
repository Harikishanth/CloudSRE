"""CloudSRE v2 Environment Client.

Sync WebSocket client for interacting with the CloudSRE environment server.
Used by train.py for GRPO training and by inference.py for evaluation.
"""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

from .models import CloudSREAction, CloudSREObservation, CloudSREState


class CloudSREEnv(
    EnvClient[CloudSREAction, CloudSREObservation, CloudSREState]
):
    """Client for the CloudSRE v2 Environment.

    OpenEnv v0.2.1 uses sync WebSocket calls.

    Example:
        >>> with CloudSREEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.alert)
        ...     result = client.step(CloudSREAction(command="curl http://localhost:8001/healthz"))
        ...     print(result.observation.command_output)
    """

    def __init__(self, base_url: str, **kwargs):
        # Service operations can be slow (subprocess spawn, cascade propagation)
        kwargs.setdefault("message_timeout_s", 120.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: CloudSREAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[CloudSREObservation]:
        obs_data = payload.get("observation", {})
        observation = CloudSREObservation(
            alert=obs_data.get("alert", ""),
            scenario_id=obs_data.get("scenario_id", ""),
            task_id=obs_data.get("task_id", ""),
            command_output=obs_data.get("command_output", ""),
            service_health=obs_data.get("service_health", {}),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 15),
            phase=obs_data.get("phase", "triage"),
            history=obs_data.get("history", []),
            feedback=obs_data.get("feedback", ""),
            cascade_triggered=obs_data.get("cascade_triggered", False),
            cascade_alert=obs_data.get("cascade_alert", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CloudSREState:
        return CloudSREState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario_id=payload.get("scenario_id", ""),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", 0.2),
            root_cause_service=payload.get("root_cause_service", ""),
            root_cause_description=payload.get("root_cause_description", ""),
            correct_fix=payload.get("correct_fix", ""),
            is_resolved=payload.get("is_resolved", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            steps_taken=payload.get("steps_taken", 0),
            cascade_triggered=payload.get("cascade_triggered", False),
            cascade_resolved=payload.get("cascade_resolved", False),
            primary_fix_applied=payload.get("primary_fix_applied", False),
            judge_persona=payload.get("judge_persona", "junior"),
            tier=payload.get("tier", 1),
            curriculum_stats=payload.get("curriculum_stats", {}),
            current_phase=payload.get("current_phase", "triage"),
        )
