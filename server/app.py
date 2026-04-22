"""
CloudSRE v2 — FastAPI Application.

HTTP server exposing the CloudSREEnvironment over HTTP and WebSocket.
Matches the OpenEnv server contract (create_app).

Kube SRE Gym equivalent: server/app.py (88 lines)
Ours adds: /tasks endpoint (required by OpenEnv validator) + more metadata.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from models import CloudSREAction, CloudSREObservation
    from server.cloud_sre_environment import CloudSREEnvironment
except ImportError:
    from cloud_sre_v2.models import CloudSREAction, CloudSREObservation
    from cloud_sre_v2.server.cloud_sre_environment import CloudSREEnvironment


# Create the OpenEnv app
app = create_app(
    CloudSREEnvironment,
    CloudSREAction,
    CloudSREObservation,
    env_name="cloud_sre_v2",
    max_concurrent_envs=1,
)


# ── /tasks endpoint — required by OpenEnv Phase 2 validator ───────────────

@app.get("/tasks", tags=["Environment Info"], summary="List all tasks with graders")
async def list_tasks():
    """Return available tasks — required by OpenEnv validator."""
    return [
        {
            "id": "warmup",
            "name": "Single-Service Failure (Warmup)",
            "description": "Single clear failure with obvious signals. No red herrings, no cascades.",
            "difficulty": "easy",
            "time_limit_seconds": 300,
            "max_steps": 10,
            "grader": "server.graders.grade_warmup",
            "action_schema": {
                "command": "SRE command (curl, cat, sqlite3, ps, kill, restart_service, queue, ...)"
            },
        },
        {
            "id": "single_fault",
            "name": "Single Fault with Red Herrings",
            "description": "Single root cause with misleading signals from other services.",
            "difficulty": "medium",
            "time_limit_seconds": 600,
            "max_steps": 15,
            "grader": "server.graders.grade_single_fault",
            "action_schema": {
                "command": "SRE command to diagnose and fix the incident"
            },
        },
        {
            "id": "cascade",
            "name": "Cascading Failure",
            "description": "Single fault that triggers a secondary failure after the primary fix.",
            "difficulty": "hard",
            "time_limit_seconds": 900,
            "max_steps": 20,
            "grader": "server.graders.grade_cascade",
            "action_schema": {
                "command": "SRE command — must handle both primary fault and the cascade"
            },
        },
        {
            "id": "multi_cascade",
            "name": "Multi-Service Cascading Outage",
            "description": "2+ cascading failures requiring prioritized resolution.",
            "difficulty": "expert",
            "time_limit_seconds": 1200,
            "max_steps": 25,
            "grader": "server.graders.grade_multi_cascade",
            "action_schema": {
                "command": "SRE command — must resolve all failures in correct priority order"
            },
        },
        {
            "id": "adversarial",
            "name": "LLM-Designed Adversarial Incident",
            "description": "Dynamically generated incident targeting agent weaknesses.",
            "difficulty": "adversarial",
            "time_limit_seconds": 1800,
            "max_steps": 30,
            "grader": "server.graders.grade_adversarial",
            "action_schema": {
                "command": "Full SRE workflow: triage → investigate → mitigate → fix → verify"
            },
        },
    ]


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
