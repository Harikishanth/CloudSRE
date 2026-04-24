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

from fastapi.responses import HTMLResponse


# ── Stateful Environment Factory ─────────────────────────────────────────
# CloudSRE manages REAL infrastructure: 6 service subprocesses, ports,
# SQLite databases, and message queues. These resources must persist across
# the full episode lifecycle (reset → step → step → ... → done).
#
# We use a singleton factory pattern to ensure the environment instance
# (and its managed resources) survives across HTTP request boundaries.
# The close() override prevents premature resource teardown between calls.
#
# This pattern is standard for environments with real infrastructure —
# see also: KubeSRE Gym, SWE-bench, WebArena.

_singleton_env = None

class PersistentCloudSRE(CloudSREEnvironment):
    """CloudSRE with persistent infrastructure across HTTP request boundaries.

    Overrides close() to prevent the HTTP server from tearing down
    managed subprocesses and infrastructure between API calls.
    """

    def close(self):
        """No-op: infrastructure must persist across the episode lifecycle."""
        pass

def _env_factory():
    """Return the persistent environment instance (created on first call)."""
    global _singleton_env
    if _singleton_env is None:
        _singleton_env = PersistentCloudSRE()
    return _singleton_env


# Create the OpenEnv app
app = create_app(
    _env_factory,
    CloudSREAction,
    CloudSREObservation,
    env_name="cloud_sre_v2",
    max_concurrent_envs=1,
)


# ── Landing Page (Root Endpoint) ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CloudSRE v2 Environment</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0f1115;
                --surface: #1e2128;
                --primary: #6366f1;
                --text: #e2e8f0;
                --muted: #94a3b8;
                --success: #10b981;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                padding: 2rem;
            }
            .container {
                max-width: 800px;
                width: 100%;
                background: var(--surface);
                padding: 3rem;
                border-radius: 1rem;
                box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
                border: 1px solid rgba(255,255,255,0.05);
            }
            h1 {
                font-weight: 800;
                font-size: 2.5rem;
                margin-top: 0;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p {
                color: var(--muted);
                line-height: 1.6;
                font-size: 1.1rem;
            }
            .status {
                display: inline-flex;
                align-items: center;
                background: rgba(16, 185, 129, 0.1);
                color: var(--success);
                padding: 0.5rem 1rem;
                border-radius: 2rem;
                font-weight: 600;
                font-size: 0.9rem;
                margin-bottom: 1.5rem;
            }
            .status::before {
                content: '';
                display: inline-block;
                width: 8px;
                height: 8px;
                background: var(--success);
                border-radius: 50%;
                margin-right: 8px;
                box-shadow: 0 0 10px var(--success);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-top: 2rem;
            }
            .card {
                background: rgba(255,255,255,0.02);
                padding: 1.5rem;
                border-radius: 0.75rem;
                border: 1px solid rgba(255,255,255,0.05);
            }
            .card h3 {
                margin: 0 0 0.5rem 0;
                font-size: 1.1rem;
                color: #fff;
            }
            .card p {
                font-size: 0.9rem;
                margin: 0;
            }
            .code-block {
                background: #000;
                padding: 1rem;
                border-radius: 0.5rem;
                font-family: monospace;
                color: #34d399;
                margin-top: 2rem;
                overflow-x: auto;
            }
            .links {
                margin-top: 2rem;
                display: flex;
                gap: 1rem;
            }
            a {
                color: var(--primary);
                text-decoration: none;
                font-weight: 600;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="status">System Online & Ready</div>
            <h1>CloudSRE v2 Environment</h1>
            <p>Welcome to the <b>CloudSRE v2</b> OpenEnv instance. This is a production-grade, 6-microservice environment with cascading failure patterns designed for training RL agents in Site Reliability Engineering.</p>
            
            <div class="grid">
                <div class="card">
                    <h3>6 Microservices</h3>
                    <p>Payment, Auth, Worker, Frontend, Cache, and Notification.</p>
                </div>
                <div class="card">
                    <h3>5 Difficulty Tiers</h3>
                    <p>From warmups to full cascading outages.</p>
                </div>
                <div class="card">
                    <h3>RLVE Aligned</h3>
                    <p>Normalized rewards, auto-curriculums, and time limits.</p>
                </div>
            </div>

            <div class="code-block">
                # Ready for training. Use the OpenEnv client:
                client = httpx.Client(base_url="https://dardrax-cloudsre-environment.hf.space")
                client.post("/reset", json={"task_id": "warmup"})
            </div>
            
            <div class="links">
                <a href="/docs">→ View API Specs (Swagger UI)</a>
                <a href="/tasks">→ View Task Tiers</a>
            </div>
        </div>
    </body>
    </html>
    """


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
