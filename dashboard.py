import os
import requests
import gradio as gr
from typing import Dict, Any

# Configure environment
ENV_URL = os.environ.get("CLOUDSRE_URL", "https://dardrax-cloudsre-environment.hf.space")

SERVICES = {
    "us-east-1": ["payment", "auth", "billing", "gateway", "loadbalancer", "config"],
    "eu-west-1": ["worker", "scheduler", "search", "storage", "metrics_collector"],
    "ap-south-1": ["frontend", "cache", "notification", "email", "dns"]
}

def trigger_incident(difficulty: int, task_id: str):
    """Trigger an incident via the reset endpoint and return state."""
    try:
        response = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id, "difficulty": difficulty},
            timeout=60 # High timeout for space wake-up
        )
        if response.status_code == 200:
            data = response.json()
            obs = data.get("observation", {})
            alert = obs.get("alert", "Incident triggered!")
            service_health = obs.get("service_health", {})
            return alert, service_health
        return f"Error: {response.text}", {}
    except Exception as e:
        return f"Failed to connect to environment: {str(e)}", {}

def execute_command(command: str, current_health: dict):
    """Execute a step command in the environment and return updated state."""
    if not command:
        return "Please enter a command.", current_health
    try:
        response = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"command": command}},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            obs = data.get("observation", {})
            out = obs.get("command_output", "")
            reward = data.get("reward", 0.0)
            service_health = obs.get("service_health", {})
            return f"➔ {command}\n{out}\n\n[Reward: {reward}]", service_health
        return f"Error: {response.text}", current_health
    except Exception as e:
        return f"Failed to connect: {str(e)}", current_health

def format_service_box(service: str, status_data: dict) -> str:
    """Format HTML for a service box."""
    status = status_data.get("status", "unknown") if status_data else "unknown"
    error = status_data.get("error", "") if status_data else ""
    
    if status == "healthy":
        color = "#2ecc71"
        icon = "✅"
    elif status == "degraded":
        color = "#f1c40f"
        icon = "⚠️"
    elif status in ("unhealthy", "down", "failed", "crashed"):
        color = "#e74c3c"
        icon = "❌"
    else:
        color = "#95a5a6"
        icon = "❓"

    html = f"""
    <div style="
        border: 2px solid {color};
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        min-width: 140px;
        flex: 1;
    ">
        <h3 style="margin: 0; color: {color}; font-size: 1.1em;">{icon} {service.upper()}</h3>
        <p style="margin: 5px 0 0 0; font-size: 0.8em; color: #555; word-wrap: break-word;">
            {error if error else 'Operational'}
        </p>
    </div>
    """
    return html

def render_dashboard(health_data: dict):
    """Render the full dashboard HTML from state."""
    if not health_data:
        health_data = {} # Default empty state (Operational/Unknown)

    html_parts = []
    for region, services in SERVICES.items():
        html_parts.append(f"<h2 style='margin-bottom: 5px; margin-top: 15px;'>🌍 {region.upper()}</h2>")
        html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 10px;'>")
        for svc in services:
            svc_health = health_data.get(svc, {})
            html_parts.append(format_service_box(svc, svc_health))
        html_parts.append("</div>")
    
    return "".join(html_parts)

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🚨 CloudSRE v2 — Live Observability Dashboard")
    gr.Markdown(f"Monitoring Environment: `{ENV_URL}`")
    
    # State variable to hold the microservice health dict
    health_state = gr.State({})
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🌐 Microservice Topology")
            dashboard_html = gr.HTML(render_dashboard({}))
            
            gr.Markdown("### 💻 Agent Terminal (`step`)")
            with gr.Row():
                cmd_input = gr.Textbox(placeholder="e.g., curl http://payment.us-east-1.internal/healthz", label="Command", scale=4)
                run_btn = gr.Button("Run Command", variant="primary", scale=1)
            terminal_output = gr.Textbox(label="Terminal Output", lines=8)
            
        with gr.Column(scale=1):
            gr.Markdown("### ⚡ Adversarial Designer (`reset`)")
            task_dropdown = gr.Dropdown(
                choices=["warmup", "single_fault", "cascade", "multi_cascade", "adversarial"],
                value="cascade",
                label="Task Tier"
            )
            difficulty_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Difficulty")
            trigger_btn = gr.Button("🔥 Trigger Incident", variant="primary")
            alert_output = gr.Textbox(label="PagerDuty Alert", lines=5)
            
    # Interactions
    trigger_btn.click(
        fn=trigger_incident,
        inputs=[difficulty_slider, task_dropdown],
        outputs=[alert_output, health_state]
    ).then(
        fn=render_dashboard,
        inputs=[health_state],
        outputs=dashboard_html
    )
    
    run_btn.click(
        fn=execute_command,
        inputs=[cmd_input, health_state],
        outputs=[terminal_output, health_state]
    ).then(
        fn=render_dashboard,
        inputs=[health_state],
        outputs=dashboard_html
    )
    
if __name__ == "__main__":
    print(f"Starting dashboard on port 7861... Connecting to {ENV_URL}")
    demo.launch(server_name="127.0.0.1", server_port=7861, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
