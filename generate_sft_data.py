"""
Generate SFT training data by running expert trajectories against the live environment.

This creates REAL demonstrations: reset → diagnose → fix → verify.
Each trajectory becomes a multi-turn chat example for SFT.

Output: sft_training_data.jsonl (chat format compatible with Unsloth SFT)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx
import json
import random
import time

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a production cloud platform.

AVAILABLE COMMANDS:
  status                    - Check all service health
  curl http://localhost:<PORT>/healthz  - Check specific service health
  cat /var/log/<service>/error.log      - Read service error logs
  cat /var/log/<service>/access.log     - Read service access logs
  ps aux | grep <service>              - Check if process is running
  restart_service <service>            - Restart a crashed/degraded service
  queue drain <N>                      - Drain N messages from the queue
  sqlite3 /data/db/main.db "<SQL>"     - Query the database
  kill -9 <PID>                        - Kill a stuck process

SERVICES: payment(8001), auth(8002), worker(8003), frontend(8004), cache(8005), notification(8006)

SRE WORKFLOW:
1. TRIAGE: Run 'status' to identify which services are unhealthy
2. INVESTIGATE: Read logs and metrics of broken services
3. FIX: Apply the appropriate fix (restart, drain, etc.)
4. VERIFY: Run 'status' again to confirm resolution

Always output ONLY the next command to run. No explanations."""

# Expert strategies for each fault type
EXPERT_STRATEGIES = {
    "process_crash": [
        ("status", "Check all service health to identify the issue"),
        ("cat /var/log/{service}/error.log", "Read error logs to understand the crash"),
        ("restart_service {service}", "Restart the crashed service"),
        ("status", "Verify all services are healthy"),
    ],
    "db_lock": [
        ("status", "Check all service health"),
        ("cat /var/log/payment/error.log", "Check payment logs for DB errors"),
        ("sqlite3 /data/db/main.db \"SELECT * FROM locks\"", "Inspect database locks"),
        ("restart_service payment", "Restart payment to clear DB lock"),
        ("status", "Verify resolution"),
    ],
    "queue_overflow": [
        ("status", "Check all service health"),
        ("cat /var/log/worker/error.log", "Check worker logs"),
        ("queue drain 50", "Drain queue backlog - first batch"),
        ("queue drain 50", "Drain queue backlog - second batch"),
        ("status", "Verify queue depth is manageable"),
    ],
    "cache_invalidation": [
        ("status", "Check all service health"),
        ("cat /var/log/cache/error.log", "Check cache logs for invalidation"),
        ("curl http://localhost:8005/healthz", "Check cache health directly"),
        ("restart_service cache", "Restart cache to rebuild"),
        ("status", "Verify cache is healthy"),
    ],
    "latency_injection": [
        ("status", "Check all service health"),
        ("cat /var/log/{service}/error.log", "Check logs for timeout/latency"),
        ("curl http://localhost:{port}/healthz", "Direct health check"),
        ("restart_service {service}", "Restart to clear latency injection"),
        ("status", "Verify resolution"),
    ],
    "webhook_storm": [
        ("status", "Check all service health"),
        ("cat /var/log/{service}/error.log", "Check logs for webhook storm"),
        ("restart_service {service}", "Restart to clear webhook backlog"),
        ("status", "Verify resolution"),
    ],
}

# Port mapping
PORT_MAP = {"payment": 8001, "auth": 8002, "worker": 8003, "frontend": 8004, "cache": 8005, "notification": 8006}

TIERS = ["warmup", "single_fault", "cascade", "multi_cascade", "adversarial"]

sft_examples = []
total_runs = 0
resolved_runs = 0

for tier in TIERS:
    # Run multiple episodes per tier
    episodes_per_tier = 8 if tier in ("warmup", "single_fault") else 5
    
    for ep in range(episodes_per_tier):
        print(f"\n--- {tier} episode {ep+1}/{episodes_per_tier} ---")
        
        # Reset
        try:
            r = client.post("/reset", json={"task_id": tier})
            data = r.json()
            obs = data.get("observation", data)
        except Exception as e:
            print(f"  Reset failed: {e}")
            continue
        
        scenario_id = obs.get("scenario_id", "")
        health = obs.get("service_health", {})
        alert = obs.get("alert", "")
        max_steps = obs.get("max_steps", 10)
        
        broken = []
        for svc, h in health.items():
            if h.get("status") != "healthy":
                broken.append((svc, h.get("status"), h.get("error", "")))
        
        print(f"  Scenario: {scenario_id}")
        print(f"  Broken: {[b[0] for b in broken]}")
        
        if not broken:
            print(f"  No faults visible, skipping")
            continue
        
        # Determine fault type from scenario_id
        fault_type = "process_crash"  # default
        if "db_lock" in scenario_id: fault_type = "db_lock"
        elif "queue" in scenario_id: fault_type = "queue_overflow"
        elif "cache" in scenario_id: fault_type = "cache_invalidation"
        elif "timeout" in scenario_id or "latency" in scenario_id: fault_type = "latency_injection"
        elif "webhook" in scenario_id or "storm" in scenario_id: fault_type = "webhook_storm"
        elif "crash" in scenario_id or "outage" in scenario_id: fault_type = "process_crash"
        
        target_svc = broken[0][0]
        target_port = PORT_MAP.get(target_svc, 8001)
        
        # Get expert strategy
        strategy = EXPERT_STRATEGIES.get(fault_type, EXPERT_STRATEGIES["process_crash"])
        
        # Build conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # First user message is the alert + initial health
        health_text = "\n".join(
            f"  {svc}: {h.get('status')} {'| ' + str(h.get('error',''))[:80] if h.get('error') else ''}"
            for svc, h in health.items()
        )
        
        first_user = f"ALERT: {alert}\n\nHEALTH:\n{health_text}\n\nStep 1/{max_steps}. Next command:"
        messages.append({"role": "user", "content": first_user})
        
        total_runs += 1
        episode_resolved = False
        
        for step_idx, (cmd_template, reasoning) in enumerate(strategy):
            # Format command with service name
            cmd = cmd_template.format(service=target_svc, port=target_port)
            
            # Assistant response: just the command (SRE style)
            messages.append({"role": "assistant", "content": cmd})
            
            # Execute the command
            try:
                r = client.post("/step", json={"action": {"command": cmd}})
                d = r.json()
                step_obs = d.get("observation", d)
            except Exception as e:
                print(f"  Step {step_idx+1} failed: {e}")
                break
            
            done = d.get("done", step_obs.get("done", False))
            reward = float(d.get("reward", step_obs.get("reward", 0)))
            cmd_output = step_obs.get("command_output", "")[:500]
            feedback = step_obs.get("feedback", "")
            step_health = step_obs.get("service_health", {})
            cascade = step_obs.get("cascade_triggered", False)
            
            if done:
                print(f"  RESOLVED at step {step_idx+1}! reward={reward:.2f}")
                episode_resolved = True
                resolved_runs += 1
                break
            
            # Check if cascade triggered — need to fix cascade target too
            if cascade:
                cascade_broken = [s for s, h in step_health.items() if h.get("status") != "healthy"]
                if cascade_broken:
                    # Add cascade fix steps
                    cascade_svc = cascade_broken[0]
                    
                    # Build next user message
                    cascade_health = "\n".join(
                        f"  {svc}: {h.get('status')} {'| ' + str(h.get('error',''))[:80] if h.get('error') else ''}"
                        for svc, h in step_health.items()
                    )
                    cascade_alert = step_obs.get("cascade_alert", "")
                    next_user = f"OUTPUT: {cmd_output[:200]}\n{feedback}\n\nCASCADE ALERT: {cascade_alert}\n\nHEALTH:\n{cascade_health}\n\nStep {step_idx+3}/{max_steps}. Next command:"
                    messages.append({"role": "user", "content": next_user})
                    
                    # Fix cascade
                    cascade_cmd = f"restart_service {cascade_svc}"
                    messages.append({"role": "assistant", "content": cascade_cmd})
                    
                    try:
                        r_cas = client.post("/step", json={"action": {"command": cascade_cmd}})
                        d_cas = r_cas.json()
                        done_cas = d_cas.get("done", d_cas.get("observation", d_cas).get("done", False))
                        if done_cas:
                            print(f"  RESOLVED after cascade fix! reward={d_cas.get('reward', 0):.2f}")
                            episode_resolved = True
                            resolved_runs += 1
                            break
                    except:
                        pass
            
            # If not last step, add the observation as next user message
            if step_idx < len(strategy) - 1:
                step_health_text = "\n".join(
                    f"  {svc}: {h.get('status')} {'| ' + str(h.get('error',''))[:80] if h.get('error') else ''}"
                    for svc, h in step_health.items()
                )
                next_user = f"OUTPUT: {cmd_output[:200]}\n{feedback}\n\nHEALTH:\n{step_health_text}\n\nStep {step_idx+2}/{max_steps}. Next command:"
                messages.append({"role": "user", "content": next_user})
        
        # Only save if resolved (successful demonstrations only)
        if episode_resolved:
            sft_examples.append({
                "messages": messages,
                "scenario_id": scenario_id,
                "tier": tier,
                "steps": step_idx + 1,
            })
            print(f"  Saved ({len(sft_examples)} total)")
        else:
            print(f"  NOT resolved, skipping (bad demo)")
        
        time.sleep(0.3)

# Save to JSONL
output_path = "sft_training_data.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for example in sft_examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"\n{'='*60}")
print(f"SFT DATA GENERATION COMPLETE")
print(f"{'='*60}")
print(f"  Total episodes:  {total_runs}")
print(f"  Resolved:        {resolved_runs}")
print(f"  Resolution rate: {resolved_runs/max(total_runs,1)*100:.0f}%")
print(f"  Saved examples:  {len(sft_examples)}")
print(f"  Output:          {output_path}")

client.close()
