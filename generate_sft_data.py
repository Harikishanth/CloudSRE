"""Generate 50+ SFT expert demonstrations across all tiers."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx, json, time

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

PORT_MAP = {"payment": 8001, "auth": 8002, "worker": 8003, "frontend": 8004, "cache": 8005, "notification": 8006}

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a production cloud platform.

AVAILABLE COMMANDS:
  status                    - Check all service health
  curl http://localhost:<PORT>/healthz  - Check specific service health
  cat /var/log/<service>/error.log      - Read service error logs
  restart_service <service>            - Restart a crashed/degraded service
  queue drain <N>                      - Drain N messages from the queue
  sqlite3 /data/db/main.db "<SQL>"     - Query the database

SERVICES: payment(8001), auth(8002), worker(8003), frontend(8004), cache(8005), notification(8006)

SRE WORKFLOW: triage (status) -> investigate (logs) -> fix (restart/drain) -> verify (status)
Output ONLY the next command. No explanations."""

def run_expert_episode(tier):
    """Run one expert episode and return the chat messages if resolved."""
    r = client.post("/reset", json={"task_id": tier})
    data = r.json()
    obs = data.get("observation", data)
    
    health = obs.get("service_health", {})
    alert = obs.get("alert", "")
    max_steps = obs.get("max_steps", 10)
    scenario_id = obs.get("scenario_id", "")
    
    broken = [(s, h.get("status"), h.get("error", "")) for s, h in health.items() if h.get("status") != "healthy"]
    if not broken:
        return None, scenario_id
    
    target_svc = broken[0][0]
    target_port = PORT_MAP.get(target_svc, 8001)
    error_text = broken[0][2].lower()
    
    # Build expert strategy based on fault
    commands = []
    # Step 1: Always triage
    commands.append("status")
    # Step 2: Investigate 
    commands.append(f"cat /var/log/{target_svc}/error.log")
    # Step 3-N: Fix
    if "queue" in error_text:
        commands.extend(["queue drain 200", "queue drain 200", "queue drain 200"])
    else:
        commands.append(f"restart_service {target_svc}")
    # Final: Verify
    commands.append("status")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    health_text = "\n".join(f"  {s}: {h.get('status','')} {'| '+str(h.get('error',''))[:80] if h.get('error') else ''}" for s, h in health.items())
    messages.append({"role": "user", "content": f"ALERT: {alert}\n\nHEALTH:\n{health_text}\n\nStep 1/{max_steps}. Next command:"})
    
    resolved = False
    final_step = 0
    
    for i, cmd in enumerate(commands):
        messages.append({"role": "assistant", "content": cmd})
        
        try:
            r = client.post("/step", json={"action": {"command": cmd}})
            d = r.json()
            step_obs = d.get("observation", d)
        except:
            break
        
        done = d.get("done", step_obs.get("done", False))
        cmd_output = step_obs.get("command_output", "")[:300]
        feedback = step_obs.get("feedback", "")
        step_health = step_obs.get("service_health", {})
        cascade = step_obs.get("cascade_triggered", False)
        
        final_step = i + 1
        
        if done:
            resolved = True
            break
        
        # If cascade triggered, fix the cascade target too
        if cascade:
            cascade_broken = [s for s, h in step_health.items() if h.get("status") != "healthy"]
            if cascade_broken:
                cascade_svc = cascade_broken[0]
                # Add cascade observation
                ch_text = "\n".join(f"  {s}: {h.get('status','')} {'| '+str(h.get('error',''))[:80] if h.get('error') else ''}" for s, h in step_health.items())
                cascade_alert = step_obs.get("cascade_alert", "")
                messages.append({"role": "user", "content": f"OUTPUT: {cmd_output[:200]}\n{feedback}\n\nCASCADE: {cascade_alert}\n\nHEALTH:\n{ch_text}\n\nStep {i+3}/{max_steps}. Next command:"})
                
                fix_cmd = f"restart_service {cascade_svc}"
                messages.append({"role": "assistant", "content": fix_cmd})
                
                try:
                    r2 = client.post("/step", json={"action": {"command": fix_cmd}})
                    d2 = r2.json()
                    if d2.get("done", d2.get("observation", d2).get("done", False)):
                        resolved = True
                        final_step = i + 2
                        break
                except:
                    pass
        
        # Build next observation for user
        if i < len(commands) - 1:
            sh_text = "\n".join(f"  {s}: {h.get('status','')} {'| '+str(h.get('error',''))[:80] if h.get('error') else ''}" for s, h in step_health.items())
            messages.append({"role": "user", "content": f"OUTPUT: {cmd_output[:200]}\n{feedback}\n\nHEALTH:\n{sh_text}\n\nStep {i+2}/{max_steps}. Next command:"})
    
    if resolved:
        return {"messages": messages, "scenario_id": scenario_id, "tier": tier, "steps": final_step}, scenario_id
    return None, scenario_id

# Generate episodes
sft_examples = []
seen_scenarios = set()
TIERS = ["warmup", "warmup", "warmup", "single_fault", "single_fault", "cascade", "multi_cascade", "adversarial"]
TARGET = 60

attempts = 0
while len(sft_examples) < TARGET and attempts < 200:
    tier = TIERS[attempts % len(TIERS)]
    example, scenario_id = run_expert_episode(tier)
    attempts += 1
    
    if example:
        sft_examples.append(example)
        is_new = scenario_id not in seen_scenarios
        seen_scenarios.add(scenario_id)
        print(f"  [{len(sft_examples):2d}/{TARGET}] {tier:15s} | {scenario_id:40s} | {example['steps']} steps {'(NEW)' if is_new else ''}")
    else:
        print(f"  [skip] {tier:15s} | {scenario_id:40s} | not resolved")
    
    time.sleep(0.2)

# Save
with open("sft_training_data.jsonl", "w", encoding="utf-8") as f:
    for ex in sft_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n{'='*60}")
print(f"SFT DATA: {len(sft_examples)} examples across {len(seen_scenarios)} unique scenarios")
print(f"Tiers: {dict((t, sum(1 for e in sft_examples if e['tier']==t)) for t in set(e['tier'] for e in sft_examples))}")
print(f"{'='*60}")

client.close()
