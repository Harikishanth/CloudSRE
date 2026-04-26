"""
Generate 100+ BALANCED SFT demonstrations across all 5 tiers.
Target: 20 per tier for even distribution.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx, json, time

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

PORT_MAP = {
    "payment": 8001, "auth": 8002, "worker": 8003, "frontend": 8004,
    "cache": 8005, "notification": 8006, "search": 8007, "gateway": 8008,
    "scheduler": 8009, "storage": 8010, "metrics_collector": 8011,
    "email": 8012, "billing": 8013, "config": 8014, "dns": 8015,
    "loadbalancer": 8016,
}

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a production cloud platform.

AVAILABLE COMMANDS:
  status                                       - Check all service health
  curl http://localhost:<PORT>/healthz          - Check specific service health
  curl http://<svc>.<region>.internal/healthz   - Check via cloud DNS
  cat /var/log/<service>/error.log              - Read service error logs
  restart_service <service>                     - Restart a crashed/degraded service
  queue drain <N>                               - Drain N messages from the queue
  sqlite3 /data/app.db "<SQL>"                  - Query the database

SERVICES (16 total):
  us-east-1: payment(8001), auth(8002), gateway(8008), billing(8013), config(8014), loadbalancer(8016)
  eu-west-1: worker(8003), search(8007), scheduler(8009), storage(8010), metrics_collector(8011)
  ap-south-1: frontend(8004), cache(8005), notification(8006), email(8012), dns(8015)

SRE WORKFLOW: triage (status) -> investigate (logs/healthz) -> fix (restart/drain/config) -> verify (status)
Output ONLY the next command. No explanations."""


def find_broken(health):
    return [(s, h.get("status",""), h.get("error","")) for s, h in health.items() if h.get("status") != "healthy"]


def run_expert_episode(tier):
    """Run a full expert episode with flexible multi-step strategy."""
    r = client.post("/reset", json={"task_id": tier})
    data = r.json()
    obs = data.get("observation", data)
    
    health = obs.get("service_health", {})
    alert = obs.get("alert", "")
    max_steps = obs.get("max_steps", 10)
    scenario_id = obs.get("scenario_id", "")
    broken = find_broken(health)
    
    if not broken:
        return None, scenario_id
    
    # Build conversation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    health_text = "\n".join(
        f"  {s}: {h.get('status','')} {'| '+str(h.get('error',''))[:80] if h.get('error') else ''}"
        for s, h in health.items()
    )
    messages.append({"role": "user", "content": f"ALERT: {alert}\n\nHEALTH:\n{health_text}\n\nStep 1/{max_steps}. Next command:"})
    
    resolved = False
    final_step = 0
    d = data  # For tracking state
    
    # Step 1: Always start with status (triage)
    messages.append({"role": "assistant", "content": "status"})
    r2 = client.post("/step", json={"action": {"command": "status"}})
    d = r2.json()
    final_step = 1
    
    if d.get("done", d.get("observation", d).get("done", False)):
        return {"messages": messages, "scenario_id": scenario_id, "tier": tier, "steps": final_step}, scenario_id
    
    # Iteratively fix all broken services
    for step in range(max_steps - 1):
        obs = d.get("observation", d)
        health = obs.get("service_health", {})
        cmd_output = obs.get("command_output", "")[:300]
        feedback = obs.get("feedback", "")
        broken = find_broken(health)
        cascade = obs.get("cascade_triggered", False)
        
        if not broken:
            # All healthy - verify
            sh_text = "\n".join(f"  {s}: {h.get('status','')}" for s, h in health.items())
            messages.append({"role": "user", "content": f"OUTPUT: {cmd_output[:200]}\n{feedback}\nHEALTH:\n{sh_text}\n\nStep {final_step+1}/{max_steps}. Next command:"})
            messages.append({"role": "assistant", "content": "status"})
            r3 = client.post("/step", json={"action": {"command": "status"}})
            d = r3.json()
            final_step += 1
            if d.get("done", d.get("observation", d).get("done", False)):
                resolved = True
                break
            continue
        
        svc, status, error = broken[0]
        error_lower = error.lower()
        
        # Choose fix
        if "queue" in error_lower and ("overflow" in error_lower or "depth" in error_lower or "overwhelmed" in error_lower or "paused" in error_lower):
            cmd = "queue drain 200"
        else:
            cmd = f"restart_service {svc}"
        
        # Add observation + action
        sh_text = "\n".join(
            f"  {s}: {h.get('status','')} {'| '+str(h.get('error',''))[:60] if h.get('error') else ''}"
            for s, h in health.items()
        )
        cascade_text = f"\nCASCADE TRIGGERED" if cascade else ""
        messages.append({"role": "user", "content": f"OUTPUT: {cmd_output[:200]}\n{feedback}{cascade_text}\nHEALTH:\n{sh_text}\n\nStep {final_step+1}/{max_steps}. Next command:"})
        messages.append({"role": "assistant", "content": cmd})
        
        r3 = client.post("/step", json={"action": {"command": cmd}})
        d = r3.json()
        final_step += 1
        
        if d.get("done", d.get("observation", d).get("done", False)):
            resolved = True
            break
        
        if final_step >= max_steps:
            break
    
    if resolved:
        return {"messages": messages, "scenario_id": scenario_id, "tier": tier, "steps": final_step}, scenario_id
    return None, scenario_id


# ═══════════════════════════════════════════════════════════════
# Generate BALANCED data: 20 per tier
# ═══════════════════════════════════════════════════════════════

TARGET_PER_TIER = 20
TIERS = ["warmup", "single_fault", "cascade", "multi_cascade", "adversarial"]
tier_examples = {t: [] for t in TIERS}
tier_attempts = {t: 0 for t in TIERS}
MAX_ATTEMPTS_PER_TIER = 80

print("=" * 80)
print("BALANCED SFT DATA GENERATION — 20 per tier, 100 total")
print("=" * 80)

for tier in TIERS:
    print(f"\n--- TIER: {tier} (target: {TARGET_PER_TIER}) ---")
    
    while len(tier_examples[tier]) < TARGET_PER_TIER and tier_attempts[tier] < MAX_ATTEMPTS_PER_TIER:
        example, scenario_id = run_expert_episode(tier)
        tier_attempts[tier] += 1
        
        if example:
            tier_examples[tier].append(example)
            print(f"  [{len(tier_examples[tier]):2d}/{TARGET_PER_TIER}] {scenario_id:45s} | {example['steps']} steps")
        else:
            print(f"  [skip] {scenario_id:45s} | not resolved")
        
        time.sleep(0.2)
    
    print(f"  Result: {len(tier_examples[tier])}/{TARGET_PER_TIER} ({tier_attempts[tier]} attempts)")

# Combine all
all_examples = []
for tier in TIERS:
    all_examples.extend(tier_examples[tier])

# Save
with open("sft_training_data.jsonl", "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# Report
print(f"\n{'='*80}")
print(f"BALANCED SFT DATA COMPLETE: {len(all_examples)} examples")
print(f"{'='*80}")
scenarios_seen = set()
for tier in TIERS:
    tier_scenarios = set(e["scenario_id"] for e in tier_examples[tier])
    scenarios_seen.update(tier_scenarios)
    print(f"  {tier:15s}: {len(tier_examples[tier]):2d} examples | {len(tier_scenarios)} unique scenarios | {tier_attempts[tier]} attempts")
print(f"\n  Total unique scenarios: {len(scenarios_seen)}")

client.close()
