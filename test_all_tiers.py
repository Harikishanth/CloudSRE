"""E2E test: Hit live HF Space with all 5 tiers — with longer timeout."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx

base = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=base, timeout=120)  # 2 min timeout

TIERS = ["warmup", "single_fault", "cascade", "multi_cascade", "adversarial"]
results = {}

for tier in TIERS:
    print(f"\n{'='*60}")
    print(f"TIER: {tier}")
    print(f"{'='*60}")
    
    try:
        r = client.post("/reset", json={"task_id": tier})
        data = r.json()
    except Exception as e:
        print(f"  RESET FAILED: {e}")
        results[tier] = "RESET_FAILED"
        continue
    
    obs = data.get("observation", data)
    scenario = obs.get("scenario_id", "?")
    max_steps = obs.get("max_steps", "?")
    health = obs.get("service_health", {})
    
    print(f"  Scenario: {scenario}")
    print(f"  Max steps: {max_steps}")
    
    unhealthy = []
    for svc, h in health.items():
        status = h.get("status", "?")
        if status != "healthy":
            unhealthy.append(f"{svc}={status}")
    
    if unhealthy:
        print(f"  Faulted: {', '.join(unhealthy)}")
    else:
        print(f"  BUG: All healthy after reset!")
        results[tier] = "NO_FAULT_VISIBLE"
        continue
    
    # Step 1: diagnostic
    try:
        r2 = client.post("/step", json={"action": {"command": "cat /var/log/payment/error.log"}})
        d2 = r2.json()
        done1 = d2.get("done", False)
        print(f"  Step 1 (logs): done={done1}, reward={d2.get('reward', 0):.3f}")
        if done1:
            print(f"  BUG: Resolved on diagnostic!")
            results[tier] = "RESOLVED_ON_DIAG"
            continue
    except Exception as e:
        print(f"  Step 1 FAILED: {type(e).__name__}")
        results[tier] = "STEP_TIMEOUT"
        continue

    # Step 2: fix the first unhealthy service
    target = unhealthy[0].split("=")[0]
    fix_cmd = f"restart_service {target}"
    try:
        r3 = client.post("/step", json={"action": {"command": fix_cmd}})
        d3 = r3.json()
        o3 = d3.get("observation", d3)
        done2 = d3.get("done", False)
        cascade = o3.get("cascade_triggered", False)
        h3 = o3.get("service_health", {})
        unhealthy2 = [f"{s}={h.get('status')}" for s, h in h3.items() if h.get("status") != "healthy"]
        
        print(f"  Step 2 ({fix_cmd}): done={done2}, reward={d3.get('reward', 0):.3f}, cascade={cascade}")
        if unhealthy2:
            print(f"  Still unhealthy: {', '.join(unhealthy2)}")
    except Exception as e:
        print(f"  Step 2 FAILED: {type(e).__name__}")
        results[tier] = "STEP_TIMEOUT"
        continue

    # Step 3: verify
    try:
        r4 = client.post("/step", json={"action": {"command": "status"}})
        d4 = r4.json()
        done3 = d4.get("done", False)
        h4 = d4.get("observation", d4).get("service_health", {})
        unhealthy3 = [f"{s}={h.get('status')}" for s, h in h4.items() if h.get("status") != "healthy"]
        
        print(f"  Step 3 (status): done={done3}, reward={d4.get('reward', 0):.3f}")
        if unhealthy3:
            print(f"  Still unhealthy: {', '.join(unhealthy3)}")
        
        if done3:
            results[tier] = "RESOLVED_3_STEPS"
        elif unhealthy3:
            results[tier] = f"NEEDS_MORE_WORK ({', '.join(unhealthy3)})"
        else:
            results[tier] = "ALL_HEALTHY_NOT_DONE"
    except Exception as e:
        print(f"  Step 3 FAILED: {type(e).__name__}")
        results[tier] = "STEP_TIMEOUT"

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for tier, result in results.items():
    status_emoji = "?" if "FAIL" in result or "BUG" in result or "TIMEOUT" in result else "?"
    print(f"  {tier:20s}: {result}")

client.close()
