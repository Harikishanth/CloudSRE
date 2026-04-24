"""
COMPREHENSIVE SMOKE TEST — Every scenario, every tier, no exceptions.

Tests each of the 21 static scenarios + 5 adversarial episodes.
Uses a flexible expert strategy that adapts to each fault type.
Reports PASS/FAIL with exact diagnostics for every single scenario.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx, time, json

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

PORT_MAP = {"payment": 8001, "auth": 8002, "worker": 8003, "frontend": 8004, "cache": 8005, "notification": 8006}

# ALL 21 static scenarios
ALL_SCENARIOS = {
    "warmup": [
        "warmup_db_lock", "warmup_process_crash", "warmup_queue_overflow",
        "warmup_auth_crash", "warmup_cache_cold", "warmup_notification_timeout",
    ],
    "single_fault": [
        "single_db_lock_redherring", "single_worker_crash_redherring",
        "single_queue_overflow", "single_payment_crash_db_redherring",
    ],
    "cascade": [
        "cascade_db_thundering_herd", "cascade_worker_crash_queue_flood",
        "cascade_db_lock_to_payment_crash", "cascade_queue_overflow_worker_crash",
        "cascade_auth_crash_payment_timeout", "cascade_db_cache_stampede",
        "cascade_payment_webhook_storm",
    ],
    "multi_cascade": [
        "multi_cascade_full_outage", "multi_cascade_payment_worker_down",
        "multi_cascade_db_queue_auth", "multi_cascade_rolling_crash",
    ],
}


def get_specific_scenario(tier, target_scenario, max_retries=25):
    """Keep resetting until we get the exact scenario we want."""
    for _ in range(max_retries):
        r = client.post("/reset", json={"task_id": tier})
        data = r.json()
        obs = data.get("observation", data)
        if obs.get("scenario_id") == target_scenario:
            return obs
    return None


def find_broken(health):
    """Return list of (service, status, error) for unhealthy services."""
    return [(s, h.get("status",""), h.get("error","")) for s, h in health.items() if h.get("status") != "healthy"]


def run_scenario(tier, scenario_id):
    """Run a full expert strategy against one specific scenario. Returns (resolved, steps, details)."""
    obs = get_specific_scenario(tier, scenario_id)
    if obs is None:
        return False, 0, f"Could not select scenario after retries"

    health = obs.get("service_health", {})
    max_steps = obs.get("max_steps", 10)
    broken = find_broken(health)

    if not broken:
        return False, 0, f"BUG: No faults visible after reset"

    # Flexible expert strategy: diagnose → fix all broken → handle cascades → verify
    steps_taken = 0
    resolved = False
    details = []

    # Step 1: status (triage)
    r = client.post("/step", json={"action": {"command": "status"}})
    d = r.json()
    steps_taken += 1
    if d.get("done", d.get("observation",d).get("done", False)):
        return True, steps_taken, "Resolved at triage"

    # Now iteratively fix broken services
    for attempt in range(max_steps - 1):
        obs = d.get("observation", d)
        health = obs.get("service_health", {})
        broken = find_broken(health)
        cascade = obs.get("cascade_triggered", False)

        if not broken:
            # All healthy — run status to trigger resolution check
            r = client.post("/step", json={"action": {"command": "status"}})
            d = r.json()
            steps_taken += 1
            if d.get("done", d.get("observation",d).get("done", False)):
                return True, steps_taken, f"Resolved after fixing all services"
            # If still not done, try one more status
            continue

        svc, status, error = broken[0]
        error_lower = error.lower()

        # Choose fix based on fault type
        if "queue" in error_lower and ("overflow" in error_lower or "depth" in error_lower or "overwhelmed" in error_lower):
            cmd = "queue drain 200"
        elif "thundering" in error_lower or "oom" in error_lower:
            cmd = f"restart_service {svc}"
        else:
            cmd = f"restart_service {svc}"

        details.append(f"Step {steps_taken+1}: {cmd} (fixing {svc}: {status})")

        r = client.post("/step", json={"action": {"command": cmd}})
        d = r.json()
        steps_taken += 1

        if d.get("done", d.get("observation",d).get("done", False)):
            return True, steps_taken, " | ".join(details)

        if steps_taken >= max_steps:
            break

    # Final check
    obs = d.get("observation", d)
    still_broken = find_broken(obs.get("service_health", {}))
    if still_broken:
        return False, steps_taken, f"Still broken: {[(s,st) for s,st,_ in still_broken]} | " + " | ".join(details)
    else:
        return False, steps_taken, f"All healthy but done=False after {steps_taken} steps | " + " | ".join(details)


# ═══════════════════════════════════════════════════════════════
# RUN ALL SCENARIOS
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("COMPREHENSIVE SMOKE TEST — ALL SCENARIOS, ALL TIERS")
print("=" * 80)

all_results = {}
total_pass = 0
total_fail = 0
tier_results = {}

for tier, scenarios in ALL_SCENARIOS.items():
    tier_pass = 0
    tier_fail = 0
    tier_results[tier] = {}

    print(f"\n{'─'*80}")
    print(f"TIER: {tier} ({len(scenarios)} scenarios)")
    print(f"{'─'*80}")

    for scenario_id in scenarios:
        resolved, steps, details = run_scenario(tier, scenario_id)

        if resolved:
            emoji = "✅"
            total_pass += 1
            tier_pass += 1
        else:
            emoji = "❌"
            total_fail += 1
            tier_fail += 1

        tier_results[tier][scenario_id] = {"resolved": resolved, "steps": steps, "details": details}
        print(f"  {emoji} {scenario_id:45s} | {steps:2d} steps | {details[:60]}")

    print(f"  ── {tier}: {tier_pass}/{len(scenarios)} passed")

# Adversarial (dynamic — run 5 episodes)
print(f"\n{'─'*80}")
print(f"TIER: adversarial (5 dynamic episodes)")
print(f"{'─'*80}")

adv_pass = 0
adv_fail = 0
for i in range(5):
    r = client.post("/reset", json={"task_id": "adversarial"})
    data = r.json()
    obs = data.get("observation", data)
    scenario_id = obs.get("scenario_id", f"adversarial_{i}")
    health = obs.get("service_health", {})
    broken = find_broken(health)

    if not broken:
        print(f"  ❌ {scenario_id:45s} | 0 steps | No faults visible")
        adv_fail += 1
        total_fail += 1
        continue

    # Run flexible expert
    max_steps = obs.get("max_steps", 30)
    steps = 0
    resolved = False
    d = data

    for attempt in range(min(max_steps, 10)):
        obs_inner = d.get("observation", d) if steps > 0 else obs
        h = obs_inner.get("service_health", {})
        b = find_broken(h)

        if not b:
            r2 = client.post("/step", json={"action": {"command": "status"}})
            d = r2.json()
            steps += 1
            if d.get("done", d.get("observation",d).get("done", False)):
                resolved = True
                break
            continue

        svc, st, err = b[0]
        if "queue" in err.lower():
            cmd = "queue drain 200"
        else:
            cmd = f"restart_service {svc}"

        r2 = client.post("/step", json={"action": {"command": cmd}})
        d = r2.json()
        steps += 1
        if d.get("done", d.get("observation",d).get("done", False)):
            resolved = True
            break

    emoji = "✅" if resolved else "❌"
    if resolved:
        adv_pass += 1
        total_pass += 1
    else:
        adv_fail += 1
        total_fail += 1
    print(f"  {emoji} {scenario_id:45s} | {steps:2d} steps | {'RESOLVED' if resolved else 'FAILED'}")

print(f"  ── adversarial: {adv_pass}/5 passed")

# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════

total = total_pass + total_fail
print(f"\n{'='*80}")
print(f"FINAL REPORT: {total_pass}/{total} scenarios passed ({total_pass/total*100:.0f}%)")
print(f"{'='*80}")

for tier in list(ALL_SCENARIOS.keys()) + ["adversarial"]:
    if tier == "adversarial":
        print(f"  adversarial:   {adv_pass}/5")
    else:
        results = tier_results[tier]
        passed = sum(1 for r in results.values() if r["resolved"])
        print(f"  {tier:15s}: {passed}/{len(results)}")

# List failures
failures = []
for tier, results in tier_results.items():
    for sid, r in results.items():
        if not r["resolved"]:
            failures.append((tier, sid, r["details"]))

if failures:
    print(f"\n{'─'*80}")
    print(f"FAILURES ({len(failures)}):")
    print(f"{'─'*80}")
    for tier, sid, details in failures:
        print(f"  {tier:15s} | {sid:45s}")
        print(f"    → {details}")

client.close()
