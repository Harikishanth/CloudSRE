"""Smoke test: Reset all 5 tiers, verify faults visible, try one fix, check resolution."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx, time

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

TIERS = ["warmup", "single_fault", "cascade", "multi_cascade", "adversarial"]

print("=" * 70)
print("SMOKE TEST — ALL 5 TIERS")
print("=" * 70)

results = {}

for tier in TIERS:
    print(f"\n{'─' * 70}")
    print(f"TIER: {tier}")
    print(f"{'─' * 70}")

    # ── RESET ──
    try:
        r = client.post("/reset", json={"task_id": tier})
        if r.status_code != 200:
            print(f"  RESET HTTP {r.status_code}: {r.text[:200]}")
            results[tier] = f"RESET_HTTP_{r.status_code}"
            continue
        data = r.json()
    except Exception as e:
        print(f"  RESET FAILED: {type(e).__name__}: {e}")
        results[tier] = "RESET_EXCEPTION"
        continue

    obs = data.get("observation", data)
    scenario = obs.get("scenario_id", "?")
    max_steps = obs.get("max_steps", "?")
    health = obs.get("service_health", {})
    cascade_triggered = obs.get("cascade_triggered", False)

    print(f"  Scenario:  {scenario}")
    print(f"  Max steps: {max_steps}")

    # Show all service health
    broken = []
    for svc, h in health.items():
        status = h.get("status", "?")
        error = h.get("error", "")
        mark = "✓" if status == "healthy" else "✗"
        print(f"  {mark} {svc:15s} = {status:10s} {('| ' + str(error)[:60]) if error else ''}")
        if status != "healthy":
            broken.append((svc, status, error))

    if not broken:
        print(f"  ⚠ BUG: No faults visible after reset!")
        results[tier] = "NO_FAULT"
        continue

    # ── STEP 1: Diagnostic ──
    try:
        r2 = client.post("/step", json={"action": {"command": f"cat /var/log/{broken[0][0]}/error.log"}})
        d2 = r2.json()
        step1_done = d2.get("done", False)
        step1_reward = float(d2.get("reward", 0))
        print(f"  Step 1 (logs): done={step1_done}, reward={step1_reward:.3f}")
    except Exception as e:
        print(f"  Step 1 FAILED: {type(e).__name__}")
        results[tier] = "STEP1_FAIL"
        continue

    # ── STEP 2: Fix (restart or drain based on error) ──
    first_broken = broken[0]
    if "queue" in str(first_broken[2]).lower():
        fix_cmd = "queue drain 50"
    else:
        fix_cmd = f"restart_service {first_broken[0]}"

    try:
        r3 = client.post("/step", json={"action": {"command": fix_cmd}})
        d3 = r3.json()
        o3 = d3.get("observation", d3)
        step2_done = d3.get("done", False)
        step2_reward = float(d3.get("reward", 0))
        step2_cascade = o3.get("cascade_triggered", False)
        step2_health = o3.get("service_health", {})
        step2_broken = [s for s, h in step2_health.items() if h.get("status") != "healthy"]

        print(f"  Step 2 ({fix_cmd}): done={step2_done}, reward={step2_reward:.3f}, cascade={step2_cascade}")
        if step2_broken:
            print(f"  Still broken: {step2_broken}")
    except Exception as e:
        print(f"  Step 2 FAILED: {type(e).__name__}")
        results[tier] = "STEP2_FAIL"
        continue

    # ── STEP 3: Verify / Fix cascade if triggered ──
    if step2_cascade and step2_broken:
        # Cascade triggered — fix the cascaded service
        cascade_svc = step2_broken[0]
        cascade_fix = f"restart_service {cascade_svc}"
        try:
            r4 = client.post("/step", json={"action": {"command": cascade_fix}})
            d4 = r4.json()
            step3_done = d4.get("done", False)
            step3_reward = float(d4.get("reward", 0))
            print(f"  Step 3 ({cascade_fix}): done={step3_done}, reward={step3_reward:.3f}")

            if not step3_done:
                # One more verify
                r5 = client.post("/step", json={"action": {"command": "status"}})
                d5 = r5.json()
                step4_done = d5.get("done", False)
                print(f"  Step 4 (status): done={step4_done}, reward={float(d5.get('reward', 0)):.3f}")
                results[tier] = "RESOLVED" if step4_done else "NOT_RESOLVED_4_STEPS"
            else:
                results[tier] = "RESOLVED"
        except Exception as e:
            print(f"  Step 3 FAILED: {type(e).__name__}")
            results[tier] = "STEP3_FAIL"
    elif step2_done:
        results[tier] = "RESOLVED"
    else:
        # Try verify
        try:
            r4 = client.post("/step", json={"action": {"command": "status"}})
            d4 = r4.json()
            step3_done = d4.get("done", False)
            step3_reward = float(d4.get("reward", 0))
            print(f"  Step 3 (status): done={step3_done}, reward={step3_reward:.3f}")
            results[tier] = "RESOLVED" if step3_done else "NOT_RESOLVED"
        except Exception as e:
            print(f"  Step 3 FAILED: {type(e).__name__}")
            results[tier] = "STEP3_FAIL"

    time.sleep(1)  # Small gap between tiers

# ── SUMMARY ──
print(f"\n{'=' * 70}")
print("SMOKE TEST RESULTS")
print(f"{'=' * 70}")
for tier in TIERS:
    status = results.get(tier, "UNKNOWN")
    emoji = "✅" if "RESOLVED" in status else "❌"
    print(f"  {emoji} {tier:20s}: {status}")

total_pass = sum(1 for v in results.values() if "RESOLVED" in v)
print(f"\n  {total_pass}/{len(TIERS)} tiers passed")

client.close()
