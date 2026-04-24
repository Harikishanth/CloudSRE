"""Test EVERY warmup scenario — verify each one can actually resolve."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx, time, json

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

WARMUP_SCENARIOS = [
    "warmup_db_lock",
    "warmup_process_crash",
    "warmup_queue_overflow",
    "warmup_auth_crash",
    "warmup_cache_cold",
    "warmup_notification_timeout",
]

results = {}

for scenario_name in WARMUP_SCENARIOS:
    print(f"\n{'='*60}")
    print(f"TESTING: {scenario_name}")
    print(f"{'='*60}")

    # Reset — keep resetting until we get this specific scenario
    # (scenarios are randomly selected, so we may need to retry)
    got_scenario = False
    for attempt in range(15):
        r = client.post("/reset", json={"task_id": "warmup"})
        data = r.json()
        obs = data.get("observation", data)
        sid = obs.get("scenario_id", "")
        if sid == scenario_name:
            got_scenario = True
            break

    if not got_scenario:
        print(f"  Could not get scenario after 15 resets (last was {sid})")
        results[scenario_name] = "COULD_NOT_SELECT"
        continue

    health = obs.get("service_health", {})
    broken = [(s, h.get("status"), h.get("error", "")) for s, h in health.items() if h.get("status") != "healthy"]
    print(f"  Broken: {broken}")

    if not broken:
        print(f"  BUG: No faults visible!")
        results[scenario_name] = "NO_FAULT"
        continue

    svc, status, error = broken[0]

    # Step 1: Read logs
    r1 = client.post("/step", json={"action": {"command": f"cat /var/log/{svc}/error.log"}})
    d1 = r1.json()

    # Step 2: Apply fix based on fault type
    if "queue" in error.lower():
        fix = "queue drain 50"
    elif "database" in error.lower() or "db" in error.lower() or "locked" in error.lower():
        # DB lock — restart might not fix it, but let's try
        fix = f"restart_service {svc}"
    elif "cache" in error.lower() or "invalidat" in error.lower():
        fix = f"restart_service {svc}"
    else:
        fix = f"restart_service {svc}"

    r2 = client.post("/step", json={"action": {"command": fix}})
    d2 = r2.json()
    o2 = d2.get("observation", d2)
    done2 = d2.get("done", o2.get("done", False))
    reward2 = d2.get("reward", o2.get("reward", 0))
    dbg2 = o2.get("debug_info", {})

    print(f"  Fix ({fix}): done={done2}, reward={reward2:.3f}")
    print(f"    all_healthy={dbg2.get('all_healthy')} | has_fix={dbg2.get('has_attempted_fix')}")

    if done2:
        print(f"  RESOLVED in 2 steps!")
        results[scenario_name] = f"RESOLVED_2_steps (reward={reward2:.2f})"
        continue

    # Still broken? Check what's still wrong
    h2 = o2.get("service_health", {})
    still_broken = [(s, h.get("status"), h.get("error", "")) for s, h in h2.items() if h.get("status") != "healthy"]
    print(f"  Still broken after fix: {still_broken}")

    # Step 3: Try a different fix or drain more
    if still_broken:
        svc2, st2, err2 = still_broken[0]
        if "queue" in err2.lower():
            fix2 = "queue drain 50"
        else:
            fix2 = f"restart_service {svc2}"

        r3 = client.post("/step", json={"action": {"command": fix2}})
        d3 = r3.json()
        o3 = d3.get("observation", d3)
        done3 = d3.get("done", o3.get("done", False))
        print(f"  Fix2 ({fix2}): done={done3}")

        if done3:
            results[scenario_name] = f"RESOLVED_3_steps"
            continue

        # Step 4: verify
        r4 = client.post("/step", json={"action": {"command": "status"}})
        d4 = r4.json()
        done4 = d4.get("done", d4.get("observation", d4).get("done", False))
        print(f"  Verify: done={done4}")

        if done4:
            results[scenario_name] = f"RESOLVED_4_steps"
        else:
            # Try drain more for queue scenarios
            for extra in range(5):
                r5 = client.post("/step", json={"action": {"command": "queue drain 50"}})
                d5 = r5.json()
                done5 = d5.get("done", d5.get("observation", d5).get("done", False))
                if done5:
                    results[scenario_name] = f"RESOLVED_{5+extra}_steps"
                    break
            else:
                results[scenario_name] = "FAILED"
    else:
        # All healthy but not done — check status
        r3 = client.post("/step", json={"action": {"command": "status"}})
        d3 = r3.json()
        done3 = d3.get("done", d3.get("observation", d3).get("done", False))
        if done3:
            results[scenario_name] = f"RESOLVED_3_steps"
        else:
            results[scenario_name] = "FAILED_all_healthy_but_not_done"

    time.sleep(0.5)

# Summary
print(f"\n{'='*60}")
print("ALL WARMUP SCENARIOS RESULTS")
print(f"{'='*60}")
for name, result in results.items():
    emoji = "✅" if "RESOLVED" in result else "❌"
    print(f"  {emoji} {name:35s}: {result}")

passed = sum(1 for v in results.values() if "RESOLVED" in v)
print(f"\n  {passed}/{len(WARMUP_SCENARIOS)} scenarios passed")

client.close()
