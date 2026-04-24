"""E2E: verify singleton fix — state persists across reset/step/step."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx, time

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

print("Waiting 200s for HF Space rebuild...")
time.sleep(200)

# Reset
print("\n=== RESET ===")
r = client.post("/reset", json={"task_id": "warmup"})
data = r.json()
obs = data.get("observation", data)
print(f"Scenario: {obs.get('scenario_id')}")

broken = [s for s, h in obs.get("service_health", {}).items() if h.get("status") != "healthy"]
print(f"Broken: {broken}")
if not broken:
    print("No broken services! Try another scenario...")
    client.close()
    exit()

# Step 1: diagnostic
print("\n=== STEP 1: diagnostic ===")
r2 = client.post("/step", json={"action": {"command": f"cat /var/log/{broken[0]}/error.log"}})
d2 = r2.json()
o2 = d2.get("observation", d2)
dbg = o2.get("debug_info", {})
print(f"done={d2.get('done')} | reward={d2.get('reward')}")
print(f"debug_info: step_count={dbg.get('step_count')} | history={dbg.get('history_cmd_types')}")

# Step 2: fix
fix = f"restart_service {broken[0]}"
print(f"\n=== STEP 2: fix ({fix}) ===")
r3 = client.post("/step", json={"action": {"command": fix}})
d3 = r3.json()
o3 = d3.get("observation", d3)
dbg = o3.get("debug_info", {})
print(f"done={d3.get('done')} | reward={d3.get('reward')}")
print(f"debug_info: step_count={dbg.get('step_count')} | history={dbg.get('history_cmd_types')}")
print(f"  all_healthy={dbg.get('all_healthy')} | has_fix={dbg.get('has_attempted_fix')} | min_steps={dbg.get('min_steps_met')}")

if d3.get("done") or (d3.get("observation", d3).get("done")):
    print("\n🎉🎉🎉 EPISODE RESOLVED AT STEP 2! 🎉🎉🎉")
    print(f"Feedback: {o3.get('feedback')}")
    client.close()
    exit()

# Step 3: verify
print(f"\n=== STEP 3: verify (status) ===")
r4 = client.post("/step", json={"action": {"command": "status"}})
d4 = r4.json()
o4 = d4.get("observation", d4)
dbg = o4.get("debug_info", {})
print(f"done={d4.get('done')} | reward={d4.get('reward')}")
print(f"debug_info: step_count={dbg.get('step_count')} | history={dbg.get('history_cmd_types')}")
print(f"  all_healthy={dbg.get('all_healthy')} | has_fix={dbg.get('has_attempted_fix')} | min_steps={dbg.get('min_steps_met')}")
print(f"  feedback={o4.get('feedback')}")

if d4.get("done") or o4.get("done"):
    print("\n🎉🎉🎉 EPISODE RESOLVED AT STEP 3! 🎉🎉🎉")
else:
    print("\n❌ Still not resolved. Checking what's wrong...")
    health = o4.get("service_health", {})
    for svc, h in health.items():
        if h.get("status") != "healthy":
            print(f"  BROKEN: {svc}: {h}")

client.close()
