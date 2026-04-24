"""Debug: check debug_info from API response to trace resolution failure."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx, time

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

print("Waiting 180s for HF Space rebuild...")
time.sleep(180)

# Reset
r = client.post("/reset", json={"task_id": "warmup"})
data = r.json()
obs = data.get("observation", data)
print(f"Scenario: {obs.get('scenario_id')}")

broken = [s for s, h in obs.get("service_health", {}).items() if h.get("status") != "healthy"]
print(f"Broken: {broken}")
if not broken:
    print("No broken services! Exiting.")
    client.close()
    exit()

# Step 1: diagnostic
r2 = client.post("/step", json={"action": {"command": f"cat /var/log/{broken[0]}/error.log"}})
d2 = r2.json()
print(f"\nStep 1 (logs): done={d2.get('done')}")
dbg1 = d2.get("observation", d2).get("debug_info", d2.get("debug_info", "NOT_FOUND"))
print(f"  debug_info: {dbg1}")

# Step 2: fix
fix = f"restart_service {broken[0]}"
r3 = client.post("/step", json={"action": {"command": fix}})
d3 = r3.json()
print(f"\nStep 2 ({fix}): done={d3.get('done')}")
dbg2 = d3.get("observation", d3).get("debug_info", d3.get("debug_info", "NOT_FOUND"))
print(f"  debug_info: {dbg2}")

# Step 3: verify
r4 = client.post("/step", json={"action": {"command": "status"}})
d4 = r4.json()
print(f"\nStep 3 (status): done={d4.get('done')}")
dbg3 = d4.get("observation", d4).get("debug_info", d4.get("debug_info", "NOT_FOUND"))
print(f"  debug_info: {dbg3}")

# Print full step 3 response for inspection
print(f"\nFull step 3 keys: {list(d4.keys())}")
obs3 = d4.get("observation", d4)
print(f"Observation keys: {list(obs3.keys())}")

client.close()
