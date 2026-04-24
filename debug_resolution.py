"""Debug: hit warmup, fix, then inspect health in detail."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

# Reset
r = client.post("/reset", json={"task_id": "warmup"})
data = r.json()
obs = data.get("observation", data)
scenario = obs.get("scenario_id", "?")
health = obs.get("service_health", {})

print(f"Scenario: {scenario}")
for svc, h in health.items():
    print(f"  {svc}: {h.get('status')} | {h.get('error', '')}")

broken = [s for s, h in health.items() if h.get("status") != "healthy"]
print(f"\nBroken: {broken}")

if not broken:
    print("No broken services! Exiting.")
    client.close()
    exit()

# Step 1: diagnostic
r2 = client.post("/step", json={"action": {"command": f"cat /var/log/{broken[0]}/error.log"}})
d2 = r2.json()
print(f"\nStep 1 (logs): done={d2.get('done')}, reward={d2.get('reward')}")

# Step 2: fix
fix_cmd = f"restart_service {broken[0]}"
r3 = client.post("/step", json={"action": {"command": fix_cmd}})
d3 = r3.json()
o3 = d3.get("observation", d3)
print(f"\nStep 2 ({fix_cmd}): done={d3.get('done')}, reward={d3.get('reward')}")
print(f"  Output: {o3.get('command_output', '')[:200]}")
print(f"  Feedback: {o3.get('feedback', '')}")

# Show EVERY service health after fix
h3 = o3.get("service_health", {})
print(f"\nHealth after fix:")
for svc, h in h3.items():
    print(f"  {svc}: {h.get('status')} | error={h.get('error')} | degraded={h.get('degraded')}")

still_broken = [s for s, h in h3.items() if h.get("status") != "healthy"]
print(f"\nStill broken after fix: {still_broken}")

# Step 3: verify
r4 = client.post("/step", json={"action": {"command": "status"}})
d4 = r4.json()
o4 = d4.get("observation", d4)
print(f"\nStep 3 (status): done={d4.get('done')}, reward={d4.get('reward')}")
print(f"  Feedback: {o4.get('feedback', '')}")

h4 = o4.get("service_health", {})
print(f"\nHealth after status:")
for svc, h in h4.items():
    print(f"  {svc}: {h.get('status')} | error={h.get('error')}")

still_broken2 = [s for s, h in h4.items() if h.get("status") != "healthy"]
print(f"\nStill broken after verify: {still_broken2}")

# Step 4: one more status
r5 = client.post("/step", json={"action": {"command": "status"}})
d5 = r5.json()
o5 = d5.get("observation", d5)
print(f"\nStep 4 (status): done={d5.get('done')}, reward={d5.get('reward')}")
h5 = o5.get("service_health", {})
for svc, h in h5.items():
    if h.get("status") != "healthy":
        print(f"  STILL BROKEN: {svc}: {h}")

client.close()
