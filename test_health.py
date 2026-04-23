"""Check the actual HF Space state: was the code REALLY updated?"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx

base = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=base, timeout=30)

# Check if the /state endpoint reveals anything
try:
    r = client.get("/state")
    print("=== /state ===")
    print(r.json())
except:
    print("/state not available")

# Check if health endpoint exists
try:
    r = client.get("/health")
    print("\n=== /health ===")
    print(r.text[:500])
except:
    print("/health not available")

# Do a reset then immediately check crashed_services
print("\n=== RESET with warmup ===")
r = client.post("/reset", json={"task_id": "warmup"})
data = r.json()
obs = data.get("observation", data)
print(f"Scenario: {obs.get('scenario_id')}")
health = obs.get("service_health", {})
for svc, h in health.items():
    print(f"  {svc}: {h.get('status')}")

# Do a step with harmless command
print("\n=== STEP: ps aux ===")
r2 = client.post("/step", json={"action": {"command": "ps aux"}})
d2 = r2.json()
o2 = d2.get("observation", d2)
print(f"done: {d2.get('done')}")
print(f"reward: {d2.get('reward')}")
print(f"output (first 300): {o2.get('command_output', '')[:300]}")

# Check if the output contains our drain rate change
print("\n=== STEP: queue drain ===")
r3 = client.post("/step", json={"action": {"command": "queue drain"}})
d3 = r3.json()
o3 = d3.get("observation", d3)
print(f"output: {o3.get('command_output', '')[:200]}")
# If output says "rate=50" our code is deployed. If "rate=10" it's old code.

client.close()
