"""Debug: call status command and print raw output."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx

BASE = "https://dardrax-cloudsre-environment.hf.space"
client = httpx.Client(base_url=BASE, timeout=120)

# Reset to a known scenario
r = client.post("/reset", json={"task_id": "warmup"})
data = r.json()
obs = data.get("observation", data)
print(f"Scenario: {obs.get('scenario_id')}")

broken = [s for s, h in obs.get("service_health", {}).items() if h.get("status") != "healthy"]
print(f"Broken: {broken}")

# Fix it
if broken:
    if "queue" in str(obs.get("service_health", {}).get(broken[0], {}).get("error", "")).lower():
        fix = "queue drain 50"
    else:
        fix = f"restart_service {broken[0]}"
    
    r2 = client.post("/step", json={"action": {"command": fix}})
    d2 = r2.json()
    print(f"\nFix ({fix}): done={d2.get('done')}, reward={d2.get('reward')}")

# Now call status and print the RAW output
r3 = client.post("/step", json={"action": {"command": "status"}})
d3 = r3.json()
o3 = d3.get("observation", d3)
print(f"\nStatus: done={d3.get('done')}, reward={d3.get('reward')}")
print(f"Feedback: {o3.get('feedback')}")
print(f"\n--- RAW COMMAND OUTPUT ---")
print(o3.get("command_output", "NO OUTPUT"))
print(f"--- END ---")

# Check for "error" patterns in the output
output = o3.get("command_output", "")
for pattern in ["Error:", "error:", "refused"]:
    if pattern in output:
        print(f"\n⚠️ FOUND '{pattern}' in output!")
        # Find context
        idx = output.index(pattern)
        start = max(0, idx - 50)
        end = min(len(output), idx + 50)
        print(f"  Context: ...{output[start:end]}...")

client.close()
