"""Verify that services run as SEPARATE OS PROCESSES with unique PIDs."""
import sys, time
sys.path.insert(0, "D:/Meta")

from services.orchestrator import ServiceOrchestrator

o = ServiceOrchestrator(
    db_path="D:/Meta/cloud_sre_v2/_test_data/app.db",
    log_dir="D:/Meta/cloud_sre_v2/_test_data/logs",
)
o.start()

print("=" * 60)
print("PROCESS ISOLATION PROOF")
print("=" * 60)

pids = set()
for name, entry in o._processes.items():
    pid = entry["pid"]
    port = entry["port"]
    alive = entry["proc"].poll() is None
    pids.add(pid)
    print(f"  {name:10s}  PID={pid:>6d}  port={port}  alive={alive}")

print(f"\nUnique PIDs: {len(pids)}")
print(f"All different: {len(pids) == 4}")

# Now kill one and show it's really dead
print("\n--- Killing worker (PID={}) ---".format(o._processes["worker"]["pid"]))
o._stop_service("worker")
time.sleep(0.5)

# Try to curl the dead service
import httpx
try:
    r = httpx.get("http://localhost:8003/healthz", timeout=2)
    print(f"Worker healthz: {r.status_code} (should have failed!)")
except httpx.ConnectError:
    print("✅ Worker port 8003: Connection refused (process is DEAD)")

# Restart and show new PID
old_pid = o._processes["worker"]["pid"]
o.restart_service("worker")
new_pid = o._processes["worker"]["pid"]
print(f"\n--- Restarted worker: PID {old_pid} → {new_pid} ---")
print(f"PID changed: {old_pid != new_pid}")

time.sleep(0.5)
try:
    r = httpx.get("http://localhost:8003/healthz", timeout=2)
    print(f"✅ Worker healthz: {r.status_code} (alive with new PID)")
except httpx.ConnectError:
    print("❌ Worker still dead")

o.shutdown()
