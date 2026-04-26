"""Replicate exact smoke test flow and capture payment stderr."""
import sys, os, time
sys.path.insert(0, "D:/Meta")
import shutil
shutil.rmtree("D:/Meta/cloud_sre_v2/_test_data", ignore_errors=True)
shutil.rmtree("D:/Meta/cloud_sre_v2/_test_log", ignore_errors=True)
os.makedirs("D:/Meta/cloud_sre_v2/_test_data", exist_ok=True)
os.makedirs("D:/Meta/cloud_sre_v2/_test_data/queue", exist_ok=True)
os.makedirs("D:/Meta/cloud_sre_v2/_test_log", exist_ok=True)

# Exact same as smoke test
from services.orchestrator import ServiceOrchestrator
orch = ServiceOrchestrator(
    db_path="D:/Meta/cloud_sre_v2/_test_data/test.db",
    log_dir="D:/Meta/cloud_sre_v2/_test_log",
)
orch.start()

# Wait extra
time.sleep(5)

# Check each process
for name, entry in orch._processes.items():
    proc = entry["proc"]
    rc = proc.poll()
    if rc is not None:
        stderr = proc.stderr.read().decode()
        stdout = proc.stdout.read().decode()
        print(f"\n=== {name}: CRASHED (rc={rc}) ===")
        print(f"STDOUT:\n{stdout[-500:]}")
        print(f"STDERR:\n{stderr[-1500:]}")
    else:
        print(f"{name}: RUNNING pid={proc.pid}")

health = orch.check_health()
for n, h in health.items():
    print(f"  {n}: {h['status']}")

orch.shutdown()
