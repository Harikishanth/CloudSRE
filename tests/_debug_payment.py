"""Debug why payment service crashes on startup."""
import sys
import os
import time

sys.path.insert(0, "D:/Meta")
os.makedirs("D:/Meta/cloud_sre_v2/_test_data", exist_ok=True)
os.makedirs("D:/Meta/cloud_sre_v2/_test_log", exist_ok=True)

from cloud_sre_v2.services.orchestrator import ServiceOrchestrator

orch = ServiceOrchestrator(
    db_path="D:/Meta/cloud_sre_v2/_test_data/test.db",
    log_dir="D:/Meta/cloud_sre_v2/_test_log",
)
orch.start()
time.sleep(3)

for name, entry in orch._processes.items():
    proc = entry["proc"]
    rc = proc.poll()
    if rc is not None:
        stderr = proc.stderr.read().decode()[-2000:]
        stdout = proc.stdout.read().decode()[-500:]
        print(f"{name}: CRASHED (rc={rc})")
        print(f"  STDOUT: {stdout}")
        print(f"  STDERR: {stderr}")
        print("---")
    else:
        print(f"{name}: RUNNING pid={proc.pid}")

health = orch.check_health()
for n, h in health.items():
    s = h["status"]
    print(f"  health: {n} = {s}")
orch.shutdown()
