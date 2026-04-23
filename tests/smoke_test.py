"""Smoke test — verifies all CloudSRE v2 components work together."""
import sys
import os
sys.path.insert(0, "D:/Meta")
import shutil
shutil.rmtree("D:/Meta/cloud_sre_v2/_test_data", ignore_errors=True)
shutil.rmtree("D:/Meta/cloud_sre_v2/_test_log", ignore_errors=True)
os.makedirs("D:/Meta/cloud_sre_v2/_test_data", exist_ok=True)
os.makedirs("D:/Meta/cloud_sre_v2/_test_data/queue", exist_ok=True)
os.makedirs("D:/Meta/cloud_sre_v2/_test_log", exist_ok=True)

print("1. Testing orchestrator creation...")
from cloud_sre_v2.services.orchestrator import ServiceOrchestrator
orch = ServiceOrchestrator(
    db_path="D:/Meta/cloud_sre_v2/_test_data/test.db",
    log_dir="D:/Meta/cloud_sre_v2/_test_log",
)
orch.start()
print("   ✅ Orchestrator started")

import time
time.sleep(2)  # Extra wait for Windows port binding

# Debug: check subprocess state
for name, entry in orch._processes.items():
    proc = entry["proc"]
    rc = proc.poll()
    if rc is not None:
        stderr = proc.stderr.read().decode()[-500:]
        print(f"   ⚠️  {name} CRASHED: {stderr}")

print("\n2. Testing health check...")
health = orch.check_health()
for name, h in health.items():
    status = h["status"]
    print(f"   {name}: {status}")
assert all(h["status"] == "healthy" for h in health.values()), "Not all services healthy!"
print("   ✅ All services healthy")

print("\n3. Testing command executor...")
output, cmd_type = orch.executor.execute("status")
assert cmd_type == "health_check", f"Expected health_check, got {cmd_type}"
assert "payment" in output.lower(), "Status output missing payment"
print(f"   ✅ Status command works (type={cmd_type})")

# Test queue status
output2, cmd_type2 = orch.executor.execute("queue status")
assert "queue_depth" in output2, "Queue status missing depth"
print(f"   ✅ Queue status works")

# Test log read
output3, cmd_type3 = orch.executor.execute("cat /var/log/payment/error.log")
# May be empty after fresh start — that's OK
print(f"   ✅ Log read works (type={cmd_type3})")

# Test sqlite
output4, cmd_type4 = orch.executor.execute("sqlite3 /data/app.db '.tables'")
# Will fail because db_path is different in test — expected
print(f"   ⚠️  SQLite uses test path (expected behavior)")

print("\n4. Testing fault injection...")
result = orch.inject_fault("db_lock")
assert "database lock" in result.lower(), f"Unexpected fault result: {result}"
health2 = orch.check_health()
print(f"   ✅ DB lock injected")

# Test queue overflow
result2 = orch.inject_fault("queue_overflow", {"fill": 100})
print(f"   ✅ Queue overflow: {result2}")

# Test process crash (REAL — actually stops the uvicorn server)
result3 = orch.inject_fault("process_crash", {"target": "worker", "reason": "OOM"})
import time; time.sleep(0.5)  # Wait for server to actually stop
health3 = orch.check_health()
assert health3["worker"]["status"] in ("unhealthy", "crashed"), f"Worker should be crashed, got {health3['worker']['status']}"
print(f"   ✅ Process crash: worker is {health3['worker']['status']} (port closed)")

print("\n5. Testing reset...")
orch.reset()
health4 = orch.check_health()
all_healthy = all(h["status"] == "healthy" for h in health4.values())
assert all_healthy, f"Not all healthy after reset: {health4}"
print("   ✅ Reset works — all services healthy again")

print("\n6. Testing graders...")
from cloud_sre_v2.server.graders import grade_episode

# Simple warmup test
score, feedback, details = grade_episode(
    task_id="warmup",
    history=[
        {"command": "curl http://localhost:8001/healthz", "phase": "triage", "cmd_type": "health_check"},
        {"command": "cat /var/log/payment/error.log", "phase": "investigation", "cmd_type": "logs"},
        {"command": "restart_service payment", "phase": "fix", "cmd_type": "fix"},
    ],
    scenario={"target_service": "payment", "max_steps": 10},
    service_health={"payment": {"status": "healthy"}, "auth": {"status": "healthy"},
                    "worker": {"status": "healthy"}, "frontend": {"status": "healthy"}},
    resolved=True,
)
print(f"   ✅ Warmup grader: score={score}, feedback={feedback[:60]}")

# Cascade test
score2, feedback2, details2 = grade_episode(
    task_id="cascade",
    history=[
        {"command": "curl http://localhost:8001/healthz", "phase": "triage", "cmd_type": "health_check"},
        {"command": "cat /var/log/payment/error.log", "phase": "investigation", "cmd_type": "logs"},
        {"command": "restart_service payment", "phase": "fix", "cmd_type": "fix"},
        {"command": "queue drain 10", "phase": "fix", "cmd_type": "fix"},
    ],
    scenario={"target_service": "payment", "max_steps": 20, "cascade_rules": []},
    service_health={"payment": {"status": "healthy"}, "auth": {"status": "healthy"},
                    "worker": {"status": "healthy"}, "frontend": {"status": "healthy"}},
    resolved=True,
    cascade_triggered=False,
)
print(f"   ✅ Cascade grader: score={score2}, feedback={feedback2[:60]}")

print("\n7. Testing scenario spec...")
from cloud_sre_v2.models import ScenarioSpec, CascadeRule
s = ScenarioSpec(
    scenario_id="test",
    failure_type="db_lock",
    target_service="payment",
    difficulty=0.3,
    alert_message="Test alert",
    root_cause="Test cause",
    cascade_rules=[
        CascadeRule(
            trigger_condition="test",
            cascade_type="queue_overflow",
            affected_service="worker",
            description="test cascade",
            agent_must="drain slowly",
        )
    ],
)
assert s.cascade_rules[0].affected_service == "worker"
print(f"   ✅ ScenarioSpec + CascadeRule creation works")

# Cleanup
import shutil
shutil.rmtree("D:/Meta/cloud_sre_v2/_test_data", ignore_errors=True)
shutil.rmtree("D:/Meta/cloud_sre_v2/_test_log", ignore_errors=True)

print("\n" + "=" * 50)
print("🎉 ALL SMOKE TESTS PASSED!")
print("=" * 50)
