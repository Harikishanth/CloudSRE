"""
CloudSRE v2 — Colab/HF Space training diagnostic smoke.

Static-only checks (no subprocess spawn) that surface issues that will bite
during training on Colab notebooks or HuggingFace Spaces. Writes NDJSON
runtime evidence to <workspace>/debug-4e9608.log.

Run from workspace root:
    python cloud_sre_v2/diag_colab_hf.py
"""

# region agent log helpers
import json
import os
import re
import sys
import time
import shutil
import inspect
import tempfile
import traceback
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
LOG_PATH = WORKSPACE / "debug-4e9608.log"
SESSION = "4e9608"
RUN_ID = "diag_initial"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    entry = {
        "sessionId": SESSION,
        "id": f"log_{int(time.time() * 1000)}_{hypothesis_id}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": RUN_ID,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _safe(test_id: str, location: str, fn):
    try:
        fn()
    except Exception as exc:
        _log(test_id, location, "TEST_RAISED",
             {"error_type": type(exc).__name__,
              "error": str(exc)[:400],
              "traceback": traceback.format_exc()[:1200]})
# endregion


# region agent log H1 — FastAPI /health route registration
def test_h1_health_route():
    from cloud_sre_v2.server.app import app
    routes = []
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = sorted(getattr(r, "methods", []) or [])
        if path:
            routes.append({"path": path, "methods": methods})
    paths = [r["path"] for r in routes]
    _log("H1", "diag_colab_hf.py:test_h1_health_route",
         "FastAPI route inventory after create_app",
         {"total_routes": len(routes),
          "has_/health": "/health" in paths,
          "has_/healthz": "/healthz" in paths,
          "has_/reset": "/reset" in paths,
          "has_/step": "/step" in paths,
          "has_/state": "/state" in paths,
          "has_/tasks": "/tasks" in paths,
          "all_paths": sorted(set(paths))})
# endregion


# region agent log H2 — SFT training data covers new services
def test_h2_sft_coverage():
    sft_path = WORKSPACE / "cloud_sre_v2" / "sft_training_data.jsonl"
    if not sft_path.exists():
        _log("H2", "diag_colab_hf.py:test_h2_sft_coverage",
             "SFT file missing", {"path": str(sft_path)})
        return
    new_services = ["search", "gateway", "scheduler", "storage",
                    "metrics_collector", "email", "billing", "config",
                    "dns", "loadbalancer"]
    new_ports = [str(p) for p in range(8007, 8017)]
    svc_hits = {svc: 0 for svc in new_services}
    port_hits = {port: 0 for port in new_ports}
    sample_prompt = ""
    total = 0
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for m in obj.get("messages", []):
                if m.get("role") == "system":
                    txt = m.get("content", "")
                    if total == 1:
                        sample_prompt = txt[:600]
                    for svc in new_services:
                        if svc in txt:
                            svc_hits[svc] += 1
                    for port in new_ports:
                        if port in txt:
                            port_hits[port] += 1
                    break
            if total >= 200:
                break
    _log("H2", "diag_colab_hf.py:test_h2_sft_coverage",
         "SFT system-prompt coverage of new services",
         {"sample_size_lines": total,
          "service_mention_counts": svc_hits,
          "port_mention_counts": port_hits,
          "any_new_service_mentioned": any(v > 0 for v in svc_hits.values()),
          "first_system_prompt_excerpt": sample_prompt})
# endregion


# region agent log H3 — _write_service_log silent failure mode
def test_h3_silent_log_swallow():
    from cloud_sre_v2.services.orchestrator import ServiceOrchestrator
    src = inspect.getsource(ServiceOrchestrator._write_service_log)
    has_silent_pattern = "except OSError" in src and "pass" in src
    orch = ServiceOrchestrator()
    tmp = Path(tempfile.mkdtemp(prefix="cloudsre_diag_"))

    orch.log_dir = str(tmp)
    orch._write_service_log("payment", "error", "DIAG_HAPPY_PATH")
    happy_file = tmp / "payment" / "error.log"
    happy_written = happy_file.exists() and happy_file.stat().st_size > 0

    fake_parent = tmp / "log_dir_is_a_file"
    fake_parent.write_text("not a dir")
    orch.log_dir = str(fake_parent)
    raised = None
    try:
        orch._write_service_log("payment", "error", "DIAG_BROKEN_PATH")
    except Exception as exc:
        raised = type(exc).__name__
    silent_on_broken = raised is None
    broken_file_exists = (fake_parent / "payment" / "error.log").exists()
    _log("H3", "diag_colab_hf.py:test_h3_silent_log_swallow",
         "_write_service_log error-handling characterization",
         {"has_silent_except_oserror": has_silent_pattern,
          "happy_path_wrote_file": happy_written,
          "broken_path_silent": silent_on_broken,
          "exception_type": raised,
          "broken_path_file_exists": broken_file_exists,
          "platform": sys.platform})
# endregion


# region agent log H4 — fallocate availability for disk_full
def test_h4_fallocate_availability():
    fpath = shutil.which("fallocate")
    truncate_path = shutil.which("truncate")
    df_path = shutil.which("df")
    _log("H4", "diag_colab_hf.py:test_h4_fallocate_availability",
         "Tool availability for OS-level disk_full injection",
         {"platform": sys.platform,
          "fallocate_resolved": fpath,
          "truncate_resolved": truncate_path,
          "df_resolved": df_path,
          "fallocate_present": fpath is not None})
# endregion


# region agent log H5 — Scenario fault wiring coverage
def test_h5_fault_wiring():
    orch_src = (WORKSPACE / "cloud_sre_v2" / "services" /
                "orchestrator.py").read_text(encoding="utf-8")
    consts_src = (WORKSPACE / "cloud_sre_v2" / "server" /
                  "constants.py").read_text(encoding="utf-8")
    injector_keys = set(re.findall(
        r'"(\w+)"\s*:\s*self\._inject_\w+', orch_src))
    failure_types = set(re.findall(r'failure_type\s*=\s*"(\w+)"', consts_src))
    cascade_types = set(re.findall(r'cascade_type\s*=\s*"(\w+)"', consts_src))
    next_ftypes = set(re.findall(
        r'next_failure_type\s*=\s*"(\w+)"', consts_src))
    benign = {"misleading_signal", "upstream_dependency_failure"}
    referenced = (failure_types | cascade_types | next_ftypes) - benign
    orphans = sorted(referenced - injector_keys)
    cascade_orphans = sorted((cascade_types | next_ftypes) - injector_keys - benign)
    _log("H5", "diag_colab_hf.py:test_h5_fault_wiring",
         "Scenario failure_type coverage in inject_fault dispatcher",
         {"injector_count": len(injector_keys),
          "failure_type_count": len(failure_types),
          "cascade_type_count": len(cascade_types),
          "next_failure_type_count": len(next_ftypes),
          "orphan_failure_types": orphans,
          "orphan_cascade_failure_types": cascade_orphans,
          "injectors": sorted(injector_keys),
          "failure_types_sample": sorted(failure_types)[:25]})
# endregion


# region agent log driver
if __name__ == "__main__":
    _log("driver", "diag_colab_hf.py:main",
         "Diagnostic run starting",
         {"workspace": str(WORKSPACE),
          "python": sys.version.split()[0],
          "platform": sys.platform})
    _safe("H1", "diag_colab_hf.py:test_h1_health_route", test_h1_health_route)
    _safe("H2", "diag_colab_hf.py:test_h2_sft_coverage", test_h2_sft_coverage)
    _safe("H3", "diag_colab_hf.py:test_h3_silent_log_swallow",
          test_h3_silent_log_swallow)
    _safe("H4", "diag_colab_hf.py:test_h4_fallocate_availability",
          test_h4_fallocate_availability)
    _safe("H5", "diag_colab_hf.py:test_h5_fault_wiring",
          test_h5_fault_wiring)
    _log("driver", "diag_colab_hf.py:main",
         "Diagnostic run complete", {"log_path": str(LOG_PATH)})
    print(f"Diagnostic complete. Log written to: {LOG_PATH}")
# endregion
