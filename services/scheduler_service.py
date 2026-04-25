"""
CloudSRE v2 — Scheduler Service (port 8009).

Job/cron scheduler. Cascade: DB down → scheduler can't persist job state → duplicate jobs fire.

Fault types:
  - scheduler_stuck: Scheduler loop frozen, no jobs execute
  - duplicate_execution: Same job runs multiple times (idempotency failure)
"""

import time
from fastapi import Request
from cloud_sre_v2.services.base_service import BaseService


class SchedulerService(BaseService):
    def __init__(self, port: int = 8009, log_dir: str = "/var/log"):
        super().__init__("scheduler", port=port, log_dir=log_dir)
        self._jobs = {}
        self._executed_count = 0
        self._failed_count = 0
        self._stuck = False
        self._duplicate_mode = False
        self._seed_jobs()
        self._register_routes()

    def _seed_jobs(self):
        now = time.time()
        self._jobs = {
            "job:billing_daily": {"schedule": "0 2 * * *", "last_run": now - 86400, "status": "completed", "runs": 365},
            "job:cleanup_temp": {"schedule": "*/30 * * * *", "last_run": now - 1800, "status": "completed", "runs": 48},
            "job:report_weekly": {"schedule": "0 6 * * 1", "last_run": now - 604800, "status": "completed", "runs": 52},
            "job:db_backup": {"schedule": "0 3 * * *", "last_run": now - 43200, "status": "completed", "runs": 365},
            "job:index_refresh": {"schedule": "*/15 * * * *", "last_run": now - 900, "status": "completed", "runs": 96},
        }
        self._executed_count = sum(j["runs"] for j in self._jobs.values())

    def _register_routes(self):
        @self.app.get("/scheduler/jobs")
        async def list_jobs():
            return {"service": "scheduler", "jobs": self._jobs, "total": len(self._jobs),
                    "stuck": self._stuck, "duplicate_mode": self._duplicate_mode}

        @self.app.get("/scheduler/stats")
        async def scheduler_stats():
            return {"service": "scheduler", "total_jobs": len(self._jobs),
                    "executed": self._executed_count, "failed": self._failed_count,
                    "stuck": self._stuck, "duplicate_mode": self._duplicate_mode}

        @self.app.post("/scheduler/trigger/{job_id}")
        async def trigger_job(job_id: str):
            if self._stuck:
                return {"error": "SCHEDULER_STUCK", "job_id": job_id}
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["last_run"] = time.time()
                self._executed_count += 1
                return {"status": "triggered", "job_id": job_id}
            return {"error": "JOB_NOT_FOUND"}

        @self.app.post("/scheduler/unstick")
        async def unstick():
            self._stuck = False
            self.logger.info("Scheduler unstuck — resuming job execution")
            return {"stuck": False}

    def inject_stuck(self):
        self._stuck = True
        self.set_degraded()
        self.logger.error("FAULT: Scheduler loop frozen — no jobs executing")

    def inject_duplicate_execution(self):
        self._duplicate_mode = True
        self._failed_count += 12
        self.set_degraded()
        self.logger.error("FAULT: Duplicate job execution detected — 12 duplicates")

    def reset(self):
        super().reset()
        self._stuck = False
        self._duplicate_mode = False
        self._failed_count = 0
        self._seed_jobs()
