"""
CloudSRE v2 — Standalone Service Worker.

Runs a single service as an independent OS process with its own PID.
Each service creates its own Database and Queue connections to shared files.

Usage:
    python -m cloud_sre_v2.services._service_worker \
        --service payment --port 8001 \
        --db-path /data/app.db --queue-dir /data/queue --log-dir /var/log

This is how real microservices work:
  - Separate processes
  - Shared database (via file)
  - Shared queue (via filesystem)
  - Independent crash domains
"""

import argparse
import sys
import os
import uvicorn
import logging

logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Run a CloudSRE service as a standalone process")
    parser.add_argument("--service", required=True, choices=["payment", "auth", "worker", "frontend", "cache", "notification"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    # Each process creates its OWN database and queue connections
    # to the SHARED files on disk — exactly like real microservices
    from cloud_sre_v2.infra.database import Database
    from cloud_sre_v2.infra.queue import MessageQueue

    db = Database(args.db_path)
    # Don't initialize schema — orchestrator already did that
    db._created = True

    queue = MessageQueue(max_size=1000, queue_dir=args.queue_dir)

    # Create the appropriate service
    if args.service == "payment":
        from cloud_sre_v2.services.payment_service import PaymentService
        svc = PaymentService(db, queue, port=args.port, log_dir=args.log_dir)
    elif args.service == "auth":
        from cloud_sre_v2.services.auth_service import AuthService
        svc = AuthService(db, port=args.port, log_dir=args.log_dir)
    elif args.service == "worker":
        from cloud_sre_v2.services.worker_service import WorkerService
        svc = WorkerService(db, queue, port=args.port, log_dir=args.log_dir)
    elif args.service == "frontend":
        from cloud_sre_v2.services.frontend_proxy import FrontendProxy
        svc = FrontendProxy(port=args.port, log_dir=args.log_dir)
    elif args.service == "cache":
        from cloud_sre_v2.services.cache_service import CacheService
        svc = CacheService(port=args.port, log_dir=args.log_dir)
    elif args.service == "notification":
        from cloud_sre_v2.services.notification_service import NotificationService
        svc = NotificationService(port=args.port, log_dir=args.log_dir)
    else:
        print(f"Unknown service: {args.service}", file=sys.stderr)
        sys.exit(1)

    # Log the real PID — this is what makes it real
    pid = os.getpid()
    print(f"[{args.service}] PID={pid} port={args.port} db={args.db_path}", flush=True)

    # Run uvicorn in THIS process (not a thread)
    uvicorn.run(
        svc.app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
