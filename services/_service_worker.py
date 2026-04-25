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
    parser.add_argument("--service", required=True, choices=[
        "payment", "auth", "worker", "frontend", "cache", "notification",
        "search", "gateway", "scheduler", "storage", "metrics_collector",
        "email", "billing", "config", "dns", "loadbalancer",
    ])
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

    # Create the appropriate service — 16 real OS processes
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
    elif args.service == "search":
        from cloud_sre_v2.services.search_service import SearchService
        svc = SearchService(port=args.port, log_dir=args.log_dir)
    elif args.service == "gateway":
        from cloud_sre_v2.services.gateway_service import GatewayService
        svc = GatewayService(port=args.port, log_dir=args.log_dir)
    elif args.service == "scheduler":
        from cloud_sre_v2.services.scheduler_service import SchedulerService
        svc = SchedulerService(port=args.port, log_dir=args.log_dir)
    elif args.service == "storage":
        from cloud_sre_v2.services.storage_service import StorageService
        svc = StorageService(port=args.port, log_dir=args.log_dir)
    elif args.service == "metrics_collector":
        from cloud_sre_v2.services.metrics_collector_service import MetricsCollectorService
        svc = MetricsCollectorService(port=args.port, log_dir=args.log_dir)
    elif args.service == "email":
        from cloud_sre_v2.services.email_service import EmailService
        svc = EmailService(port=args.port, log_dir=args.log_dir)
    elif args.service == "billing":
        from cloud_sre_v2.services.billing_service import BillingService
        svc = BillingService(port=args.port, log_dir=args.log_dir)
    elif args.service == "config":
        from cloud_sre_v2.services.config_service import ConfigService
        svc = ConfigService(port=args.port, log_dir=args.log_dir)
    elif args.service == "dns":
        from cloud_sre_v2.services.dns_service import DNSService
        svc = DNSService(port=args.port, log_dir=args.log_dir)
    elif args.service == "loadbalancer":
        from cloud_sre_v2.services.loadbalancer_service import LoadBalancerService
        svc = LoadBalancerService(port=args.port, log_dir=args.log_dir)
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
