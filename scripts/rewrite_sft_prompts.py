"""Rewrite the system prompt in sft_training_data.jsonl to expose all 16 services.

The original generator (generate_sft_data.py) hardcoded a 6-service prompt that
no longer matches the production environment after the 10-service expansion.
This script rewrites only the system message (messages[0]) on every line and
preserves the user/assistant demonstrations verbatim.

The original file is preserved as sft_training_data_v6services.jsonl so the
change is reversible without re-running the dataset generation.

Usage (from cloud_sre_v2/ or anywhere):
    python rewrite_sft_prompts.py
    python rewrite_sft_prompts.py --input <path> --output <path>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

NEW_SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a production cloud platform.

AVAILABLE COMMANDS:
  status                                       - Check all service health
  curl http://localhost:<PORT>/healthz          - Check specific service health
  curl http://<svc>.<region>.internal/healthz   - Check via cloud DNS
  cat /var/log/<service>/error.log              - Read service error logs
  restart_service <service>                     - Restart a crashed/degraded service
  queue drain <N>                               - Drain N messages from the queue
  sqlite3 /data/app.db "<SQL>"                  - Query the database

SERVICES (16 total):
  us-east-1: payment(8001), auth(8002), gateway(8008), billing(8013), config(8014), loadbalancer(8016)
  eu-west-1: worker(8003), search(8007), scheduler(8009), storage(8010), metrics_collector(8011)
  ap-south-1: frontend(8004), cache(8005), notification(8006), email(8012), dns(8015)

SRE WORKFLOW: triage (status) -> investigate (logs/healthz) -> fix (restart/drain/config) -> verify (status)
Output ONLY the next command. No explanations."""


def rewrite_file(input_path: Path, output_path: Path, backup_path: Path | None) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if backup_path is not None and not backup_path.exists():
        shutil.copy2(input_path, backup_path)

    rewritten = 0
    skipped_no_system = 0
    total = 0
    bytes_in = input_path.stat().st_size

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            total += 1
            ex = json.loads(line)
            messages = ex.get("messages", [])
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = NEW_SYSTEM_PROMPT
                rewritten += 1
            else:
                ex["messages"] = [{"role": "system", "content": NEW_SYSTEM_PROMPT}] + messages
                skipped_no_system += 1
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)
    bytes_out = output_path.stat().st_size

    return {
        "input": str(input_path),
        "output": str(output_path),
        "backup": str(backup_path) if backup_path else None,
        "examples_total": total,
        "system_message_rewritten": rewritten,
        "system_message_prepended": skipped_no_system,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input", default=str(here / "sft_training_data.jsonl"))
    parser.add_argument("--output", default=str(here / "sft_training_data.jsonl"))
    parser.add_argument("--backup", default=str(here / "sft_training_data_v6services.jsonl"))
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing the .v6services.jsonl backup (not recommended)")
    args = parser.parse_args(argv)

    backup_path = None if args.no_backup else Path(args.backup)
    result = rewrite_file(Path(args.input), Path(args.output), backup_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
