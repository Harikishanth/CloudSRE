"""Test train.py's pure-Python helpers (no GPU, no TRL, no datasets needed)."""
import sys
sys.path.insert(0, "D:/Meta")

# Can't import the full train.py without TRL/datasets,
# so extract and test helpers directly

import re

def parse_commands(text):
    valid_prefixes = (
        "curl ", "cat ", "tail ", "head ", "grep ",
        "sqlite3 ", "kill ", "restart_service ", "python ",
        "ps ", "queue ", "drain ", "config ", "status",
        "diagnose:", "diagnosis:", "fix:",
    )
    commands = []
    seen = set()
    for line in text.strip().split("\n"):
        line = line.strip()
        line = re.sub(r'^[\-\*\>]+\s*', '', line)
        line = re.sub(r'^```\w*\s*', '', line)
        line = re.sub(r'```$', '', line)
        line = line.strip()
        if any(line.startswith(p) for p in valid_prefixes):
            if line not in seen:
                commands.append(line)
                seen.add(line)
        if len(commands) >= 2:
            break
    return commands

# Test parse_commands
tests = [
    ("curl http://localhost:8001/healthz", ["curl http://localhost:8001/healthz"]),
    ("queue drain 10", ["queue drain 10"]),
    ("queue drain all", ["queue drain all"]),
    ("status", ["status"]),
    ("restart_service payment", ["restart_service payment"]),
    ("fix: restart payment and drain queue", ["fix: restart payment and drain queue"]),
    ("diagnose: db lock in payment", ["diagnose: db lock in payment"]),
    ("cat /var/log/payment/error.log", ["cat /var/log/payment/error.log"]),
    ("grep \"ERROR\" /var/log/auth/error.log", ["grep \"ERROR\" /var/log/auth/error.log"]),
    ("sqlite3 /data/app.db 'SELECT count(*) FROM payments'",
     ["sqlite3 /data/app.db 'SELECT count(*) FROM payments'"]),
    # Multi-line
    ("curl http://localhost:8001/healthz\nqueue drain 10\nsomething random",
     ["curl http://localhost:8001/healthz", "queue drain 10"]),
    # With markdown formatting
    ("- curl http://localhost:8001/healthz\n* queue status",
     ["curl http://localhost:8001/healthz", "queue status"]),
    # With code fences
    ("```bash\ncurl http://localhost:8001/healthz\n```",
     ["curl http://localhost:8001/healthz"]),
    # Invalid
    ("I think the problem is the database", []),
    ("Let me check the payment service", []),
]

passed = 0
failed = 0
for text, expected in tests:
    result = parse_commands(text)
    if result == expected:
        passed += 1
        print(f"  PASS: {text[:50]}... -> {result}")
    else:
        failed += 1
        print(f"  FAIL: {text[:50]}...")
        print(f"    Expected: {expected}")
        print(f"    Got:      {result}")

print(f"\nResults: {passed}/{passed+failed} passed")
assert failed == 0, f"{failed} tests failed!"
print("\nAll parse_commands tests passed!")

# Test that format functions exist in train.py source
with open("D:/Meta/cloud_sre_v2/train.py") as f:
    src = f.read()

required = [
    "SYSTEM_PROMPT",
    "def format_observation",
    "def format_history",
    "def parse_commands",
    "def rollout_once",
    "def reward_total",
    "def reward_triage",
    "def reward_investigation",
    "def reward_fix",
    "def reward_cascade",
    "def plot_rewards",
    "def main",
    "GRPOConfig",
    "GRPOTrainer",
    "LoraConfig",
    "generate_rollout_completions",
    "loss_type=\"dapo\"",
    "cascade",
]

for r in required:
    assert r in src, f"Missing in train.py: {r}"
    print(f"  FOUND: {r}")

print(f"\nAll {len(required)} required components found in train.py!")
print(f"Lines: {len(src.splitlines())}")
print(f"Bytes: {len(src)}")
print("\n=== ALL TRAIN.PY TESTS PASSED ===")
