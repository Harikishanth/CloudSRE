"""Test dynamic scenario generator produces unique scenarios each call."""
import sys
sys.path.insert(0, "D:/Meta")

from cloud_sre_v2.server.constants import _generate_dynamic_scenario, TASK_CONFIGS

# Mock orchestrator (generator doesn't call orchestrator methods)
class MockOrch:
    pass

orch = MockOrch()

print("=== Testing Dynamic Scenario Generator ===\n")

# Generate 10 scenarios and verify uniqueness
scenarios = []
for i in range(10):
    s = _generate_dynamic_scenario(orch, difficulty_range=(0.3, 0.9))
    scenarios.append(s)
    print(f"Episode {i+1:2d}: {s.scenario_id}")
    print(f"  Fault: {s.failure_type} -> {s.target_service}")
    print(f"  Difficulty: {s.difficulty}")
    print(f"  Red herrings: {len(s.misleading_signals)}")
    print(f"  Cascades: {len(s.cascade_rules)}")
    print(f"  Root cause: {s.root_cause[:60]}")
    print()

# Check uniqueness
ids = [s.scenario_id for s in scenarios]
unique_ids = set(ids)
print(f"Generated {len(ids)} scenarios, {len(unique_ids)} unique IDs")
assert len(unique_ids) == len(ids), "Scenario IDs should be unique!"

# Check variety
fault_types = set(s.failure_type for s in scenarios)
targets = set(s.target_service for s in scenarios)
print(f"Fault types used: {fault_types}")
print(f"Target services: {targets}")

# Verify adversarial tier uses the generator
adv_picker = TASK_CONFIGS["adversarial"]["scenario_picker"]
print(f"\nAdversarial picker: {adv_picker.__name__}")
assert "adversarial" in adv_picker.__name__

# Generate 3 adversarial scenarios and verify they're different
a1 = adv_picker(orch)
a2 = adv_picker(orch)
a3 = adv_picker(orch)
print(f"Adversarial 1: {a1.scenario_id} (fault={a1.failure_type})")
print(f"Adversarial 2: {a2.scenario_id} (fault={a2.failure_type})")
print(f"Adversarial 3: {a3.scenario_id} (fault={a3.failure_type})")
assert a1.scenario_id != a2.scenario_id, "Scenarios should be unique!"

print("\n=== ALL DYNAMIC GENERATOR TESTS PASSED ===")
