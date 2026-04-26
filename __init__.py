"""CloudSRE v2 — Cloud Site Reliability Engineering Environment.

A cascading incident response RL environment where an AI agent learns to
debug real microservices and handle the cascading failures that occur
after the primary fix.

Usage:
    # As a package
    from cloud_sre_v2 import CloudSREAction, CloudSREObservation, CloudSREState
    from cloud_sre_v2 import CloudSREEnv  # sync client

    # Training utilities
    from cloud_sre_v2 import get_training_utils
    tu = get_training_utils()
"""

# Lazy imports — DO NOT eagerly import models or client here.
# Service worker subprocesses import services.xxx which
# triggers this __init__.py. If we eagerly import client.py, it pulls
# in openenv-core which may not be installed or may hang.

__all__ = [
    "CloudSREAction",
    "CloudSREObservation",
    "CloudSREState",
    "CloudSREEnv",
    "ScenarioSpec",
    "CascadeRule",
    "IncidentStep",
    "AdversarialScenarioSpec",
]


def __getattr__(name):
    """Lazy import to avoid triggering openenv chain in service workers."""
    if name in ("CloudSREAction", "CloudSREObservation", "CloudSREState",
                "ScenarioSpec", "CascadeRule", "IncidentStep", "AdversarialScenarioSpec"):
        from .models import (
            CloudSREAction, CloudSREObservation, CloudSREState,
            ScenarioSpec, CascadeRule, IncidentStep, AdversarialScenarioSpec,
        )
        return locals()[name]
    if name == "CloudSREEnv":
        from .client import CloudSREEnv
        return CloudSREEnv
    raise AttributeError(f"module 'cloud_sre_v2' has no attribute {name!r}")


def get_training_utils():
    """Lazy-import training utilities from train.py.

    Returns a dict with: SYSTEM_PROMPT, rollout_once, format_observation,
    format_history, parse_commands, reward_total, plot_rewards.

    Usage in Colab/HF:
        from cloud_sre_v2 import get_training_utils
        tu = get_training_utils()
        SYSTEM_PROMPT = tu["SYSTEM_PROMPT"]
        rollout_once = tu["rollout_once"]
    """
    from . import train as _train
    return {
        "SYSTEM_PROMPT": _train.SYSTEM_PROMPT,
        "rollout_once": _train.rollout_once,
        "format_observation": _train.format_observation,
        "format_history": _train.format_history,
        "parse_commands": _train.parse_commands,
        "reward_total": _train.reward_total,
        "plot_rewards": _train.plot_rewards,
    }
