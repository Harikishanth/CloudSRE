"""
CloudSRE v2 — Prometheus-Style Metrics Infrastructure.

Real metrics collection that each microservice exposes via /metrics endpoint.
The agent reads these to diagnose which service is failing.

Each service has counters, gauges, and histograms that update in real-time
as real requests flow through real services.

Kube SRE Gym equivalent: They only have pod status (Running/CrashLoopBackOff).
We have real application-level metrics — error rates, latencies, queue depths.
"""

import time
import threading
from typing import Dict, Any, List, Optional


class Counter:
    """A monotonically increasing counter (Prometheus-style).

    Example: request_count, error_count, bytes_transferred
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1):
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value

    def reset(self):
        with self._lock:
            self._value = 0


class Gauge:
    """A value that can go up and down (Prometheus-style).

    Example: cpu_percent, memory_mb, active_connections
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float):
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0):
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self):
        with self._lock:
            self._value = 0.0


class Histogram:
    """Tracks distribution of values (Prometheus-style).

    Example: request_duration_ms, response_size_bytes
    Records observations and provides sum, count, avg, p50, p95, p99.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: List[float] = []
        self._lock = threading.Lock()

    def observe(self, value: float):
        with self._lock:
            self._values.append(value)
            # Keep last 1000 observations to bound memory
            if len(self._values) > 1000:
                self._values = self._values[-1000:]

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def sum(self) -> float:
        return sum(self._values) if self._values else 0.0

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0-100)."""
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "sum": round(self.sum, 2),
            "avg": round(self.avg, 2),
            "p50": round(self.percentile(50), 2),
            "p95": round(self.percentile(95), 2),
            "p99": round(self.percentile(99), 2),
        }

    def reset(self):
        with self._lock:
            self._values.clear()


class ServiceMetrics:
    """Metrics collection for a single microservice.

    Each service creates one of these. It tracks:
      - Request count, error count, error rate
      - Response latency (histogram)
      - CPU and memory gauges
      - Custom gauges for service-specific metrics

    The agent reads these via: curl http://localhost:<port>/metrics
    """

    def __init__(self, service_name: str):
        self.service_name = service_name

        # Standard metrics every service has
        self.request_count = Counter(f"{service_name}_requests_total")
        self.error_count = Counter(f"{service_name}_errors_total")
        self.latency = Histogram(f"{service_name}_request_duration_ms")

        # Resource gauges
        self.cpu_percent = Gauge(f"{service_name}_cpu_percent")
        self.memory_mb = Gauge(f"{service_name}_memory_mb")
        self.active_connections = Gauge(f"{service_name}_active_connections")

        # Custom gauges for service-specific data
        self._custom_gauges: Dict[str, Gauge] = {}
        self._custom_counters: Dict[str, Counter] = {}

    def record_request(self, duration_ms: float, is_error: bool = False):
        """Record a request. Called by service middleware on every HTTP request."""
        self.request_count.inc()
        self.latency.observe(duration_ms)
        if is_error:
            self.error_count.inc()

    def add_gauge(self, name: str, description: str = "") -> Gauge:
        """Add a custom gauge. Returns the gauge for direct manipulation."""
        full_name = f"{self.service_name}_{name}"
        gauge = Gauge(full_name, description)
        self._custom_gauges[name] = gauge
        return gauge

    def add_counter(self, name: str, description: str = "") -> Counter:
        """Add a custom counter."""
        full_name = f"{self.service_name}_{name}"
        counter = Counter(full_name, description)
        self._custom_counters[name] = counter
        return counter

    @property
    def error_rate(self) -> float:
        """Current error rate (0.0 to 1.0)."""
        total = self.request_count.value
        if total == 0:
            return 0.0
        return round(self.error_count.value / total, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as a dictionary for /metrics endpoint.

        The agent sees this as JSON when it does:
            curl http://localhost:<port>/metrics
        """
        result = {
            "service": self.service_name,
            "requests_total": self.request_count.value,
            "errors_total": self.error_count.value,
            "error_rate": self.error_rate,
            "latency": self.latency.summary(),
            "cpu_percent": round(self.cpu_percent.value, 1),
            "memory_mb": round(self.memory_mb.value, 1),
            "active_connections": int(self.active_connections.value),
        }

        # Add custom metrics
        for name, gauge in self._custom_gauges.items():
            result[name] = round(gauge.value, 2)
        for name, counter in self._custom_counters.items():
            result[name] = counter.value

        return result

    def to_prometheus(self) -> str:
        """Export in Prometheus text exposition format.

        For agents that understand Prometheus-native format:
            # HELP payment_requests_total Total requests
            # TYPE payment_requests_total counter
            payment_requests_total 1847
        """
        lines = []
        d = self.to_dict()
        for key, value in d.items():
            if key == "service":
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"{self.service_name}_{key}_{sub_key} {sub_value}")
            else:
                lines.append(f"{self.service_name}_{key} {value}")
        return "\n".join(lines)

    def reset(self):
        """Reset all metrics. Called at episode start."""
        self.request_count.reset()
        self.error_count.reset()
        self.latency.reset()
        self.cpu_percent.set(0.0)
        self.memory_mb.set(0.0)
        self.active_connections.set(0.0)
        for gauge in self._custom_gauges.values():
            gauge.reset()
        for counter in self._custom_counters.values():
            counter.reset()
