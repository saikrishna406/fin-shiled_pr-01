"""Circuit Breaker & Resilience Control Engine for Fin-Shield Analytics.

Monitors IDS alert frequency, buffer pool capacity, and latency SLAs.
Triggers systemic traffic throttling, policy fallbacks, or emergency flow pauses.
"""

from typing import Dict, Any, List


class CircuitBreaker:
    """Resilience and systemic risk circuit breaker."""

    def __init__(
        self,
        ids_alert_threshold_ratio: float = 0.30,
        max_buffer_capacity: int = 1000,
        latency_sla_ms: float = 100.0,
    ):
        self.ids_alert_threshold_ratio = ids_alert_threshold_ratio
        self.max_buffer_capacity = max_buffer_capacity
        self.latency_sla_ms = latency_sla_ms

        self.is_triggered = False
        self.trigger_reason = ""
        self.trigger_count = 0

    def evaluate_system_health(
        self,
        recent_alert_count: int,
        window_size: int,
        current_buffer_depth: int,
        last_latency_ms: float,
    ) -> Dict[str, Any]:
        """Check system state against threshold parameters."""
        alert_ratio = (recent_alert_count / window_size) if window_size > 0 else 0.0

        if alert_ratio >= self.ids_alert_threshold_ratio:
            self.is_triggered = True
            self.trigger_reason = f"IDS alert rate breached: {alert_ratio*100:.1f}% >= {self.ids_alert_threshold_ratio*100:.1f}%"
            self.trigger_count += 1
        elif current_buffer_depth >= self.max_buffer_capacity:
            self.is_triggered = True
            self.trigger_reason = f"Buffer capacity exceeded: {current_buffer_depth} >= {self.max_buffer_capacity}"
            self.trigger_count += 1
        elif last_latency_ms > self.latency_sla_ms:
            self.is_triggered = True
            self.trigger_reason = f"Inference latency SLA breached: {last_latency_ms:.2f} ms > {self.latency_sla_ms} ms"
            self.trigger_count += 1
        else:
            self.is_triggered = False
            self.trigger_reason = "System operating within normal parameters"

        return {
            "is_triggered": self.is_triggered,
            "trigger_reason": self.trigger_reason,
            "trigger_count": self.trigger_count,
            "alert_ratio": round(alert_ratio, 4),
            "buffer_depth": current_buffer_depth,
            "last_latency_ms": round(last_latency_ms, 2),
        }
