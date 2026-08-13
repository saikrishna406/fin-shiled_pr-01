"""Behavioral Intrusion Detection System (IDS) Engine for Fin-Shield Analytics.

Monitors transaction stream for velocity spikes, amount anomalies, burst activity,
device anomalies, flash-crash attack patterns, and reserve drain alerts.
"""

from typing import Dict, List, Any
import numpy as np


class BehavioralIDS:
    """Transaction-level Behavioral Intrusion Detection System."""

    def __init__(self, velocity_threshold_1m: int = 5, amount_zscore_threshold: float = 3.0):
        self.velocity_threshold_1m = velocity_threshold_1m
        self.amount_zscore_threshold = amount_zscore_threshold

    def analyze_transaction(self, feature_row: Dict[str, Any], liquidity_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Inspect transaction features and issue behavioral security alerts."""
        alerts = []
        severity_scores = []

        # 1. Velocity Spike / Burst Activity
        tx_1m = feature_row.get("tx_count_1m", 0)
        if tx_1m >= self.velocity_threshold_1m:
            alerts.append({
                "alert_type": "VELOCITY_SPIKE",
                "severity": "HIGH",
                "message": f"High transaction velocity: {tx_1m} tx/min",
            })
            severity_scores.append(80.0)

        # 2. Amount Anomaly
        zscore = feature_row.get("amount_zscore", 0.0)
        if zscore >= self.amount_zscore_threshold:
            alerts.append({
                "alert_type": "AMOUNT_ANOMALY",
                "severity": "HIGH",
                "message": f"Amount z-score deviation: {zscore:.2f}",
            })
            severity_scores.append(75.0)

        # 3. New/Untrusted Device Anomaly
        if feature_row.get("is_new_device", 0) == 1:
            alerts.append({
                "alert_type": "DEVICE_ANOMALY",
                "severity": "MEDIUM",
                "message": "Transaction initiated from previously unseen device",
            })
            severity_scores.append(50.0)

        # 4. Foreign / High Risk Country
        if feature_row.get("is_high_risk_country", 0) == 1:
            alerts.append({
                "alert_type": "GEOGRAPHIC_RISK",
                "severity": "HIGH",
                "message": "Originating from high-risk jurisdiction",
            })
            severity_scores.append(85.0)

        # 5. Flash Crash / Liquidity Drain Scenario
        if liquidity_state:
            res_ratio = liquidity_state.get("reserve_utilization", 0.0)
            if res_ratio > 0.80:
                alerts.append({
                    "alert_type": "RESERVE_DRAIN",
                    "severity": "CRITICAL",
                    "message": f"Bank reserve utilization critical: {res_ratio*100:.1f}%",
                })
                severity_scores.append(95.0)

        # Calculate composite IDS alert score [0, 100]
        if severity_scores:
            ids_score = float(np.max(severity_scores))
        else:
            ids_score = 0.0

        return {
            "ids_score": round(ids_score, 2),
            "alerts": alerts,
            "has_alerts": len(alerts) > 0,
        }
