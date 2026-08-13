"""Risk Scoring and Routing Policy Decision Engine for Fin-Shield Analytics.

Combines XGBoost supervised fraud probability, Isolation Forest anomaly score,
Behavioral IDS alerts, and bank liquidity stress into a normalized Fin-Shield Risk Score (0-100).
Determines transaction routing path: Express Route, Dynamic Buffer Pool, or Reject/Circuit Breaker.
"""

from typing import Dict, Any, List
import numpy as np


class RiskEngine:
    """Configurable risk scoring engine with explainability attributions."""

    def __init__(
        self,
        w_ml: float = 0.40,
        w_anomaly: float = 0.25,
        w_ids: float = 0.20,
        w_liquidity: float = 0.15,
        low_max: float = 39.0,
        med_max: float = 69.0,
        high_max: float = 89.0,
    ):
        self.w_ml = w_ml
        self.w_anomaly = w_anomaly
        self.w_ids = w_ids
        self.w_liquidity = w_liquidity

        self.low_max = low_max
        self.med_max = med_max
        self.high_max = high_max

    def compute_risk_score(
        self,
        xgb_prob: float,
        iso_anomaly_score: float,
        ids_result: Dict[str, Any],
        liquidity_stress_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute composite risk score and assign routing decision."""
        # Scale all inputs to [0, 100]
        ml_term = float(xgb_prob) * 100.0
        anomaly_term = float(iso_anomaly_score) * 100.0
        ids_term = float(ids_result.get("ids_score", 0.0))
        liq_term = float(liquidity_stress_score) * 100.0

        final_score = (
            self.w_ml * ml_term
            + self.w_anomaly * anomaly_term
            + self.w_ids * ids_term
            + self.w_liquidity * liq_term
        )

        final_score = float(np.clip(final_score, 0.0, 100.0))

        # Assign Risk Band and Routing Action
        if final_score <= self.low_max:
            risk_band = "LOW"
            route_action = "EXPRESS_ROUTE"
        elif final_score <= self.med_max:
            risk_band = "MEDIUM"
            route_action = "DYNAMIC_BUFFER"
        elif final_score <= self.high_max:
            risk_band = "HIGH"
            route_action = "DYNAMIC_BUFFER_REVIEW"
        else:
            risk_band = "CRITICAL"
            route_action = "REJECT_OR_CIRCUIT_BREAKER"

        # Explainability attributions
        attributions = {
            "ml_supervised_contribution": round(self.w_ml * ml_term, 2),
            "ml_anomaly_contribution": round(self.w_anomaly * anomaly_term, 2),
            "ids_behavior_contribution": round(self.w_ids * ids_term, 2),
            "liquidity_stress_contribution": round(self.w_liquidity * liq_term, 2),
        }

        return {
            "final_risk_score": round(final_score, 2),
            "risk_band": risk_band,
            "route_action": route_action,
            "attributions": attributions,
            "ids_alerts": ids_result.get("alerts", []),
        }
