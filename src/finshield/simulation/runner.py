"""End-to-End Simulation Runner for Fin-Shield Analytics.

Executes streaming banking transactions through feature extraction, ML inference,
Behavioral IDS, Risk Scoring, Banking Core, Liquidity Engine, VSLC Netting,
Atomic T+0 Settlement, and Permissioned Ledger logging.
"""

from typing import Dict, List, Any
import pandas as pd
from finshield.features.extractor import FeatureExtractor, get_feature_columns
from finshield.ids.engine import BehavioralIDS
from finshield.risk.engine import RiskEngine
from finshield.banking.simulator import BankingCoreSimulator
from finshield.liquidity.engine import LiquidityEngine
from finshield.vslc.netting import VSLCNettingEngine
from finshield.settlement.engine import AtomicSettlementEngine
from finshield.blockchain.ledger import PermissionedLedger
from finshield.simulation.circuit_breaker import CircuitBreaker


class SimulationRunner:
    """Executes end-to-end Fin-Shield Analytics research simulation."""

    def __init__(self, models: Dict[str, Any] = None):
        self.models = models or {}
        self.feature_extractor = FeatureExtractor()
        self.ids = BehavioralIDS()
        self.risk_engine = RiskEngine()
        self.banking_sim = BankingCoreSimulator()
        self.liquidity_engine = LiquidityEngine()
        self.vslc_engine = VSLCNettingEngine()
        self.ledger = PermissionedLedger()
        self.settlement_engine = AtomicSettlementEngine(self.banking_sim, self.ledger)
        self.circuit_breaker = CircuitBreaker()

    def run_simulation(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete end-to-end simulation across dataset."""
        feature_df = self.feature_extractor.fit_transform(raw_df)
        feature_cols = get_feature_columns()

        xgb_model = self.models.get("xgboost")
        iso_model = self.models.get("isolation_forest")

        settlement_results = []
        buffered_txs = []
        recent_alerts = []

        for idx, row in feature_df.iterrows():
            raw_tx = raw_df.iloc[idx].to_dict()
            feat_row = row.to_dict()
            X_sample = pd.DataFrame([feat_row])[feature_cols]

            # 1. Model Predictions
            if xgb_model is not None:
                xgb_prob = float(xgb_model.predict_proba(X_sample)[0, 1])
            else:
                xgb_prob = 0.05 if feat_row.get("amount_zscore", 0) < 2.0 else 0.85

            if iso_model is not None:
                score = -float(iso_model.decision_function(X_sample)[0])
                iso_score = max(0.0, min(1.0, (score + 0.5)))
            else:
                iso_score = 0.1 if feat_row.get("is_new_device", 0) == 0 else 0.7

            # 2. Behavioral IDS Inspection
            liq_state = self.liquidity_engine.get_state()
            ids_res = self.ids.analyze_transaction(feat_row, liq_state)
            if ids_res["has_alerts"]:
                recent_alerts.append(1)

            # 3. Risk Engine Scoring & Decision
            risk_res = self.risk_engine.compute_risk_score(
                xgb_prob=xgb_prob,
                iso_anomaly_score=iso_score,
                ids_result=ids_res,
                liquidity_stress_score=liq_state["reserve_utilization"],
            )

            # 4. Circuit Breaker Check
            cb_status = self.circuit_breaker.evaluate_system_health(
                recent_alert_count=len(recent_alerts),
                window_size=max(1, idx + 1),
                current_buffer_depth=len(buffered_txs),
                last_latency_ms=1.5,
            )

            if cb_status["is_triggered"] and risk_res["risk_band"] == "HIGH":
                risk_res["route_action"] = "REJECT_OR_CIRCUIT_BREAKER"

            # 5. Settlement Execution
            settle_res = self.settlement_engine.process_transaction(
                tx=raw_tx,
                risk_result=risk_res,
                liquidity_engine=self.liquidity_engine,
            )

            if settle_res["status"] == "BUFFERED":
                buffered_txs.append(raw_tx)
                self.vslc_engine.add_to_buffer(raw_tx)

            settlement_results.append(settle_res)

        # 6. Run VSLC Netting on Buffered Obligations
        netting_res = self.vslc_engine.run_multilateral_netting()

        # 7. Verify Permissioned Ledger Hash-Chain
        ledger_verification = self.ledger.verify_integrity()

        return {
            "total_transactions_processed": len(raw_df),
            "settlement_records": settlement_results,
            "liquidity_final_state": self.liquidity_engine.get_state(),
            "vslc_netting_summary": netting_res,
            "ledger_verification": ledger_verification,
            "circuit_breaker_triggers": self.circuit_breaker.trigger_count,
        }
