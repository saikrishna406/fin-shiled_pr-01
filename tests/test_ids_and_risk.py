from finshield.ids.engine import BehavioralIDS
from finshield.risk.engine import RiskEngine

def test_ids_and_risk_scoring():
    ids = BehavioralIDS()
    feat = {"tx_count_1m": 10, "amount_zscore": 4.5, "is_new_device": 1, "is_high_risk_country": 1}
    ids_res = ids.analyze_transaction(feat)

    assert ids_res["has_alerts"] is True
    assert ids_res["ids_score"] > 50.0

    risk_engine = RiskEngine()
    decision = risk_engine.compute_risk_score(
        xgb_prob=0.90,
        iso_anomaly_score=0.80,
        ids_result=ids_res,
        liquidity_stress_score=0.20,
    )

    assert decision["final_risk_score"] > 60.0
    assert decision["risk_band"] in ["HIGH", "CRITICAL"]
