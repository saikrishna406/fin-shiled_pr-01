from finshield.liquidity.engine import LiquidityEngine
from finshield.vslc.netting import VSLCNettingEngine

def test_liquidity_and_vslc_netting():
    liq = LiquidityEngine(opening_reserve=1000000.0)
    assert liq.available_reserve > 0

    liq.record_outflow(200000.0)
    assert liq.reserve_utilization > 0.0

    vslc = VSLCNettingEngine()
    txs = [
        {"bank_id": "BANK_A", "counterparty_bank_id": "BANK_B", "amount": 100.0},
        {"bank_id": "BANK_B", "counterparty_bank_id": "BANK_A", "amount": 90.0},
    ]
    res = vslc.run_multilateral_netting(txs)

    assert res["gross_total"] == 190.0
    assert res["net_total"] == 10.0
    assert res["liquidity_saved"] == 180.0
    assert res["compression_ratio"] > 0.90
