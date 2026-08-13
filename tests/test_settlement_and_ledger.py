from finshield.blockchain.ledger import PermissionedLedger

def test_permissioned_ledger_integrity():
    ledger = PermissionedLedger()
    ledger.record_event({"tx_id": "TX_001", "status": "SETTLED"})
    ledger.record_event({"tx_id": "TX_002", "status": "SETTLED"})

    res = ledger.verify_integrity()
    assert res["status"] == "PASS"
    assert res["blocks_verified"] == 3

    # Tamper with block payload
    ledger.chain[1].payload["status"] = "TAMPERED"
    tamper_res = ledger.verify_integrity()
    assert tamper_res["status"] == "FAIL"
