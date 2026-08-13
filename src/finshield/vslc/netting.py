"""Velocity Staging and Continuous Multilateral Netting (VSLC) Engine for Fin-Shield Analytics.

Stage 1: Velocity Staging (Buffers volatile/high-risk transaction streams).
Stage 2: Continuous Multilateral Netting (Computes net interbank obligations,
reducing gross settlement liquidity required for T+0 settlement).
"""

from typing import Dict, List, Any
import numpy as np


class VSLCNettingEngine:
    """Multilateral netting engine computing compressed settlement obligations."""

    def __init__(self, netting_window_ms: int = 500):
        self.netting_window_ms = netting_window_ms
        self.buffered_obligations: List[Dict[str, Any]] = []

    def add_to_buffer(self, tx: Dict[str, Any]):
        """Add transaction to dynamic buffer pool for netting."""
        self.buffered_obligations.append(tx)

    def execute_netting() -> Dict[str, Any]:
        """Compute multilateral netting over buffered obligations."""

    def run_multilateral_netting(self, transactions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute gross vs net settlement values and compression ratio."""
        txs = transactions if transactions is not None else self.buffered_obligations
        if not txs:
            return {
                "gross_total": 0.0,
                "net_total": 0.0,
                "liquidity_saved": 0.0,
                "compression_ratio": 0.0,
                "participant_positions": {},
                "netted_settlements": [],
            }

        # Track gross outgoing and incoming per bank
        bank_outgoing: Dict[str, float] = {}
        bank_incoming: Dict[str, float] = {}

        for tx in txs:
            s_bank = tx["bank_id"]
            r_bank = tx["counterparty_bank_id"]
            amt = float(tx["amount"])

            bank_outgoing[s_bank] = bank_outgoing.get(s_bank, 0.0) + amt
            bank_incoming[r_bank] = bank_incoming.get(r_bank, 0.0) + amt

        all_banks = set(list(bank_outgoing.keys()) + list(bank_incoming.keys()))
        net_positions: Dict[str, float] = {}

        for b in all_banks:
            outg = bank_outgoing.get(b, 0.0)
            incg = bank_incoming.get(b, 0.0)
            net_positions[b] = outg - incg

        gross_total = sum(bank_outgoing.values())
        net_total = sum(max(0.0, pos) for pos in net_positions.values())

        liquidity_saved = max(0.0, gross_total - net_total)
        compression_ratio = (
            float(1.0 - (net_total / gross_total)) if gross_total > 0 else 0.0
        )

        return {
            "gross_total": round(gross_total, 2),
            "net_total": round(net_total, 2),
            "liquidity_saved": round(liquidity_saved, 2),
            "compression_ratio": round(compression_ratio, 4),
            "net_positions": {b: round(pos, 2) for b, pos in net_positions.items()},
            "transaction_count": len(txs),
        }
