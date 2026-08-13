"""T+0 Atomic Settlement State Machine for Fin-Shield Analytics.

Executes atomic state transitions (CREATED -> RISK_CHECKED -> APPROVED/BUFFERED -> SETTLED / REJECTED)
and guarantees all-or-nothing settlement finality.
"""

from enum import Enum
import time
from typing import Dict, Any


class SettlementState(str, Enum):
    CREATED = "CREATED"
    RISK_CHECKED = "RISK_CHECKED"
    APPROVED = "APPROVED"
    BUFFERED = "BUFFERED"
    NETTED = "NETTED"
    SETTLED = "SETTLED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class AtomicSettlementEngine:
    """Orchestrates T+0 atomic settlement state machine."""

    def __init__(self, banking_simulator: Any, ledger_adapter: Any):
        self.banking_simulator = banking_simulator
        self.ledger_adapter = ledger_adapter

    def process_transaction(
        self,
        tx: Dict[str, Any],
        risk_result: Dict[str, Any],
        liquidity_engine: Any,
    ) -> Dict[str, Any]:
        """Execute atomic settlement workflow for transaction."""
        tx_id = tx["transaction_id"]
        sender_id = tx["sender_id"]
        receiver_id = tx["receiver_id"]
        amount = float(tx["amount"])

        # 1. State: CREATED -> RISK_CHECKED
        state = SettlementState.RISK_CHECKED
        route_action = risk_result["route_action"]
        start_time = time.perf_counter()

        if route_action == "EXPRESS_ROUTE":
            # Direct Atomic Settlement
            sender_acc = self.banking_simulator.get_or_create_account(sender_id, tx["bank_id"])
            receiver_acc = self.banking_simulator.get_or_create_account(receiver_id, tx["counterparty_bank_id"])

            if sender_acc.available_balance >= amount:
                success = self.banking_simulator.execute_transfer(sender_id, receiver_id, amount, was_locked=False)
                if success:
                    state = SettlementState.SETTLED
                    liquidity_engine.record_outflow(amount)
                    settlement_msg = "Express Atomic T+0 Settlement Completed"
                else:
                    state = SettlementState.REJECTED
                    settlement_msg = "Transfer failed in banking core"
            else:
                state = SettlementState.REJECTED
                settlement_msg = "Insufficient customer balance"

        elif route_action in ["DYNAMIC_BUFFER", "DYNAMIC_BUFFER_REVIEW"]:
            # Route to dynamic buffer pool
            sender_acc = self.banking_simulator.get_or_create_account(sender_id, tx["bank_id"])
            locked = self.banking_simulator.lock_funds(sender_id, amount)

            if locked:
                state = SettlementState.BUFFERED
                liquidity_engine.lock_buffer_liquidity(amount)
                settlement_msg = "Funds locked in Dynamic Buffer Pool for VSLC netting"
            else:
                state = SettlementState.REJECTED
                settlement_msg = "Insufficient funds for buffer lock"

        else:
            # CRITICAL / REJECT / CIRCUIT_BREAKER
            state = SettlementState.REJECTED
            settlement_msg = "Rejected by Fin-Shield Risk Circuit Breaker"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        settlement_record = {
            "transaction_id": tx_id,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "amount": amount,
            "status": state.value,
            "risk_score": risk_result["final_risk_score"],
            "route_action": route_action,
            "settlement_time_ms": round(elapsed_ms, 3),
            "message": settlement_msg,
        }

        # Write immutable record to permissioned ledger
        if self.ledger_adapter is not None:
            self.ledger_adapter.record_event(settlement_record)

        return settlement_record
