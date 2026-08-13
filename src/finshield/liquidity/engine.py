"""Intraday Liquidity Engine for Fin-Shield Analytics.

Tracks bank reserves, incoming/outgoing obligations, locked buffer liquidity,
reserve utilization rates, and projected reserve stress levels.
"""

from typing import Dict, Any


class LiquidityEngine:
    """Manages intraday liquidity reserves and projected liquidity stress."""

    def __init__(
        self,
        bank_id: str = "BANK_A",
        opening_reserve: float = 100000000.0,
        mandatory_reserve_rate: float = 0.10,
    ):
        self.bank_id = bank_id
        self.opening_reserve = float(opening_reserve)
        self.mandatory_reserve_hold = float(opening_reserve * mandatory_reserve_rate)

        self.confirmed_inflows = 0.0
        self.settled_outflows = 0.0
        self.locked_buffer_funds = 0.0

        self.expected_inflows = 0.0
        self.expected_outflows = 0.0

    @property
    def available_reserve(self) -> float:
        """Calculate current available reserve pool balance."""
        res = (
            self.opening_reserve
            + self.confirmed_inflows
            - self.settled_outflows
            - self.locked_buffer_funds
            - self.mandatory_reserve_hold
        )
        return max(0.0, res)

    @property
    def reserve_utilization(self) -> float:
        """Calculate reserve utilization fraction."""
        used = self.opening_reserve - self.available_reserve
        return max(0.0, min(1.0, used / self.opening_reserve))

    def get_projected_reserve(self, window_delta_sec: int = 300) -> float:
        """Compute projected reserve in t + delta window."""
        proj = (
            self.available_reserve
            + self.expected_inflows
            - self.expected_outflows
        )
        return max(0.0, proj)

    def record_outflow(self, amount: float):
        """Record settled outgoing payment obligation."""
        self.settled_outflows += amount

    def record_inflow(self, amount: float):
        """Record settled incoming payment obligation."""
        self.confirmed_inflows += amount

    def lock_buffer_liquidity(self, amount: float):
        """Lock liquidity in dynamic buffer pool."""
        self.locked_buffer_funds += amount

    def unlock_buffer_liquidity(self, amount: float):
        """Release locked buffer liquidity."""
        self.locked_buffer_funds = max(0.0, self.locked_buffer_funds - amount)

    def get_state(self) -> Dict[str, Any]:
        """Return comprehensive liquidity snapshot."""
        return {
            "bank_id": self.bank_id,
            "opening_reserve": self.opening_reserve,
            "available_reserve": round(self.available_reserve, 2),
            "reserve_utilization": round(self.reserve_utilization, 4),
            "locked_buffer_funds": round(self.locked_buffer_funds, 2),
            "confirmed_inflows": round(self.confirmed_inflows, 2),
            "settled_outflows": round(self.settled_outflows, 2),
            "projected_reserve": round(self.get_projected_reserve(), 2),
        }
