"""Banking Core Simulator for Fin-Shield Analytics.

Simulates accounts, account balances, interbank relationships, payment queue,
locked buffer pools, and core ledger execution state.
"""

from typing import Dict, List, Any, Optional


class Account:
    """Represents a simulated customer account."""

    def __init__(self, account_id: str, customer_id: str, bank_id: str, initial_balance: float = 50000.0):
        self.account_id = account_id
        self.customer_id = customer_id
        self.bank_id = bank_id
        self.balance = float(initial_balance)
        self.locked_balance = 0.0

    @property
    def available_balance(self) -> float:
        return self.balance - self.locked_balance


class BankingCoreSimulator:
    """Core banking simulator managing accounts and intraday state."""

    def __init__(self, banks: List[str] = None):
        self.banks = banks or ["BANK_A", "BANK_B", "BANK_C"]
        self.accounts: Dict[str, Account] = {}
        self.payment_queue: List[Dict[str, Any]] = []

    def get_or_create_account(self, customer_id: str, bank_id: str) -> Account:
        """Retrieve or register account for customer."""
        if customer_id not in self.accounts:
            self.accounts[customer_id] = Account(
                account_id=f"ACC_{customer_id}",
                customer_id=customer_id,
                bank_id=bank_id,
                initial_balance=100000.0,
            )
        return self.accounts[customer_id]

    def lock_funds(self, customer_id: str, amount: float) -> bool:
        """Lock customer funds in buffer pool."""
        acc = self.accounts.get(customer_id)
        if acc and acc.available_balance >= amount:
            acc.locked_balance += amount
            return True
        return False

    def release_locked_funds(self, customer_id: str, amount: float):
        """Release locked customer funds upon rejection or rollback."""
        acc = self.accounts.get(customer_id)
        if acc:
            acc.locked_balance = max(0.0, acc.locked_balance - amount)

    def execute_transfer(self, sender_id: str, receiver_id: str, amount: float, was_locked: bool = False) -> bool:
        """Execute atomic payment transfer between sender and receiver."""
        sender = self.accounts.get(sender_id)
        receiver = self.accounts.get(receiver_id)

        if not sender or not receiver:
            return False

        if was_locked:
            if sender.locked_balance >= amount:
                sender.locked_balance -= amount
                sender.balance -= amount
                receiver.balance += amount
                return True
            return False
        else:
            if sender.available_balance >= amount:
                sender.balance -= amount
                receiver.balance += amount
                return True
            return False
