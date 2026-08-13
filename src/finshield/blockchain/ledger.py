"""Permissioned Ledger and Blockchain Adapter for Fin-Shield Analytics.

Implements an append-only, tamper-evident hash-chain ledger layer.
Each block hashes the payload, timestamp, state transition, and previous block hash.
"""

import hashlib
import json
import time
from typing import Dict, List, Any


class LedgerBlock:
    """Represents an immutable block in the permissioned settlement ledger."""

    def __init__(self, index: int, timestamp: str, payload: Dict[str, Any], previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.payload = payload
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 digest of block contents."""
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "payload": self.payload,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()


class PermissionedLedger:
    """Append-only tamper-evident blockchain ledger abstraction."""

    def __init__(self):
        self.chain: List[LedgerBlock] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis_block = LedgerBlock(
            index=0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            payload={"event": "GENESIS", "message": "Fin-Shield Permissioned Ledger Initialized"},
            previous_hash="0" * 64,
        )
        self.chain.append(genesis_block)

    def record_event(self, payload: Dict[str, Any]) -> LedgerBlock:
        """Append new settlement event to ledger hash chain."""
        prev_block = self.chain[-1]
        new_block = LedgerBlock(
            index=len(self.chain),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            payload=payload,
            previous_hash=prev_block.hash,
        )
        self.chain.append(new_block)
        return new_block

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify hash-chain validity across all recorded settlement blocks."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Recompute hash
            if current.hash != current.compute_hash():
                return {
                    "status": "FAIL",
                    "reason": f"Tampering detected at block {current.index}: Hash mismatch",
                    "block_index": current.index,
                }

            # Check chain link
            if current.previous_hash != previous.hash:
                return {
                    "status": "FAIL",
                    "reason": f"Tampering detected at block {current.index}: Previous hash link broken",
                    "block_index": current.index,
                }

        return {
            "status": "PASS",
            "blocks_verified": len(self.chain),
            "latest_hash": self.chain[-1].hash if self.chain else None,
        }
