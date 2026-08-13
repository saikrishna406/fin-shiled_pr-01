"""PaySim Dataset Adapter for Fin-Shield Analytics.

Converts PaySim synthetic mobile money transaction datasets (PS_20161101120007.csv)
into the standard Fin-Shield Analytics transaction schema.
"""

from datetime import datetime, timedelta
import os
from typing import Optional
import pandas as pd


class PaySimAdapter:
    """Adapts PaySim mobile money dataset to Fin-Shield transaction schema."""

    @staticmethod
    def convert(paysim_df: pd.DataFrame, base_date: datetime = None) -> pd.DataFrame:
        """Convert PaySim DataFrame to Fin-Shield standard transaction format."""
        if base_date is None:
            base_date = datetime(2026, 8, 1, 0, 0, 0)

        records = []
        
        # Determine step to timestamp conversion (1 step = 1 hour)
        for idx, row in paysim_df.iterrows():
            step_hours = int(row.get("step", 1))
            tx_time = base_date + timedelta(hours=step_hours - 1, seconds=idx % 3600)

            sender = str(row.get("nameOrig", f"CUST_{idx:05d}"))
            receiver = str(row.get("nameDest", f"DEST_{idx:05d}"))
            amount = float(row.get("amount", 0.0))
            tx_type = str(row.get("type", "TRANSFER")).upper()
            is_fraud = int(row.get("isFraud", 0))

            # Infer merchant vs customer
            merchant_id = receiver if receiver.startswith("M") else None

            # Deterministic Bank Mapping based on sender/receiver hash
            sender_bank_idx = abs(hash(sender)) % 3
            receiver_bank_idx = abs(hash(receiver)) % 3
            banks = ["BANK_A", "BANK_B", "BANK_C"]
            sender_bank = banks[sender_bank_idx]
            receiver_bank = banks[receiver_bank_idx]

            # Device ID proxy
            device_id = f"DEV_{abs(hash(sender)) % 10000000:08d}"

            records.append({
                "transaction_id": f"PAYSIM_{idx+1:07d}",
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": amount,
                "currency": "USD",
                "timestamp": tx_time.isoformat(),
                "transaction_type": tx_type,
                "country": "US",
                "device_id": device_id,
                "merchant_id": merchant_id,
                "bank_id": sender_bank,
                "counterparty_bank_id": receiver_bank,
                "fraud_label": is_fraud,
            })

        return pd.DataFrame(records)

    @classmethod
    def load_and_convert(cls, csv_path: str, output_path: str = "datasets/paysim_converted.csv", max_rows: Optional[int] = None) -> str:
        """Load PaySim CSV file, convert to Fin-Shield schema, and save to output_path."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"PaySim file not found at: {csv_path}")

        df_raw = pd.read_csv(csv_path, nrows=max_rows)
        df_converted = cls.convert(df_raw)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_converted.to_csv(output_path, index=False)
        return output_path
