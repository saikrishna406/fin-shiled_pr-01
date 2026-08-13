"""Synthetic Banking Transaction Generator for Fin-Shield Analytics.

Generates realistic banking transactions with customer profiles, merchant profiles,
device IDs, location context, amounts, interbank metadata, velocity markers, and ground-truth fraud labels.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """Generates synthetic banking transaction streams with controllable attack scenarios."""

    def __init__(
        self,
        num_transactions: int = 10000,
        fraud_ratio: float = 0.05,
        num_customers: int = 500,
        num_merchants: int = 50,
        seed: int = 42,
    ):
        self.num_transactions = num_transactions
        self.fraud_ratio = fraud_ratio
        self.num_customers = num_customers
        self.num_merchants = num_merchants
        self.seed = seed
        
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.banks = ["BANK_A", "BANK_B", "BANK_C"]
        self.countries = ["US", "UK", "DE", "FR", "JP", "SG", "CA"]
        self.high_risk_countries = ["XX", "YY", "ZZ"]
        self.transaction_types = ["TRANSFER", "MERCHANT", "CORPORATE", "CROSS_BANK"]

        # Generate customer profiles
        self.customers = [f"CUST_{i:04d}" for i in range(self.num_customers)]
        self.customer_banks = {
            cust: random.choice(self.banks) for cust in self.customers
        }
        self.customer_home_country = {
            cust: random.choice(self.countries) for cust in self.customers
        }
        self.customer_avg_amount = {
            cust: round(float(np.random.lognormal(mean=4.5, sigma=0.8)), 2)
            for cust in self.customers
        }
        self.customer_primary_device = {
            cust: f"DEV_{uuid.uuid4().hex[:8]}" for cust in self.customers
        }

        # Generate merchant profiles
        self.merchants = [f"MERCH_{i:03d}" for i in range(self.num_merchants)]
        self.merchant_risk_weight = {
            m: random.choice([0.01, 0.02, 0.05, 0.15, 0.30]) for m in self.merchants
        }

    def generate(self, start_time: datetime = None) -> pd.DataFrame:
        """Generate synthetic transaction dataset DataFrame."""
        if start_time is None:
            start_time = datetime(2026, 8, 1, 9, 0, 0)

        records: List[Dict[str, Any]] = []
        num_frauds = int(self.num_transactions * self.fraud_ratio)
        fraud_indices = set(random.sample(range(self.num_transactions), num_frauds))

        current_time = start_time
        for i in range(self.num_transactions):
            # Time progression with realistic inter-arrival times (1-10 seconds)
            time_delta_sec = max(0.1, np.random.exponential(scale=3.0))
            current_time += timedelta(seconds=time_delta_sec)

            sender_id = random.choice(self.customers)
            receiver_id = random.choice(self.customers)
            while receiver_id == sender_id:
                receiver_id = random.choice(self.customers)

            sender_bank = self.customer_banks[sender_id]
            receiver_bank = self.customer_banks[receiver_id]
            
            # Select transaction type
            if sender_bank != receiver_bank:
                tx_type = "CROSS_BANK"
            else:
                tx_type = random.choice(["TRANSFER", "MERCHANT", "CORPORATE"])

            merchant_id = random.choice(self.merchants) if tx_type == "MERCHANT" else None
            device_id = self.customer_primary_device[sender_id]
            country = self.customer_home_country[sender_id]

            is_fraud = 1 if i in fraud_indices else 0

            # Normal vs Fraudulent parameter skewing
            if is_fraud:
                # Fraudulent: higher amount, potential device anomaly, high risk country
                avg_amt = self.customer_avg_amount[sender_id]
                amount = round(float(avg_amt * np.random.uniform(3.5, 12.0)), 2)
                
                if random.random() < 0.6:
                    device_id = f"DEV_UNKNOWN_{uuid.uuid4().hex[:6]}"
                if random.random() < 0.4:
                    country = random.choice(self.high_risk_countries)
            else:
                # Legitimate: centered around customer baseline
                avg_amt = self.customer_avg_amount[sender_id]
                amount = round(max(1.0, float(np.random.normal(loc=avg_amt, scale=avg_amt * 0.3))), 2)
                
                if random.random() < 0.05:
                    device_id = f"DEV_NEW_{uuid.uuid4().hex[:6]}"

            records.append({
                "transaction_id": f"TX_{i+1:07d}",
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "amount": amount,
                "currency": "USD",
                "timestamp": current_time.isoformat(),
                "transaction_type": tx_type,
                "country": country,
                "device_id": device_id,
                "merchant_id": merchant_id,
                "bank_id": sender_bank,
                "counterparty_bank_id": receiver_bank,
                "fraud_label": is_fraud,
            })

        df = pd.DataFrame(records)
        return df


def generate_dataset_to_file(output_path: str, count: int = 10000, seed: int = 42) -> str:
    """Helper function to generate dataset and save to file."""
    generator = SyntheticDataGenerator(num_transactions=count, seed=seed)
    df = generator.generate()
    df.to_csv(output_path, index=False)
    return output_path
