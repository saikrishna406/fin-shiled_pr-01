"""Feature Extraction Engine for Fin-Shield Analytics.

Transforms raw transactions into numeric and contextual features for ML models and IDS logic.
Prevents data leakage by utilizing historical state windows up to current timestamp.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class FeatureExtractor:
    """Computes engineered features for raw banking transactions."""

    def __init__(self, high_risk_countries: List[str] = None):
        self.high_risk_countries = set(high_risk_countries or ["XX", "YY", "ZZ"])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit historical baselines and compute engineered feature table."""
        df = df.copy()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt").reset_index(drop=True)

        features = []

        # State tracking structures across time
        customer_history: Dict[str, List[Tuple[pd.Timestamp, float, str, str]]] = {}
        counterparty_pair_history: Dict[Tuple[str, str], int] = {}
        customer_devices: Dict[str, set] = {}
        merchant_counts: Dict[str, int] = {}

        for idx, row in df.iterrows():
            ts = row["timestamp_dt"]
            sender = row["sender_id"]
            receiver = row["receiver_id"]
            amount = float(row["amount"])
            device = row["device_id"]
            country = row["country"]
            merchant = row["merchant_id"]
            sender_bank = row["bank_id"]
            receiver_bank = row["counterparty_bank_id"]

            # Initialize customer history
            if sender not in customer_history:
                customer_history[sender] = []
                customer_devices[sender] = set()

            history = customer_history[sender]

            # 1. Amount Statistics
            past_amounts = [h[1] for h in history]
            if past_amounts:
                mean_amt = float(np.mean(past_amounts))
                std_amt = float(np.std(past_amounts)) if len(past_amounts) > 1 and np.std(past_amounts) > 0 else 1.0
                amt_dev = amount - mean_amt
                amt_zscore = amt_dev / std_amt
            else:
                mean_amt = amount
                amt_dev = 0.0
                amt_zscore = 0.0

            log_amt = float(np.log1p(amount))

            # 2. Velocity Statistics (1-minute & 5-minute windows)
            window_1m_start = ts - pd.Timedelta(minutes=1)
            window_5m_start = ts - pd.Timedelta(minutes=5)

            txs_1m = [h for h in history if h[0] >= window_1m_start]
            txs_5m = [h for h in history if h[0] >= window_5m_start]

            tx_count_1m = len(txs_1m)
            tx_count_5m = len(txs_5m)
            amount_sum_1m = sum(h[1] for h in txs_1m)
            burst_count = tx_count_1m

            # 3. Behavioral Features
            is_new_device = 1 if (history and device not in customer_devices[sender]) else 0
            cust_tx_count_total = len(history)

            # 4. Counterparty & Geography Features
            pair_key = (sender, receiver)
            prev_pair_count = counterparty_pair_history.get(pair_key, 0)
            is_cross_bank = 1 if sender_bank != receiver_bank else 0
            is_high_risk_country = 1 if country in self.high_risk_countries else 0

            # 5. Merchant Features
            if merchant and pd.notna(merchant):
                m_count = merchant_counts.get(merchant, 0)
                merchant_risk_score = min(0.5, m_count * 0.01)
            else:
                merchant_risk_score = 0.0

            feature_row = {
                "transaction_id": row["transaction_id"],
                "raw_amount": amount,
                "log_amount": log_amt,
                "amount_dev_from_mean": amt_dev,
                "amount_zscore": amt_zscore,
                "tx_count_1m": tx_count_1m,
                "tx_count_5m": tx_count_5m,
                "amount_sum_1m": amount_sum_1m,
                "burst_count": burst_count,
                "is_new_device": is_new_device,
                "cust_tx_count_total": cust_tx_count_total,
                "prev_pair_count": prev_pair_count,
                "is_cross_bank": is_cross_bank,
                "is_high_risk_country": is_high_risk_country,
                "merchant_risk_score": merchant_risk_score,
            }
            features.append(feature_row)

            # Update historical state for future rows
            history.append((ts, amount, device, country))
            customer_devices[sender].add(device)
            counterparty_pair_history[pair_key] = prev_pair_count + 1
            if merchant and pd.notna(merchant):
                merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1

        feature_df = pd.DataFrame(features)

        # Merge with target ground truth if present
        if "fraud_label" in df.columns:
            feature_df["fraud_label"] = df["fraud_label"].values

        return feature_df


def get_feature_columns() -> List[str]:
    """Returns the list of engineered feature names used for ML model input."""
    return [
        "raw_amount",
        "log_amount",
        "amount_dev_from_mean",
        "amount_zscore",
        "tx_count_1m",
        "tx_count_5m",
        "amount_sum_1m",
        "burst_count",
        "is_new_device",
        "cust_tx_count_total",
        "prev_pair_count",
        "is_cross_bank",
        "is_high_risk_country",
        "merchant_risk_score",
    ]
