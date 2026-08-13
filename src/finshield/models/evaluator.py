"""Model Evaluation Pipeline for Fin-Shield Analytics.

Evaluates trained ML models on an untouched test set.
Computes research metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC,
Confusion Matrix, FPR, FNR, Latency, and Throughput. Exports results.
"""

import json
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from finshield.features.extractor import get_feature_columns


class ModelEvaluator:
    """Evaluates models and generates research-ready reproducible evaluation metrics."""

    def __init__(self, feature_cols: list = None):
        self.feature_cols = feature_cols or get_feature_columns()

    def evaluate_model(self, model: Any, test_df: pd.DataFrame, is_isolation_forest: bool = False, scaler: Any = None) -> Dict[str, Any]:
        """Evaluate a single model on test set and measure performance + latency."""
        X_test = test_df[self.feature_cols].copy()
        y_test = test_df["fraud_label"].copy().values

        if scaler is not None:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        # Measure latency & throughput
        start_time = time.perf_counter()
        if is_isolation_forest:
            # Isolation forest returns 1 for inliers, -1 for anomalies
            raw_preds = model.predict(X_eval)
            y_pred = np.where(raw_preds == -1, 1, 0)
            scores = -model.decision_function(X_eval)
            # Min-max scale anomaly score to [0, 1]
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            y_pred = model.predict(X_eval)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_eval)[:, 1]
            else:
                y_prob = y_pred

        elapsed_sec = time.perf_counter() - start_time
        latency_ms = (elapsed_sec / len(X_test)) * 1000.0
        throughput_tps = len(X_test) / max(1e-6, elapsed_sec)

        # Standard Classification Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = 0.5

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "inference_latency_ms": round(latency_ms, 4),
            "throughput_tps": round(throughput_tps, 2),
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.tolist(),
        }

    def evaluate_all(self, models: Dict[str, Any], test_df: pd.DataFrame, scaler: Any = None) -> Dict[str, Any]:
        """Evaluate all models and format comparison summary."""
        results = {}
        for name, model in models.items():
            is_iso = (name == "isolation_forest")
            use_scaler = scaler if name == "logistic_regression" else None
            res = self.evaluate_model(model, test_df, is_isolation_forest=is_iso, scaler=use_scaler)
            results[name] = res
        return results
