"""ML Training Pipeline for Fin-Shield Analytics.

Trains baseline models (Logistic Regression, Decision Tree, Random Forest),
primary supervised classifier (XGBoost), and primary anomaly detector (Isolation Forest).
Saves trained artifacts with metadata.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from finshield.features.extractor import get_feature_columns


class ModelTrainer:
    """Trains and serializes candidate ML models for Fin-Shield Analytics."""

    def __init__(self, artifacts_dir: str = "artifacts/models"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.feature_cols = get_feature_columns()

    def train_all(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Train all candidate models on feature DataFrame."""
        X_train = train_df[self.feature_cols].copy()
        y_train = train_df["fraud_label"].copy()

        # Fit Scaler
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.feature_cols)
        joblib.dump(self.scaler, os.path.join(self.artifacts_dir, "scaler.joblib"))

        models = {}

        # 1. Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        lr.fit(X_scaled, y_train)
        models["logistic_regression"] = lr
        joblib.dump(lr, os.path.join(self.artifacts_dir, "logistic_regression.joblib"))

        # 2. Decision Tree
        dt = DecisionTreeClassifier(random_state=42, max_depth=6, class_weight="balanced")
        dt.fit(X_train, y_train)
        models["decision_tree"] = dt
        joblib.dump(dt, os.path.join(self.artifacts_dir, "decision_tree.joblib"))

        # 3. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
        rf.fit(X_train, y_train)
        models["random_forest"] = rf
        joblib.dump(rf, os.path.join(self.artifacts_dir, "random_forest.joblib"))

        # 4. XGBoost (Primary Supervised)
        scale_pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
        joblib.dump(xgb, os.path.join(self.artifacts_dir, "xgboost.joblib"))

        # 5. Isolation Forest (Primary Anomaly Detector)
        contamination = float(np.mean(y_train)) if np.mean(y_train) > 0 else 0.05
        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
        iso.fit(X_train)
        models["isolation_forest"] = iso
        joblib.dump(iso, os.path.join(self.artifacts_dir, "isolation_forest.joblib"))

        return models
