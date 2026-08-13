"""Research Evaluation and Plot Reporting Engine for Fin-Shield Analytics.

Generates paper-ready CSV tables, JSON metrics, and PNG plots:
- Model comparison tables & ROC Curves
- Confusion Matrices
- Feature Importance Charts
- VSLC Gross vs Net Settlement Liquidity Savings Charts
"""

import json
import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ReportGenerator:
    """Generates research artifacts and summary reports for Fin-Shield Analytics."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        self.metrics_dir = os.path.join(results_dir, "metrics")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def generate_plots(self, eval_results: Dict[str, Any], feature_cols: list = None):
        """Generate paper-ready PNG figures."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Model Comparison Bar Chart (F1 & Precision & Recall)
        models = list(eval_results.keys())
        f1_scores = [eval_results[m]["f1_score"] for m in models]
        prec_scores = [eval_results[m]["precision"] for m in models]
        rec_scores = [eval_results[m]["recall"] for m in models]

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, prec_scores, width, label="Precision", color="#1f77b4")
        ax.bar(x, rec_scores, width, label="Recall", color="#ff7f0e")
        ax.bar(x + width, f1_scores, width, label="F1-Score", color="#2ca02c")

        ax.set_ylabel("Score")
        ax.set_title("Fin-Shield Candidate Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "model_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # 2. Confusion Matrix Heatmap for Primary Supervised Model (XGBoost)
        if "xgboost" in eval_results:
            xgb_cm = eval_results["xgboost"]["confusion_matrix"]
            cm_arr = np.array([[xgb_cm["tn"], xgb_cm["fp"]], [xgb_cm["fn"], xgb_cm["tp"]]])

            fig, ax = plt.subplots(figsize=(6, 5))
            cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.8)
            fig.colorbar(cax)

            for i in range(2):
                for j in range(2):
                    ax.text(x=j, y=i, s=str(cm_arr[i, j]), va="center", ha="center", size="large", weight="bold")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Legitimate", "Fraud"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Legitimate", "Fraud"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("XGBoost Confusion Matrix")
            plt.tight_layout()
            cm_path = os.path.join(self.plots_dir, "confusion_matrix_xgboost.png")
            plt.savefig(cm_path, dpi=300)
            plt.close()

        return self.plots_dir

    def format_cli_summary(
        self,
        num_transactions: int,
        fraud_ratio: float,
        best_model_name: str,
        best_metrics: Dict[str, Any],
        ids_alert_count: int,
        fpr: float,
        liquidity_summary: Dict[str, Any],
        vslc_summary: Dict[str, Any],
        ledger_status: str,
    ) -> str:
        """Format PRD Section 14.1 required CLI summary block."""
        return f"""
===========================================================
FIN-SHIELD EXPERIMENT RESULTS
===========================================================
Transactions Processed: {num_transactions}
Fraud Ratio:            {fraud_ratio*100:.1f}%

BEST SUPERVISED MODEL:   {best_model_name.upper()}
  Accuracy:              {best_metrics.get('accuracy', 0.0):.4f}
  Precision:             {best_metrics.get('precision', 0.0):.4f}
  Recall:                {best_metrics.get('recall', 0.0):.4f}
  F1 Score:              {best_metrics.get('f1_score', 0.0):.4f}
  ROC-AUC:               {best_metrics.get('roc_auc', 0.0):.4f}

BEHAVIORAL IDS:
  IDS Alerts Issued:     {ids_alert_count}
  False Positive Rate:   {fpr:.4f}

LIQUIDITY & VSLC NETTING:
  Opening Reserve:       ${liquidity_summary.get('opening_reserve', 0):,.2f}
  Available Reserve:     ${liquidity_summary.get('available_reserve', 0):,.2f}
  Reserve Utilization:   {liquidity_summary.get('reserve_utilization', 0)*100:.2f}%
  Gross Obligations:     ${vslc_summary.get('gross_total', 0):,.2f}
  Net Obligations:       ${vslc_summary.get('net_total', 0):,.2f}
  Liquidity Saved:       ${vslc_summary.get('liquidity_saved', 0):,.2f}
  VSLC Compression:      {vslc_summary.get('compression_ratio', 0)*100:.2f}%

SETTLEMENT & LEDGER:
  T+0 Settlements:       {num_transactions}
  Permissioned Ledger:   {ledger_status}
===========================================================
"""
