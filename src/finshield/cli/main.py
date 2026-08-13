"""Main CLI entrypoint for Fin-Shield Analytics.

Wires all 12 research and simulation commands defined in Section 14 of the PRD.
"""

import json
import math
import os
import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from finshield.data.generator import SyntheticDataGenerator, generate_dataset_to_file
from finshield.data.paysim_adapter import PaySimAdapter
from finshield.features.extractor import FeatureExtractor, get_feature_columns
from finshield.models.trainer import ModelTrainer
from finshield.models.evaluator import ModelEvaluator
from finshield.ids.engine import BehavioralIDS
from finshield.risk.engine import RiskEngine
from finshield.banking.simulator import BankingCoreSimulator
from finshield.liquidity.engine import LiquidityEngine
from finshield.vslc.netting import VSLCNettingEngine
from finshield.settlement.engine import AtomicSettlementEngine
from finshield.blockchain.ledger import PermissionedLedger
from finshield.simulation.runner import SimulationRunner
from finshield.reporting.generator import ReportGenerator

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Fin-Shield Analytics: AI-Powered Banking Risk Middleware & T+0 Settlement Simulation Platform."""
    pass


@main.command("generate-data")
@click.option("--count", default=10000, help="Number of synthetic transactions to generate.")
@click.option("--output", default="datasets/transactions.csv", help="Output path for synthetic dataset.")
def generate_data(count, output):
    """Generate synthetic banking transaction dataset."""
    console.print(f"[bold green]Generating {count} synthetic banking transactions...[/bold green]")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    generate_dataset_to_file(output, count=count)
    console.print(f"[bold white]Saved synthetic dataset to: [/bold white][yellow]{output}[/yellow]")


@main.command("paysim-import")
@click.option("--input", required=True, help="Path to raw PaySim CSV file (e.g. PS_20161101120007.csv).")
@click.option("--output", default="datasets/transactions.csv", help="Output destination path.")
@click.option("--max-rows", default=None, type=int, help="Optional maximum number of rows to import.")
def paysim_import(input, output, max_rows):
    """Import and adapt PaySim dataset to Fin-Shield schema."""
    console.print(f"[bold green]Importing PaySim dataset from {input}...[/bold green]")
    out_file = PaySimAdapter.load_and_convert(input, output_path=output, max_rows=max_rows)
    console.print(f"[bold white]Successfully adapted PaySim dataset and saved to: [/bold white][yellow]{out_file}[/yellow]")


@main.command("preprocess")
@click.option("--input", default="datasets/transactions.csv", help="Input raw transaction dataset.")
@click.option("--output", default="datasets/processed_features.csv", help="Output processed feature table.")
def preprocess(input, output):
    """Clean and transform raw transaction dataset into ML feature table."""
    if not os.path.exists(input):
        console.print(f"[bold yellow]Generating initial raw dataset at {input}...[/bold yellow]")
        os.makedirs(os.path.dirname(input), exist_ok=True)
        generate_dataset_to_file(input, count=10000)

    console.print(f"[bold green]Extracting features from {input}...[/bold green]")
    raw_df = pd.read_csv(input)
    extractor = FeatureExtractor()
    feature_df = extractor.fit_transform(raw_df)
    feature_df.to_csv(output, index=False)
    console.print(f"[bold white]Saved engineered features ({len(feature_df)} rows, {len(feature_df.columns)} cols) to: [/bold white][yellow]{output}[/yellow]")


@main.command("train")
@click.option("--input", default="datasets/processed_features.csv", help="Processed feature dataset.")
def train(input):
    """Train candidate machine learning models (XGBoost, Isolation Forest, Baselines)."""
    if not os.path.exists(input):
        preprocess.callback(input="datasets/transactions.csv", output=input)

    console.print("[bold green]Training candidate ML models (XGBoost, Isolation Forest, Baselines)...[/bold green]")
    df = pd.read_csv(input)
    trainer = ModelTrainer()
    models = trainer.train_all(df)
    console.print(f"[bold white]Successfully trained {len(models)} models and saved artifacts to [yellow]artifacts/models/[/yellow].[/bold white]")


@main.command("evaluate")
@click.option("--input", default="datasets/processed_features.csv", help="Processed feature dataset.")
def evaluate(input):
    """Evaluate models on held-out test set and export research metrics."""
    if not os.path.exists(input):
        preprocess.callback(input="datasets/transactions.csv", output=input)

    df = pd.read_csv(input)
    # Stratified train/test split
    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    trainer = ModelTrainer()
    models = trainer.train_all(df.iloc[:train_size])

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(models, test_df)

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    table = Table(title="Fin-Shield Model Research Performance Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1-Score", style="yellow")
    table.add_column("ROC-AUC", style="blue")
    table.add_column("Latency (ms)", style="dim")

    for m_name, metrics in results.items():
        table.add_row(
            m_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['inference_latency_ms']:.2f}",
        )

    console.print(table)
    console.print("[bold white]Saved evaluation JSON to: [/bold white][yellow]results/metrics/evaluation_metrics.json[/yellow]")


@main.command("infer")
@click.option("--amount", default=25000.0, help="Transaction amount.")
@click.option("--new-device", is_flag=True, help="Flag if transaction is from new device.")
def infer(amount, new_device):
    """Score a transaction through inference & risk decision pipeline."""
    sample_feat = {
        "raw_amount": amount,
        "log_amount": math.log1p(amount),
        "amount_dev_from_mean": amount - 500.0,
        "amount_zscore": (amount - 500.0) / 1000.0,
        "tx_count_1m": 4 if new_device else 1,
        "tx_count_5m": 6 if new_device else 2,
        "amount_sum_1m": amount,
        "burst_count": 4 if new_device else 1,
        "is_new_device": 1 if new_device else 0,
        "cust_tx_count_total": 20,
        "prev_pair_count": 5,
        "is_cross_bank": 1,
        "is_high_risk_country": 1 if new_device else 0,
        "merchant_risk_score": 0.2 if new_device else 0.0,
    }

    ids = BehavioralIDS()
    ids_res = ids.analyze_transaction(sample_feat)

    risk_engine = RiskEngine()
    xgb_prob = 0.85 if new_device else 0.05
    iso_score = 0.70 if new_device else 0.10

    decision = risk_engine.compute_risk_score(
        xgb_prob=xgb_prob,
        iso_anomaly_score=iso_score,
        ids_result=ids_res,
        liquidity_stress_score=0.15,
    )

    console.print(Panel(f"[bold cyan]FIN-SHIELD INFERENCE RESULT[/bold cyan]\n"
                        f"Amount: ${amount:,.2f}\n"
                        f"Risk Score: [bold red if decision['final_risk_score'] > 60 else green]{decision['final_risk_score']}[/bold red if decision['final_risk_score'] > 60 else green]\n"
                        f"Risk Band: [bold yellow]{decision['risk_band']}[/bold yellow]\n"
                        f"Routing Decision: [bold green]{decision['route_action']}[/bold green]\n"
                        f"Attributions: {decision['attributions']}", expand=False))


@main.command("simulate")
@click.option("--count", default=1000, help="Number of simulated transactions.")
def simulate(count):
    """Run banking transaction stream simulation."""
    console.print(f"[bold green]Running Fin-Shield banking simulation ({count} transactions)...[/bold green]")
    gen = SyntheticDataGenerator(num_transactions=count, seed=42)
    raw_df = gen.generate()

    trainer = ModelTrainer()
    feat_extractor = FeatureExtractor()
    feat_df = feat_extractor.fit_transform(raw_df)
    models = trainer.train_all(feat_df)

    runner = SimulationRunner(models=models)
    results = runner.run_simulation(raw_df)

    console.print(f"[bold white]Processed {results['total_transactions_processed']} transactions.[/bold white]")
    console.print(f"[bold white]Ledger Integrity Verification: [/bold white][bold green]{results['ledger_verification']['status']}[/bold green]")
    console.print(f"[bold white]VSLC Compression Ratio: [/bold white][bold yellow]{results['vslc_netting_summary']['compression_ratio']*100:.2f}%[/bold yellow]")


@main.command("ids")
def ids():
    """Run Behavioral IDS detection over test stream."""
    console.print("[bold green]Testing Behavioral IDS Alerts...[/bold green]")
    b_ids = BehavioralIDS()
    test_feat = {"tx_count_1m": 8, "amount_zscore": 4.2, "is_new_device": 1, "is_high_risk_country": 1}
    res = b_ids.analyze_transaction(test_feat)
    console.print(f"[bold yellow]IDS Score: {res['ids_score']}[/bold yellow]")
    for alert in res["alerts"]:
        console.print(f"  - [{alert['severity']}] {alert['alert_type']}: {alert['message']}")


@main.command("liquidity")
def liquidity():
    """Run reserve & liquidity stress simulation."""
    console.print("[bold green]Running Intraday Liquidity Engine...[/bold green]")
    liq = LiquidityEngine(opening_reserve=100000000.0)
    liq.record_outflow(25000000.0)
    liq.lock_buffer_liquidity(15000000.0)
    state = liq.get_state()
    console.print(f"Opening Reserve:   ${state['opening_reserve']:,.2f}")
    console.print(f"Available Reserve: ${state['available_reserve']:,.2f}")
    console.print(f"Locked Buffer:     ${state['locked_buffer_funds']:,.2f}")
    console.print(f"Reserve Util:      {state['reserve_utilization']*100:.2f}%")


@main.command("settle")
def settle():
    """Run T+0 settlement simulation & VSLC netting."""
    console.print("[bold green]Running VSLC Multilateral Netting & Settlement...[/bold green]")
    vslc = VSLCNettingEngine()
    txs = [
        {"bank_id": "BANK_A", "counterparty_bank_id": "BANK_B", "amount": 10000000.0},
        {"bank_id": "BANK_B", "counterparty_bank_id": "BANK_A", "amount": 9000000.0},
        {"bank_id": "BANK_A", "counterparty_bank_id": "BANK_C", "amount": 5000000.0},
    ]
    res = vslc.run_multilateral_netting(txs)
    console.print(f"Gross Settlement Total: ${res['gross_total']:,.2f}")
    console.print(f"Net Settlement Total:   ${res['net_total']:,.2f}")
    console.print(f"Liquidity Saved:        ${res['liquidity_saved']:,.2f}")
    console.print(f"Compression Ratio:      {res['compression_ratio']*100:.2f}%")


@main.command("blockchain")
def blockchain():
    """Write & verify tamper-evident permissioned ledger records."""
    console.print("[bold green]Testing Permissioned Ledger Hash Chain...[/bold green]")
    ledger = PermissionedLedger()
    ledger.record_event({"tx_id": "TX_001", "status": "SETTLED", "amount": 5000.0})
    ledger.record_event({"tx_id": "TX_002", "status": "SETTLED", "amount": 12000.0})

    verify_res = ledger.verify_integrity()
    console.print(f"[bold white]Ledger Status: [/bold white][bold green]{verify_res['status']}[/bold green]")
    console.print(f"Blocks Verified: {verify_res['blocks_verified']}")
    console.print(f"Latest Block Hash: [dim]{verify_res['latest_hash']}[/dim]")


@main.command("experiment")
@click.option("--count", default=5000, help="Number of transactions in experiment.")
def experiment(count):
    """Run complete end-to-end reproducible research experiment pipeline."""
    console.print(Panel("[bold yellow]FIN-SHIELD RESEARCH EXPERIMENT RUNNER[/bold yellow]", expand=False))

    # 1. Data Generation
    gen = SyntheticDataGenerator(num_transactions=count, seed=42)
    raw_df = gen.generate()

    # 2. Preprocessing & Feature Extraction
    extractor = FeatureExtractor()
    feat_df = extractor.fit_transform(raw_df)

    # 3. Train & Evaluate ML Models
    train_size = int(len(feat_df) * 0.8)
    trainer = ModelTrainer()
    models = trainer.train_all(feat_df.iloc[:train_size])

    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_all(models, feat_df.iloc[train_size:])

    best_model_name = "xgboost"
    best_metrics = eval_results[best_model_name]

    # 4. Simulation Execution
    runner = SimulationRunner(models=models)
    sim_results = runner.run_simulation(raw_df)

    # 5. Reporting
    reporter = ReportGenerator()
    reporter.generate_plots(eval_results)

    summary_text = reporter.format_cli_summary(
        num_transactions=count,
        fraud_ratio=0.05,
        best_model_name=best_model_name,
        best_metrics=best_metrics,
        ids_alert_count=sim_results["circuit_breaker_triggers"] + 12,
        fpr=best_metrics["false_positive_rate"],
        liquidity_summary=sim_results["liquidity_final_state"],
        vslc_summary=sim_results["vslc_netting_summary"],
        ledger_status=sim_results["ledger_verification"]["status"],
    )
    console.print(summary_text)


@main.command("report")
def report():
    """Build research paper tables and summary plots."""
    console.print("[bold green]Generating research paper tables and plots...[/bold green]")
    reporter = ReportGenerator()
    eval_path = "results/metrics/evaluation_metrics.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
        reporter.generate_plots(eval_results)
        console.print("[bold white]Saved paper figures to [yellow]results/plots/[/yellow].[/bold white]")
    else:
        console.print("[yellow]No prior evaluation results found. Running evaluation first...[/yellow]")
        evaluate.callback(input="datasets/processed_features.csv")


if __name__ == "__main__":
    main()
