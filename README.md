# Fin-Shield Analytics

**AI-Powered Banking Risk Middleware & T+0 Atomic Settlement Simulation Platform**

Fin-Shield Analytics is a research-first, CLI-first middleware and settlement orchestration prototype designed to evaluate how combining transaction-level ML risk scoring, behavioral Intrustion Detection System (IDS), liquidity-aware routing, and Velocity Staging Multilateral Netting (VSLC) can reduce unsafe settlement exposure while preserving low-latency processing and reducing gross liquidity required for simulated T+0 settlement.

## System Architecture

```
EXTERNAL ACTORS
Customers / Merchants / Corporate Clients / Other Banks
                   |
                   v
PAYMENT INITIATION
Mobile / API / Payment Gateway / Corporate System
                   |
                   v
SIMULATED BANK
Accounts -> Core Ledger -> Treasury -> Reserve -> Payment Queue
                   |
                   v
FIN-SHIELD MIDDLEWARE
Ingestion -> Feature Extraction -> ML + IDS -> Risk Engine
                   |
         +---------+----------+
         |                    |
     LOW RISK            HIGH / RISKY
         |                    |
    EXPRESS ROUTE       DYNAMIC BUFFER
         |                    |
         |                  VSLC
         |             (Netting/Compression)
         |                    |
         +---------+----------+
                   |
             LIQUIDITY CHECK
                   |
             DECISION ENGINE
                   |
         +---------+----------+
         |                    |
      APPROVE               REJECT
  SMART CONTRACT              |
  PERMISSIONED LEDGER         |
  ATOMIC T+0 SETTLEMENT       |
     RECEIVER BANK
```

## Quick Start (CLI)

```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows

# Install package in editable mode
pip install -e .

# Run CLI Commands
finshield generate-data
finshield preprocess
finshield train
finshield evaluate
finshield infer
finshield simulate
finshield ids
finshield liquidity
finshield settle
finshield experiment
finshield report
```

## Directory Structure

```
fin-shield/
├── configs/          # Experiment & Risk configurations
├── src/finshield/    # Core Python package
├── tests/            # Pytest test suite
├── datasets/         # Generated synthetic datasets
├── artifacts/        # Trained models & preprocessing pipelines
├── results/          # Experiment metrics, plots, and ledgers
└── docs/             # Documentation & PRD
```
