import os
import yaml

def test_risk_config_exists_and_valid():
    config_path = os.path.join("configs", "risk.yaml")
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    assert "weights" in data
    assert "thresholds" in data
    weights_sum = sum(data["weights"].values())
    assert abs(weights_sum - 1.0) < 1e-5

def test_liquidity_config_exists_and_valid():
    config_path = os.path.join("configs", "liquidity.yaml")
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    assert "bank_reserves" in data
    assert "vslc_netting" in data

def test_experiments_config_exists_and_valid():
    config_path = os.path.join("configs", "experiments.yaml")
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    assert "reproducibility" in data
    assert data["reproducibility"]["random_seed"] == 42
