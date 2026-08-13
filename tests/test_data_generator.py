from finshield.data.generator import SyntheticDataGenerator

def test_synthetic_data_generator():
    gen = SyntheticDataGenerator(num_transactions=100, seed=42)
    df = gen.generate()
    assert len(df) == 100
    assert "transaction_id" in df.columns
    assert "fraud_label" in df.columns
    assert df["fraud_label"].sum() > 0
