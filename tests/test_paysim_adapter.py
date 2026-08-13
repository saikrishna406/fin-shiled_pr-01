import pandas as pd
from finshield.data.paysim_adapter import PaySimAdapter

def test_paysim_adapter():
    mock_paysim_data = {
        "step": [1, 1, 2],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [9839.64, 1864.28, 181.00],
        "nameOrig": ["C1231006815", "C1305486145", "C840083671"],
        "oldbalanceOrg": [170136.0, 21249.0, 181.0],
        "newbalanceOrig": [160296.36, 19384.72, 0.0],
        "nameDest": ["M1979787155", "C553264065", "C38997010"],
        "oldbalanceDest": [0.0, 0.0, 21182.0],
        "newbalanceDest": [0.0, 0.0, 0.0],
        "isFraud": [0, 1, 1],
        "isFlaggedFraud": [0, 0, 0],
    }

    df_raw = pd.DataFrame(mock_paysim_data)
    df_conv = PaySimAdapter.convert(df_raw)

    assert len(df_conv) == 3
    assert "transaction_id" in df_conv.columns
    assert "fraud_label" in df_conv.columns
    assert df_conv["fraud_label"].tolist() == [0, 1, 1]
    assert df_conv["merchant_id"].iloc[0] == "M1979787155"
