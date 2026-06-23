import pandas as pd

from src.data.prepare_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_dataset,
    clean_raw,
    engineer_features,
)
from src.data.preprocessing import split_features_target


def test_clean_raw_removes_undocumented_education_codes():
    df = pd.DataFrame({"EDUCATION": [0, 1, 2, 5, 6], "MARRIAGE": [0, 1, 2, 3, 1]})
    cleaned = clean_raw(df)
    assert set(cleaned["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(cleaned["MARRIAGE"].unique()) <= {1, 2, 3}


def test_engineer_features_adds_expected_columns():
    pay_cols = {f"PAY_{i}": [0, 1] for i in [0, 2, 3, 4, 5, 6]}
    bill_cols = {f"BILL_AMT{i}": [1000 * i, 1100 * i] for i in range(1, 7)}
    payamt_cols = {f"PAY_AMT{i}": [500, 600] for i in range(1, 7)}
    df = pd.DataFrame({**pay_cols, **bill_cols, **payamt_cols, "LIMIT_BAL": [50000, 100000]})

    out = engineer_features(df)
    for col in ["max_delay", "mean_delay", "n_months_delayed", "avg_bill_amt",
                "avg_pay_amt", "payment_to_bill_ratio", "credit_utilization", "bill_trend"]:
        assert col in out.columns
    assert len(out) == 2


def test_build_dataset_shape_and_target():
    df = build_dataset()
    assert TARGET_COLUMN in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert set(df[TARGET_COLUMN].unique()) <= {0, 1}
    assert len(df) == 30000


def test_split_features_target():
    df = build_dataset().sample(n=50, random_state=1)
    X, y = split_features_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == 50
