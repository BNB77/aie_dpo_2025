"""
Подготовка датасета кредитного скоринга на основе открытого датасета
UCI "Default of Credit Card Clients" (Yeh, I-C. & Lien, C.-H., 2009).

Источник: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Сырой файл лежит в data/raw/uci_credit_card_default_raw.csv (30000 строк,
скачан и сконвертирован из официального .xls один раз, без персональных
данных — только обезличенный ID клиента).

Скрипт делает очистку и feature engineering и сохраняет:
- data/credit_data_full.csv — полный обработанный датасет (30000 строк)
- data/credit_data_sample.csv — демо-выборка (300 строк) для быстрого запуска
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "uci_credit_card_default_raw.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "data"

PAY_COLS = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EDUCATION: документированные коды 1-4, встречаются 0/5/6 - сворачиваем в "others" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: документированные коды 1-3, встречается 0 - сворачиваем в "others" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["max_delay"] = df[PAY_COLS].max(axis=1)
    df["mean_delay"] = df[PAY_COLS].mean(axis=1)
    df["n_months_delayed"] = (df[PAY_COLS] > 0).sum(axis=1)

    df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)
    df["avg_pay_amt"] = df[PAYAMT_COLS].mean(axis=1)

    total_bill = df[BILL_COLS].sum(axis=1)
    total_pay = df[PAYAMT_COLS].sum(axis=1)
    # отношение платежей к счетам может уходить в десятки тысяч при почти нулевом
    # суммарном счёте (деление на ~1) - обрезаем выбросы, иначе модель будет
    # переоценивать важность этого признака из-за пары экстремальных строк
    df["payment_to_bill_ratio"] = (total_pay / (total_bill.abs() + 1.0)).clip(upper=5.0)

    df["credit_utilization"] = (df["avg_bill_amt"] / df["LIMIT_BAL"].replace(0, np.nan)).fillna(0)
    df["credit_utilization"] = df["credit_utilization"].clip(lower=-5, upper=5)

    # тренд задолженности: положительный - бил растёт со временем, отрицательный - снижается
    months = np.arange(6)
    bills = df[BILL_COLS].to_numpy()
    bill_mean = bills.mean(axis=1, keepdims=True)
    months_mean = months.mean()
    cov = ((bills - bill_mean) * (months - months_mean)).sum(axis=1)
    var = ((months - months_mean) ** 2).sum()
    df["bill_trend"] = cov / var

    return df


FEATURE_COLUMNS = [
    "LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE",
    "max_delay", "mean_delay", "n_months_delayed",
    "avg_bill_amt", "avg_pay_amt", "payment_to_bill_ratio",
    "credit_utilization", "bill_trend",
]
TARGET_COLUMN = "default"


def build_dataset() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    raw = clean_raw(raw)
    raw = engineer_features(raw)
    raw = raw.rename(columns={"default payment next month": TARGET_COLUMN})
    return raw[FEATURE_COLUMNS + [TARGET_COLUMN]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()

    full_path = OUT_DIR / "credit_data_full.csv"
    df.to_csv(full_path, index=False)

    sample_path = OUT_DIR / "credit_data_sample.csv"
    df.sample(n=300, random_state=SEED).to_csv(sample_path, index=False)

    print(f"Сохранено: {full_path} ({len(df)} строк)")
    print(f"Сохранена демо-выборка: {sample_path} (300 строк)")
    print(f"Доля default=1: {df[TARGET_COLUMN].mean():.4f}")


if __name__ == "__main__":
    main()
