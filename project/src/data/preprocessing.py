"""Подготовка признаков для моделей кредитного скоринга."""
import pandas as pd

from src.data.prepare_data import FEATURE_COLUMNS, TARGET_COLUMN

__all__ = ["FEATURE_COLUMNS", "TARGET_COLUMN", "load_dataset", "split_features_target"]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
