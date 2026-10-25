"""Shared utilities for all capstone scenarios.

IMPORTANT DESIGN RULE
---------------------
The prediction task assumes a pre-settlement prediction cutoff. Only fields that
would be known by that cutoff are included in PRE_PREDICTION_FEATURES.
Post-outcome columns (failure_reason, failure_reason_description) are excluded
from predictive modeling to prevent target leakage.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "trade_settlement_synthetic.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "settlement_failed"
POST_OUTCOME_COLUMNS = [
    "settlement_failed", "settlement_delay_hours",
    "failure_reason", "failure_reason_description"
]
ID_DATE_COLUMNS = ["trade_id", "trade_date", "settlement_date"]

# We intentionally exclude raw free text from the standard tabular models.
# pre_settlement_note is handled separately in the NLP scenario.
PRE_PREDICTION_FEATURES = [
    "asset_class", "buy_sell", "currency", "market", "settlement_location",
    "counterparty_id", "custodian", "trade_amount", "quantity",
    "settlement_days", "cross_border", "month_end", "quarter_end",
    "market_holiday_mismatch", "ssi_status", "confirmation_delay_minutes",
    "allocation_delay_minutes", "affirmation_delay_minutes", "matching_attempts",
    "trade_amendments", "hours_until_cutoff", "prior_fail_rate_30d",
    "prior_fail_rate_90d", "prior_trade_count_30d", "security_liquidity_score",
    "market_volatility_index", "inventory_shortfall_indicator",
    "counterparty_avg_trade_amount", "trade_size_ratio"
]


def load_data() -> pd.DataFrame:
    """Load the synthetic dataset and parse its date fields."""
    df = pd.read_csv(DATA_PATH, parse_dates=["trade_date", "settlement_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15):
    """Split chronologically rather than randomly.

    Why? Random train/test splits can leak future market/counterparty behavior into
    the past. A temporal split more closely simulates production deployment.
    """
    n = len(df)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(), df.iloc[valid_end:].copy()


def get_X_y(df: pd.DataFrame):
    return df[PRE_PREDICTION_FEATURES].copy(), df[TARGET].astype(int).copy()


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Create reusable preprocessing for mixed financial tabular data."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical_cols),
        ],
        remainder="drop",
    )


def evaluate_classifier(model, X, y, label: str = "Model") -> dict:
    """Evaluate a classifier using rare-event metrics rather than accuracy alone."""
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)

    metrics = {
        "model": label,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
    }
    print("\n", label)
    print("=" * len(label))
    print(classification_report(y, pred, digits=3, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y, pred))
    print({k: round(v, 4) for k, v in metrics.items() if k != "model"})
    return metrics


def save_metrics(metrics: dict, filename: str) -> None:
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / filename, index=False)


def make_pipeline(estimator, X: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
        ("model", estimator),
    ])
