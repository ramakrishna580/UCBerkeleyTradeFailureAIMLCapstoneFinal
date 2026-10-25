"""Scenario 13: CatBoost classifier using native categorical features.

Why this implementation is valuable:
CatBoost can directly consume categorical columns, which avoids potentially huge
one-hot matrices for fields such as counterparty, custodian, market and currency.
This is closer to how CatBoost would typically be evaluated for financial tabular data.
"""
from catboost import CatBoostClassifier
import pandas as pd
from common import load_data, temporal_split, get_X_y, evaluate_classifier, save_metrics


def prepare_catboost_frame(X: pd.DataFrame):
    X = X.copy()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    # CatBoost expects categorical values to be strings/integers and not NaN floats.
    for c in cat_cols:
        X[c] = X[c].fillna("UNKNOWN").astype(str)
    return X, cat_cols


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    X_train, cat_cols = prepare_catboost_frame(X_train)
    X_test, _ = prepare_catboost_frame(X_test)

    model = CatBoostClassifier(
        iterations=450,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        verbose=False,
        random_seed=42,
        cat_features=cat_cols,
    )
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test, "CatBoost (native categoricals)")
    save_metrics(metrics, "13_catboost_metrics.csv")

if __name__ == "__main__":
    main()
