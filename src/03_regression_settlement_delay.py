"""Scenario 03: Regression - predict settlement delay hours.

This is a SECONDARY learning problem used to demonstrate regression curriculum.
To avoid a trivial target dominated by zero-delay successful trades, this example
focuses on trades that actually experienced a settlement failure.

Models: Linear Regression, Ridge, Random Forest Regressor.
Metrics: MAE, RMSE, R^2.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import load_data, temporal_split, get_X_y, build_preprocessor, OUTPUT_DIR, PRE_PREDICTION_FEATURES
from sklearn.pipeline import Pipeline


def main():
    df = load_data().query("settlement_failed == 1").copy()
    train, valid, test = temporal_split(df)
    X_train, y_train = train[PRE_PREDICTION_FEATURES], train["settlement_delay_hours"]
    X_test, y_test = test[PRE_PREDICTION_FEATURES], test["settlement_delay_hours"]

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1),
    }
    rows = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        })
    result = pd.DataFrame(rows).sort_values("MAE")
    print(result.to_string(index=False))
    result.to_csv(OUTPUT_DIR / "03_regression_results.csv", index=False)

if __name__ == "__main__":
    main()
