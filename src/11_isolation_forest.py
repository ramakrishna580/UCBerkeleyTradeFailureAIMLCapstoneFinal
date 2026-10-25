"""Scenario 11: Isolation Forest anomaly detection.

Purpose: detect unusual trades without using the failure label during training.
Afterward, compare anomaly groups with actual failure rates to judge whether
anomalous trades are operationally meaningful.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from common import load_data, OUTPUT_DIR

FEATURES = ["trade_amount", "confirmation_delay_minutes", "matching_attempts",
            "trade_amendments", "prior_fail_rate_30d", "security_liquidity_score",
            "market_volatility_index", "trade_size_ratio", "hours_until_cutoff"]


def main():
    df = load_data()
    X = StandardScaler().fit_transform(df[FEATURES])
    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=42, n_jobs=-1)
    raw = iso.fit_predict(X)  # -1 anomaly, +1 normal
    score = -iso.score_samples(X)  # larger = more unusual
    result = df[["trade_id", "settlement_failed"] + FEATURES].copy()
    result["is_anomaly"] = (raw == -1).astype(int)
    result["anomaly_score"] = score
    print(result.groupby("is_anomaly")["settlement_failed"].agg(["count","mean"]))
    result.nlargest(100, "anomaly_score").to_csv(OUTPUT_DIR / "11_top_anomalies.csv", index=False)

if __name__ == "__main__":
    main()
