"""Scenario 02: Domain-driven feature engineering.

This script demonstrates how subject-matter knowledge creates stronger signals
than simply feeding raw fields to an algorithm.
"""
import numpy as np
from common import load_data, OUTPUT_DIR


def engineer_features(df):
    out = df.copy()
    out["log_trade_amount"] = np.log1p(out["trade_amount"])
    out["high_value_trade"] = (out["trade_amount"] >= out["trade_amount"].quantile(0.95)).astype(int)
    out["late_confirmation"] = (out["confirmation_delay_minutes"] > 60).astype(int)
    out["late_affirmation"] = (out["affirmation_delay_minutes"] > 60).astype(int)
    out["complex_matching"] = (out["matching_attempts"] >= 3).astype(int)
    out["counterparty_risk_band"] = np.select(
        [out["prior_fail_rate_30d"] >= 0.12, out["prior_fail_rate_30d"] >= 0.06],
        ["HIGH", "MEDIUM"], default="LOW")
    out["time_pressure"] = (out["hours_until_cutoff"] < 8).astype(int)
    out["operational_pressure_score"] = (
        2*out["late_confirmation"] + out["late_affirmation"] + out["complex_matching"]
        + 2*(out["ssi_status"] != "VALID").astype(int) + out["time_pressure"]
    )
    return out


def main():
    df = engineer_features(load_data())
    cols = ["trade_id", "log_trade_amount", "high_value_trade", "late_confirmation",
            "complex_matching", "counterparty_risk_band", "time_pressure",
            "operational_pressure_score", "settlement_failed"]
    df[cols].to_csv(OUTPUT_DIR / "02_engineered_feature_sample.csv", index=False)
    print(df[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
