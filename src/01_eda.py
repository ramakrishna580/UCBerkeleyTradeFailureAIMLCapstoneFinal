"""Scenario 01: Exploratory Data Analysis (EDA).

Business questions:
- How imbalanced is the failure target?
- Which business dimensions have elevated observed failure rates?
- Are failed trades different in size, timing, SSI status, or counterparty history?

The script saves publication-ready figures into outputs/.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import load_data, OUTPUT_DIR


def save_bar(series, title, ylabel, filename, rotation=0):
    ax = series.plot(kind="bar", figsize=(9, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(series.index.name or "")
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def main():
    df = load_data()
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(10))
    print("\nTarget distribution:\n", df["settlement_failed"].value_counts())
    print("Failure rate:", f"{df['settlement_failed'].mean():.2%}")

    save_bar(
        df["settlement_failed"].value_counts(normalize=True).sort_index(),
        "Settlement Outcome Distribution", "Share of Trades", "01_target_distribution.png")

    for col, fname in [
        ("asset_class", "02_failure_by_asset_class.png"),
        ("ssi_status", "03_failure_by_ssi.png"),
        ("market", "04_failure_by_market.png"),
    ]:
        rates = df.groupby(col)["settlement_failed"].mean().sort_values(ascending=False)
        save_bar(rates, f"Observed Failure Rate by {col.replace('_',' ').title()}", "Failure Rate", fname, rotation=25)

    # Bucket confirmation delay to show a monotonic operational-risk relationship.
    delay_bucket = pd.cut(df["confirmation_delay_minutes"], [-1, 15, 30, 60, 90, 10_000],
                          labels=["0-15", "15-30", "30-60", "60-90", "90+"])
    delay_rates = df.assign(delay_bucket=delay_bucket).groupby("delay_bucket", observed=True)["settlement_failed"].mean()
    save_bar(delay_rates, "Failure Rate by Confirmation Delay", "Failure Rate", "05_failure_by_confirmation_delay.png")

    # High-risk counterparty table for business interpretation.
    cp = df.groupby("counterparty_id").agg(
        trades=("trade_id", "count"), failures=("settlement_failed", "sum"),
        observed_fail_rate=("settlement_failed", "mean"),
        avg_prior_fail_rate=("prior_fail_rate_30d", "mean")
    ).query("trades >= 100").sort_values("observed_fail_rate", ascending=False)
    cp.head(20).to_csv(OUTPUT_DIR / "01_top_counterparty_risk.csv")
    print("\nTop counterparties by observed fail rate:\n", cp.head(10))

if __name__ == "__main__":
    main()
