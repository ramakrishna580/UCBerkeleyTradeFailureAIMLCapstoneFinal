"""Generate a realistic synthetic trade-settlement dataset for the capstone.

The generator deliberately separates:
1) PRE-PREDICTION features: safe to use when predicting settlement failure.
2) POST-OUTCOME fields: only known after settlement, useful for diagnostics/NLP,
   but MUST NOT be used as predictors because they would create data leakage.

The generated data is synthetic and contains no confidential institutional data.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 25_000
OUT = Path(__file__).resolve().parents[1] / "data" / "trade_settlement_synthetic.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    start = pd.Timestamp("2023-01-02")
    dates = pd.bdate_range(start, periods=760)

    asset_classes = np.array(["Equity", "Corporate Bond", "Government Bond", "ETF"])
    currencies = np.array(["USD", "EUR", "GBP", "JPY", "CAD"])
    markets = np.array(["US", "UK", "EU", "JP", "CA"])
    settlement_locations = np.array(["DTC", "Euroclear", "Clearstream", "CREST", "JASDEC", "CDS"])
    counterparties = np.array([f"CP{n:03d}" for n in range(1, 81)])
    custodians = np.array([f"CUST{n:02d}" for n in range(1, 13)])

    trade_date = pd.to_datetime(rng.choice(dates, N_ROWS))
    asset_class = rng.choice(asset_classes, N_ROWS, p=[0.43, 0.27, 0.18, 0.12])
    currency = rng.choice(currencies, N_ROWS, p=[0.62, 0.14, 0.10, 0.08, 0.06])
    market = rng.choice(markets, N_ROWS, p=[0.60, 0.11, 0.13, 0.09, 0.07])
    counterparty_id = rng.choice(counterparties, N_ROWS)
    custodian = rng.choice(custodians, N_ROWS)
    buy_sell = rng.choice(["BUY", "SELL"], N_ROWS)

    # Trade amounts are right-skewed, like many financial transaction datasets.
    trade_amount = np.exp(rng.normal(np.log(1_200_000), 1.15, N_ROWS)).clip(10_000, 120_000_000)
    quantity = (trade_amount / rng.uniform(20, 220, N_ROWS)).astype(int).clip(1)

    # Settlement cycles: US-style T+1 is common after 2024; some bonds/cross-border remain more varied in synthetic data.
    settlement_days = np.where(
        (trade_date >= pd.Timestamp("2024-05-28")) & (market == "US"),
        rng.choice([1, 2], N_ROWS, p=[0.93, 0.07]),
        rng.choice([1, 2, 3], N_ROWS, p=[0.20, 0.72, 0.08]),
    )
    settlement_date = trade_date + pd.to_timedelta(settlement_days, unit="D")

    cross_border = (rng.random(N_ROWS) < 0.23).astype(int)
    month_end = trade_date.is_month_end.astype(int)
    quarter_end = trade_date.is_quarter_end.astype(int)
    market_holiday_mismatch = ((rng.random(N_ROWS) < (0.025 + 0.055 * cross_border))).astype(int)

    # Counterparty risk tiers create consistent but not deterministic historical behavior.
    cp_risk_map = {cp: rng.beta(1.8, 18.0) for cp in counterparties}
    cp_base = np.array([cp_risk_map[c] for c in counterparty_id])
    prior_fail_rate_30d = np.clip(cp_base + rng.normal(0, 0.018, N_ROWS), 0, 0.35)
    prior_fail_rate_90d = np.clip(cp_base * 0.85 + rng.normal(0, 0.012, N_ROWS), 0, 0.30)
    prior_trade_count_30d = rng.poisson(120, N_ROWS) + 5

    ssi_status = rng.choice(["VALID", "CHANGED", "MISSING"], N_ROWS, p=[0.92, 0.055, 0.025])
    confirmation_delay_minutes = rng.gamma(shape=2.0, scale=16.0, size=N_ROWS)
    allocation_delay_minutes = rng.gamma(shape=1.7, scale=12.0, size=N_ROWS)
    affirmation_delay_minutes = rng.gamma(shape=1.8, scale=14.0, size=N_ROWS)
    matching_attempts = np.clip(rng.poisson(0.55, N_ROWS) + 1, 1, 8)
    trade_amendments = np.clip(rng.poisson(0.18, N_ROWS), 0, 5)
    hours_until_cutoff = np.clip(rng.normal(17, 7, N_ROWS), 0.5, 48)
    security_liquidity_score = np.clip(rng.beta(4.2, 2.2, N_ROWS), 0.02, 0.99)
    market_volatility_index = np.clip(rng.normal(20, 7, N_ROWS), 8, 65)
    inventory_shortfall_indicator = ((rng.random(N_ROWS) < (0.018 + 0.025 * (buy_sell == "SELL")))).astype(int)

    # Trade-size ratio compares this trade with a synthetic counterparty normal size.
    counterparty_avg_trade_amount = np.exp(rng.normal(np.log(1_000_000), 0.75, N_ROWS)).clip(50_000, 20_000_000)
    trade_size_ratio = np.clip(trade_amount / counterparty_avg_trade_amount, 0.02, 40)

    # A note available BEFORE settlement. It can be used in optional pre-settlement NLP experiments.
    pre_notes = []
    for i in range(N_ROWS):
        notes = []
        if ssi_status[i] == "MISSING": notes.append("settlement instruction missing")
        elif ssi_status[i] == "CHANGED": notes.append("settlement instruction recently changed")
        if confirmation_delay_minutes[i] > 75: notes.append("late confirmation")
        if matching_attempts[i] >= 4: notes.append("multiple matching attempts")
        if inventory_shortfall_indicator[i]: notes.append("possible inventory constraint")
        if market_holiday_mismatch[i]: notes.append("cross-market holiday review")
        if prior_fail_rate_30d[i] > 0.12: notes.append("counterparty recent fail rate elevated")
        pre_notes.append("; ".join(notes) if notes else "normal pre-settlement processing")

    # Latent failure probability. Coefficients intentionally create non-linear business-relevant relationships.
    log_amount = np.log1p(trade_amount)
    risk = (
        -5.00
        + 2.50 * (ssi_status == "MISSING")
        + 1.15 * (ssi_status == "CHANGED")
        + 8.0 * prior_fail_rate_30d
        + 0.016 * np.maximum(confirmation_delay_minutes - 40, 0)
        + 0.35 * np.maximum(matching_attempts - 2, 0)
        + 0.45 * trade_amendments
        + 1.50 * inventory_shortfall_indicator
        + 1.10 * market_holiday_mismatch
        + 0.48 * cross_border
        + 0.40 * month_end
        + 0.32 * quarter_end
        + 0.022 * np.maximum(market_volatility_index - 25, 0)
        - 1.05 * security_liquidity_score
        + 0.30 * np.log1p(trade_size_ratio)
        + 0.12 * (log_amount - log_amount.mean())
        - 0.025 * np.minimum(hours_until_cutoff, 24)
        + rng.normal(0, 0.22, N_ROWS)
    )
    fail_probability = sigmoid(risk)
    settlement_failed = rng.binomial(1, fail_probability)

    # Secondary regression target: settlement delay. Successful trades usually have zero delay.
    base_delay = rng.gamma(1.5, 5.0, N_ROWS)
    failure_delay = (
        5
        + 15 * (ssi_status == "MISSING")
        + 8 * inventory_shortfall_indicator
        + 4 * market_holiday_mismatch
        + 0.08 * confirmation_delay_minutes
        + 12 * prior_fail_rate_30d
        + base_delay
    )
    settlement_delay_hours = np.where(settlement_failed == 1, failure_delay, np.maximum(0, rng.normal(0.3, 0.7, N_ROWS)))

    # POST-OUTCOME diagnostic fields. These must not be included as model predictors.
    failure_reason = np.full(N_ROWS, "SETTLED", dtype=object)
    failed_idx = np.where(settlement_failed == 1)[0]
    for i in failed_idx:
        candidates = []
        weights = []
        if ssi_status[i] != "VALID": candidates.append("SSI / Settlement Instruction"); weights.append(3.2)
        if inventory_shortfall_indicator[i]: candidates.append("Insufficient Securities / Inventory"); weights.append(3.0)
        if confirmation_delay_minutes[i] > 60: candidates.append("Late Confirmation / Affirmation"); weights.append(2.3)
        if matching_attempts[i] >= 3: candidates.append("Trade Matching Break"); weights.append(2.0)
        if market_holiday_mismatch[i]: candidates.append("Market Holiday / Calendar Mismatch"); weights.append(2.0)
        if prior_fail_rate_30d[i] > 0.10: candidates.append("Counterparty Operational Issue"); weights.append(1.8)
        if not candidates:
            candidates = ["Other Operational Exception", "Funding / Cash Issue"]
            weights = [1.0, 0.8]
        weights = np.array(weights) / np.sum(weights)
        failure_reason[i] = rng.choice(candidates, p=weights)

    reason_text_map = {
        "SETTLED": "Trade settled successfully with no final exception.",
        "SSI / Settlement Instruction": "Settlement failed because settlement instruction details were missing, changed, or inconsistent.",
        "Insufficient Securities / Inventory": "Settlement failed due to insufficient securities availability or inventory constraint.",
        "Late Confirmation / Affirmation": "Settlement failed after delayed confirmation or affirmation prevented timely processing.",
        "Trade Matching Break": "Settlement failed because transaction details did not match after repeated matching attempts.",
        "Market Holiday / Calendar Mismatch": "Settlement failed because market or settlement calendar differences affected the expected settlement date.",
        "Counterparty Operational Issue": "Settlement failed due to a counterparty operational processing issue.",
        "Other Operational Exception": "Settlement failed due to another operational exception requiring manual investigation.",
        "Funding / Cash Issue": "Settlement failed because required funding or cash was not available by the settlement deadline.",
    }
    failure_reason_description = np.array([reason_text_map[x] for x in failure_reason])

    # Map asset class to plausible location, but preserve some variety.
    default_location = np.select(
        [market == "US", market == "UK", market == "EU", market == "JP", market == "CA"],
        ["DTC", "CREST", "Euroclear", "JASDEC", "CDS"],
        default="Clearstream",
    )
    settlement_location = np.where(rng.random(N_ROWS) < 0.83, default_location, rng.choice(settlement_locations, N_ROWS))

    df = pd.DataFrame({
        "trade_id": [f"TRD{i:07d}" for i in range(1, N_ROWS + 1)],
        "trade_date": trade_date,
        "settlement_date": settlement_date,
        "asset_class": asset_class,
        "buy_sell": buy_sell,
        "currency": currency,
        "market": market,
        "settlement_location": settlement_location,
        "counterparty_id": counterparty_id,
        "custodian": custodian,
        "trade_amount": np.round(trade_amount, 2),
        "quantity": quantity,
        "settlement_days": settlement_days,
        "cross_border": cross_border,
        "month_end": month_end,
        "quarter_end": quarter_end,
        "market_holiday_mismatch": market_holiday_mismatch,
        "ssi_status": ssi_status,
        "confirmation_delay_minutes": np.round(confirmation_delay_minutes, 2),
        "allocation_delay_minutes": np.round(allocation_delay_minutes, 2),
        "affirmation_delay_minutes": np.round(affirmation_delay_minutes, 2),
        "matching_attempts": matching_attempts,
        "trade_amendments": trade_amendments,
        "hours_until_cutoff": np.round(hours_until_cutoff, 2),
        "prior_fail_rate_30d": np.round(prior_fail_rate_30d, 4),
        "prior_fail_rate_90d": np.round(prior_fail_rate_90d, 4),
        "prior_trade_count_30d": prior_trade_count_30d,
        "security_liquidity_score": np.round(security_liquidity_score, 4),
        "market_volatility_index": np.round(market_volatility_index, 2),
        "inventory_shortfall_indicator": inventory_shortfall_indicator,
        "counterparty_avg_trade_amount": np.round(counterparty_avg_trade_amount, 2),
        "trade_size_ratio": np.round(trade_size_ratio, 4),
        "pre_settlement_note": pre_notes,
        "settlement_failed": settlement_failed,
        "settlement_delay_hours": np.round(settlement_delay_hours, 2),
        "failure_reason": failure_reason,
        "failure_reason_description": failure_reason_description,
    }).sort_values(["trade_date", "trade_id"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Created {OUT}")
    print(f"Rows: {len(df):,}")
    print(f"Failure rate: {df['settlement_failed'].mean():.2%}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
