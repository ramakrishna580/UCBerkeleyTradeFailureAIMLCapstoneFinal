"""Scenario 18: GenAI-ready explanation layer WITHOUT requiring an external API.

Design principle:
- ML model owns the numeric risk prediction.
- SHAP/model logic owns the ranked risk drivers.
- A future LLM may only summarize those structured facts.
- The LLM must not invent causes, change the score, or autonomously alter trades.

This script shows a deterministic template that can later be swapped for an LLM.
"""

def build_risk_narrative(trade_id: str, probability: float, drivers: list[str], actions: list[str]) -> str:
    band = "CRITICAL" if probability >= .80 else "HIGH" if probability >= .60 else "MEDIUM" if probability >= .30 else "LOW"
    return (
        f"Trade {trade_id} has a calibrated settlement-failure probability of {probability:.1%} "
        f"and is classified as {band} risk. The strongest model-supported drivers are "
        f"{', '.join(drivers)}. Recommended human review actions are {', '.join(actions)}. "
        "This explanation summarizes model outputs and does not constitute an autonomous settlement decision."
    )


def main():
    print(build_risk_narrative(
        "TRD0098321", .87,
        ["missing/changed SSI", "elevated counterparty 30-day fail rate", "late confirmation"],
        ["validate SSI", "contact counterparty", "review settlement cutoff"]
    ))

if __name__ == "__main__":
    main()
