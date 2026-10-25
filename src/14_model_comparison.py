"""Scenario 14: Aggregate model metrics into one scorecard.

Run the individual model scripts first. The output helps the report compare
precision/recall/F1/ROC-AUC/PR-AUC rather than cherry-picking one metric.
"""
from pathlib import Path
import pandas as pd
from common import OUTPUT_DIR


def main():
    files = sorted(OUTPUT_DIR.glob("*_metrics.csv"))
    if not files:
        raise SystemExit("Run classification model scripts first; no *_metrics.csv files found.")
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    table = table.sort_values(["pr_auc", "recall"], ascending=False)
    table.to_csv(OUTPUT_DIR / "14_model_scorecard.csv", index=False)
    print(table.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
