"""Scenario 15: Probability-threshold and business-cost optimization.

Why this matters:
0.50 is a mathematical default, not necessarily the correct business threshold.
A missed failure (false negative) may be much more expensive than investigating
a false alarm (false positive).

Illustrative costs below are assumptions. Replace them with defensible business
estimates in the final report.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from common import load_data, temporal_split, get_X_y, make_pipeline, OUTPUT_DIR

FALSE_NEGATIVE_COST = 5000.0
FALSE_POSITIVE_COST = 50.0


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_valid, y_valid = get_X_y(valid)
    model = make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42), X_train)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_valid)[:,1]

    rows=[]
    for threshold in np.arange(0.05, 0.96, 0.025):
        pred=(prob>=threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_valid, pred).ravel()
        cost = fn*FALSE_NEGATIVE_COST + fp*FALSE_POSITIVE_COST
        rows.append({"threshold":threshold,"TP":tp,"FP":fp,"FN":fn,"TN":tn,"estimated_cost":cost})
    result=pd.DataFrame(rows).sort_values("estimated_cost")
    print("Best thresholds by estimated cost:\n", result.head(10).to_string(index=False))
    result.to_csv(OUTPUT_DIR / "15_threshold_cost_curve.csv", index=False)

if __name__ == "__main__":
    main()
