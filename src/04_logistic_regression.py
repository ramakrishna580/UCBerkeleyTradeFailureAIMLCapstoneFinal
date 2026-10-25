"""Scenario 04: Logistic Regression baseline.

Purpose:
- Establish an interpretable baseline.
- Demonstrate class weighting for rare failures.
- Provide a benchmark that advanced models must beat.
"""
from sklearn.linear_model import LogisticRegression
from common import load_data, temporal_split, get_X_y, make_pipeline, evaluate_classifier, save_metrics


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    model = make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42), X_train)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test, "Logistic Regression")
    save_metrics(metrics, "04_logistic_metrics.csv")

if __name__ == "__main__":
    main()
