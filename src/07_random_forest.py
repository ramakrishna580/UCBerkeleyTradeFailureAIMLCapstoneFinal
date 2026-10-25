"""Scenario 07: Random Forest ensemble.

Demonstrates bagging, feature subsampling, variance reduction, and a stronger
nonlinear tabular baseline than a single decision tree.
"""
from sklearn.ensemble import RandomForestClassifier
from common import load_data, temporal_split, get_X_y, make_pipeline, evaluate_classifier, save_metrics


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    rf = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=5,
                                class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model = make_pipeline(rf, X_train, scale_numeric=False)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test, "Random Forest")
    save_metrics(metrics, "07_random_forest_metrics.csv")

if __name__ == "__main__":
    main()
