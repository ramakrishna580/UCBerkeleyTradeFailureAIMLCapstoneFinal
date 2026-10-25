"""Scenario 06: Decision Tree classifier.

A shallow tree is intentionally used because interpretability is the point of
this scenario. It can expose simple business-style rules such as SSI status +
confirmation delay + prior counterparty risk.
"""
from sklearn.tree import DecisionTreeClassifier
from common import load_data, temporal_split, get_X_y, make_pipeline, evaluate_classifier, save_metrics


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=42)
    model = make_pipeline(tree, X_train, scale_numeric=False)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test, "Decision Tree")
    save_metrics(metrics, "06_decision_tree_metrics.csv")

if __name__ == "__main__":
    main()
