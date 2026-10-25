"""Scenario 08: Support Vector Machine.

SVM demonstrates a nonlinear classical classifier. It can be computationally
expensive on large trade datasets, so a chronological training subset is used.
"""
from sklearn.svm import SVC
from common import load_data, temporal_split, get_X_y, make_pipeline, evaluate_classifier, save_metrics


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    X_train, y_train = X_train.tail(10000), y_train.tail(10000)
    svm = SVC(C=2.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    model = make_pipeline(svm, X_train)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test, "SVM (RBF)")
    save_metrics(metrics, "08_svm_metrics.csv")

if __name__ == "__main__":
    main()
