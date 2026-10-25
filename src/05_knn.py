"""Scenario 05: K-Nearest Neighbors.

KNN is included as a curriculum benchmark. It is sensitive to feature scale and
may struggle with high-dimensional sparse one-hot encoded trade data. That is a
useful finding rather than a project failure.
"""
from sklearn.neighbors import KNeighborsClassifier
from common import load_data, temporal_split, get_X_y, make_pipeline, evaluate_classifier, save_metrics


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    # Restrict training size for KNN runtime while preserving chronology.
    X_train, y_train = X_train.tail(12000), y_train.tail(12000)
    model = make_pipeline(KNeighborsClassifier(n_neighbors=11, weights="distance"), X_train, scale_numeric=True)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test, "KNN")
    save_metrics(metrics, "05_knn_metrics.csv")

if __name__ == "__main__":
    main()
