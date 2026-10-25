"""Scenario 12: XGBoost gradient boosting.

This is a strong candidate for the primary final model because settlement data
is structured/tabular and risk relationships are nonlinear.
"""
from xgboost import XGBClassifier
from common import load_data, temporal_split, get_X_y, make_pipeline, evaluate_classifier, save_metrics


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    pos = y_train.sum(); neg = len(y_train) - pos
    xgb = XGBClassifier(
        n_estimators=450, max_depth=5, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=3, reg_lambda=1.5,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="logloss", random_state=42, n_jobs=-1)
    model = make_pipeline(xgb, X_train, scale_numeric=False)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test, "XGBoost")
    save_metrics(metrics, "12_xgboost_metrics.csv")

if __name__ == "__main__":
    main()
