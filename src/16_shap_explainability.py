"""Scenario 16: Explainable AI using SHAP.

This example fits an XGBoost model on PRE-PREDICTION fields and creates:
- global feature importance (mean absolute SHAP values)
- local explanations for a few highest-risk test trades

SHAP explanations support human-in-the-loop operations; they do not replace
business judgment.
"""
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from common import load_data, temporal_split, get_X_y, build_preprocessor, OUTPUT_DIR


def main():
    train, valid, test = temporal_split(load_data())
    X_train, y_train = get_X_y(train); X_test, y_test = get_X_y(test)
    pre = build_preprocessor(X_train, scale_numeric=False)
    Xt = pre.fit_transform(X_train); Xq = pre.transform(X_test)
    names = pre.get_feature_names_out()
    pos=y_train.sum(); neg=len(y_train)-pos
    model=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.85,
                        colsample_bytree=.85,scale_pos_weight=neg/max(pos,1),
                        eval_metric="logloss",random_state=42,n_jobs=-1)
    model.fit(Xt,y_train)
    explainer=shap.TreeExplainer(model)
    sample_n=min(1200, Xq.shape[0])
    Xsample=Xq[:sample_n]
    sv=explainer.shap_values(Xsample)
    mean_abs=np.asarray(np.abs(sv).mean(axis=0)).ravel()
    pd.DataFrame({"feature":names,"mean_abs_shap":mean_abs}).sort_values("mean_abs_shap",ascending=False).head(40).to_csv(OUTPUT_DIR/"16_global_shap_importance.csv",index=False)

    risk=model.predict_proba(Xq)[:,1]
    top_idx=np.argsort(risk)[-5:][::-1]
    local=[]
    for idx in top_idx:
        vals=np.asarray(explainer.shap_values(Xq[idx])).ravel()
        top=np.argsort(np.abs(vals))[-8:][::-1]
        for rank,j in enumerate(top,1):
            local.append({"test_row":int(idx),"predicted_risk":float(risk[idx]),"rank":rank,
                          "feature":names[j],"shap_value":float(vals[j])})
    pd.DataFrame(local).to_csv(OUTPUT_DIR/"16_local_shap_examples.csv",index=False)
    print("Saved global and local SHAP tables.")

if __name__ == "__main__":
    main()
