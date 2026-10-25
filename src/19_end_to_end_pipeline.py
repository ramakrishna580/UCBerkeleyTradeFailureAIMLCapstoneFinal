"""Scenario 19: End-to-end production-style pipeline.

This script demonstrates the complete supervised workflow:
load -> temporal split -> preprocess -> train -> calibrate threshold -> score ->
rank high-risk trades -> save an operations queue.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from common import load_data, temporal_split, get_X_y, make_pipeline, OUTPUT_DIR


def main():
    train, valid, test=temporal_split(load_data())
    X_train,y_train=get_X_y(train); X_valid,y_valid=get_X_y(valid); X_test,y_test=get_X_y(test)
    pos=y_train.sum(); neg=len(y_train)-pos
    est=XGBClassifier(n_estimators=400,max_depth=5,learning_rate=.04,subsample=.85,
                      colsample_bytree=.85,scale_pos_weight=neg/max(pos,1),
                      eval_metric="logloss",random_state=42,n_jobs=-1)
    model=make_pipeline(est,X_train,scale_numeric=False)
    model.fit(X_train,y_train)

    # Threshold chosen on VALIDATION data. Here we maximize F1 as a neutral demo;
    # use the cost-optimization script when business costs are available.
    vp=model.predict_proba(X_valid)[:,1]
    precision,recall,thresholds=precision_recall_curve(y_valid,vp)
    f1=2*precision*recall/np.maximum(precision+recall,1e-9)
    best=int(np.nanargmax(f1[:-1])); threshold=float(thresholds[best])
    print(f"Selected validation threshold: {threshold:.3f}")

    tp=model.predict_proba(X_test)[:,1]
    queue=test[["trade_id","trade_date","settlement_date","counterparty_id","asset_class","trade_amount","ssi_status","pre_settlement_note","settlement_failed"]].copy()
    queue["predicted_failure_probability"]=tp
    queue["predicted_high_risk"]=(tp>=threshold).astype(int)
    queue["risk_band"]=pd.cut(tp,[-.001,.30,.60,.80,1.0],labels=["LOW","MEDIUM","HIGH","CRITICAL"])
    queue.sort_values("predicted_failure_probability",ascending=False).to_csv(OUTPUT_DIR/"19_operations_risk_queue.csv",index=False)
    print(queue.sort_values("predicted_failure_probability",ascending=False).head(15).to_string(index=False))

if __name__ == "__main__":
    main()
