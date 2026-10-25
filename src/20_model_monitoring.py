"""Scenario 20: Simple model/data monitoring concepts.

Production ML must detect changing input distributions and outcome rates. This
example compares early vs late periods using Population Stability Index (PSI)
for selected numeric features and reports failure-rate drift.
"""
import numpy as np
import pandas as pd
from common import load_data, OUTPUT_DIR


def psi(expected, actual, bins=10):
    quantiles=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(quantiles)<3: return 0.0
    e=np.histogram(expected,bins=quantiles)[0]/len(expected)
    a=np.histogram(actual,bins=quantiles)[0]/len(actual)
    e=np.clip(e,1e-6,None); a=np.clip(a,1e-6,None)
    return float(np.sum((a-e)*np.log(a/e)))


def main():
    df=load_data(); cut=int(len(df)*.6)
    baseline,current=df.iloc[:cut],df.iloc[cut:]
    features=["trade_amount","confirmation_delay_minutes","prior_fail_rate_30d",
              "security_liquidity_score","market_volatility_index","trade_size_ratio"]
    rows=[{"feature":c,"psi":psi(baseline[c].to_numpy(),current[c].to_numpy())} for c in features]
    result=pd.DataFrame(rows).sort_values("psi",ascending=False)
    result["interpretation"]=pd.cut(result["psi"],[-1,.10,.25,999],labels=["stable","moderate shift","significant shift"])
    result.to_csv(OUTPUT_DIR/"20_data_drift_psi.csv",index=False)
    print(result.to_string(index=False))
    print("Baseline fail rate:",baseline.settlement_failed.mean(),"Current fail rate:",current.settlement_failed.mean())

if __name__ == "__main__":
    main()
