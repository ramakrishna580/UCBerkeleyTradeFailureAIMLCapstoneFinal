"""Scenario 10: K-Means trade-risk segmentation.

Unsupervised learning question:
'Are there naturally occurring operational-risk profiles even without using the
settlement_failed target?'

The target is used only AFTER clustering to profile clusters; it is not passed
to K-Means.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from common import load_data, OUTPUT_DIR

FEATURES = ["trade_amount", "confirmation_delay_minutes", "matching_attempts",
            "trade_amendments", "hours_until_cutoff", "prior_fail_rate_30d",
            "security_liquidity_score", "market_volatility_index", "trade_size_ratio"]


def main():
    df = load_data()
    X = StandardScaler().fit_transform(df[FEATURES])
    inertias, silhouettes = [], []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_, sample_size=5000, random_state=42))
    pd.DataFrame({"k": range(2, 8), "inertia": inertias, "silhouette": silhouettes}).to_csv(OUTPUT_DIR / "10_kmeans_selection.csv", index=False)

    final = KMeans(n_clusters=4, random_state=42, n_init=20).fit(X)
    prof = df[FEATURES + ["settlement_failed"]].copy()
    prof["cluster"] = final.labels_
    summary = prof.groupby("cluster").agg({**{c:"mean" for c in FEATURES}, "settlement_failed":"mean"})
    summary.to_csv(OUTPUT_DIR / "10_cluster_profiles.csv")
    print(summary.round(3))

    plt.figure(figsize=(8,5)); plt.plot(range(2,8), inertias, marker="o")
    plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("K-Means Elbow Method")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "10_kmeans_elbow.png", dpi=160); plt.close()

if __name__ == "__main__":
    main()
