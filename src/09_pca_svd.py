"""Scenario 09: PCA and SVD dimensionality reduction.

Purpose:
- Demonstrate the relationship between standardized numeric data, SVD, and PCA.
- Determine how many components explain 80%, 90%, and 95% of numeric variance.
- This scenario intentionally uses numeric features only for clear interpretation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import load_data, PRE_PREDICTION_FEATURES, OUTPUT_DIR


def main():
    df = load_data()
    X = df[PRE_PREDICTION_FEATURES].select_dtypes(include=np.number)
    Xs = StandardScaler().fit_transform(X)

    # SVD: X = U Sigma V^T. PCA directions are related to the right singular vectors.
    U, sigma, VT = np.linalg.svd(Xs, full_matrices=False)
    svd_variance_proxy = sigma**2 / np.sum(sigma**2)

    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    thresholds = {f"components_{int(t*100)}pct": int(np.argmax(cum >= t) + 1) for t in [0.80, 0.90, 0.95]}
    print("SVD singular values (first 10):", np.round(sigma[:10], 3))
    print("PCA thresholds:", thresholds)

    pd.DataFrame({"component": np.arange(1, len(cum)+1),
                  "explained_variance_ratio": pca.explained_variance_ratio_,
                  "cumulative_variance": cum}).to_csv(OUTPUT_DIR / "09_pca_variance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cum)+1), cum, marker="o")
    plt.axhline(0.80, linestyle="--"); plt.axhline(0.90, linestyle="--"); plt.axhline(0.95, linestyle="--")
    plt.xlabel("Number of Principal Components"); plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree / Cumulative Variance")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "09_pca_scree.png", dpi=160); plt.close()

if __name__ == "__main__":
    main()
