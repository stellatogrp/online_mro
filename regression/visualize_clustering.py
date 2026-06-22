"""Visualize the clustering value vs. number of clusters K.

For the synthetic regression data, clusters the running sample into K kmeans
centroids and reports the same "clustering value" the experiments use
(``utils.calc_cluster_val_reg``):

  * Wasserstein-1 distance  W_1(P_n, P_K)  between the empirical measure and the
    K weighted centroids -- this is what enters the MRO approximation / regret
    bound, so it is *the* clustering value of interest;
  * within-cluster RMSE  sqrt(mean squared distance to centroid)  -- the usual
    k-means distortion / elbow curve.

Both are swept over K for a few sample sizes n, since the clustering value
depends on how many points are being summarised.  The experiment's K choices
(K_arr = [15, 25, 30] in reg_orig.py) are marked for reference.

Run:  python visualize_clustering.py  ->  writes clustering_value.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from utils import generate_regression_data, calc_cluster_val_reg

# Match the driver defaults.
M = 50
K_TRUE = 10
SEED = 12345
DIM = M + 1  # joint (covariate | response) clustering dimension

data, beta_true = generate_regression_data(n_total=20000, m=M, k_true=K_TRUE, seed=SEED)

N_VALUES = [200, 500, 1000]
K_RANGE = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
K_EXPERIMENT = [15, 25, 30]


def cluster_value(samples, K, beta):
    """Build the kmeans k_dict (as reg_orig does) and call calc_cluster_val_reg."""
    num_dat = samples.shape[0]
    km = KMeans(n_clusters=K, n_init=3).fit(samples)
    labels = km.labels_
    k_dict = {
        'K': K,
        'd': km.cluster_centers_,
        'w': np.bincount(labels, minlength=K) / num_dat,
        'data': {k: samples[labels == k] for k in range(K)},
    }
    w_distance, square_val, sig_val = calc_cluster_val_reg(
        K, k_dict, num_dat, beta, samples, M)
    return w_distance, np.sqrt(square_val), sig_val


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Clustering value vs. number of clusters K  "
    "(synthetic regression data, joint (x, y) clustering)",
    fontsize=13, fontweight="bold",
)
cmap = plt.cm.viridis(np.linspace(0.15, 0.8, len(N_VALUES)))

for ci, n in enumerate(N_VALUES):
    samples = data[:n, :DIM]
    wdist, rmse = [], []
    for K in K_RANGE:
        wd, rm, _ = cluster_value(samples, K, beta_true)
        wdist.append(wd)
        rmse.append(rm)
    axes[0].plot(K_RANGE, wdist, "o-", color=cmap[ci], label=f"n = {n}")
    axes[1].plot(K_RANGE, rmse, "o-", color=cmap[ci], label=f"n = {n}")

for ax, title, ylab in (
    (axes[0], r"Wasserstein-1 clustering value  $W_1(\hat P_n,\,\hat P_K)$", r"$W_1$ distance"),
    (axes[1], "Within-cluster RMSE (k-means distortion)", "RMSE to centroid"),
):
    for Kx in K_EXPERIMENT:
        ax.axvline(Kx, color="0.7", ls=":", lw=1, zorder=0)
    ax.set_title(title)
    ax.set_xlabel("number of clusters K")
    ax.set_ylabel(ylab)
    ax.legend(title=f"K_arr={K_EXPERIMENT}", loc="upper right")
    ax.grid(alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering_value.png")
fig.savefig(out, dpi=130)

print("K       " + "".join(f"{K:>8}" for K in K_RANGE))
for n in N_VALUES:
    samples = data[:n, :DIM]
    wd = [cluster_value(samples, K, beta_true)[0] for K in K_RANGE]
    print(f"W1 n={n:<4} " + "".join(f"{v:8.3f}" for v in wd))
print(f"saved -> {out}")
