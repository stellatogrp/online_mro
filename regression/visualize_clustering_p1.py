"""Visualize the current clustering and the label composition within each cluster
(p = 1 hinge-loss / classification setting).

The DRO sparse-SVM clusters the joint (x, y) data of dimension m + 1, where the
last coordinate is the label y in {-1, +1}.  Because clustering is done in the
joint space, each cluster mixes the two classes and its *centroid label*
y_k^c (the weighted mean label) is fractional -- this is exactly the soft label
the mean-robust hinge surrogate uses (see WRITEUP_p1.md, sec. 1.4).

This script clusters a running sample into K kmeans centroids exactly as
``reg_orig_p1.py`` does, then draws:

  * (left)        a 2-D PCA scatter of the covariates, coloured by cluster and
                  with marker shape giving the true label (+1 vs -1); cluster
                  centroids are overlaid, ringed by their soft label y_k^c.
  * (top right)   per-cluster label composition: a stacked bar of the +1 / -1
                  counts in each cluster (clusters sorted by size).
  * (bottom right) the centroid soft label y_k^c in [-1, 1] per cluster -- how
                  far each cluster is from being label-pure (+-1 = pure).

Run:  python visualize_clustering_p1.py  ->  writes clustering_labels_p1.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from utils_p1 import generate_classification_data, label_aware_kmeans

# Match the p1 driver defaults.
M = 10            # covariate dimension
K_TRUE = 5        # true sparsity
SEED = 12345
N = 500           # running sample size to visualise
K = 10            # number of clusters (reg_orig_p1 K_arr = [10, 25])
DIM = M + 1       # joint (covariate | label) clustering dimension

data, beta_true = generate_classification_data(
    n_total=20000, m=M, k_true=K_TRUE, seed=SEED)
samples = data[:N, :DIM]
X = samples[:, :M]
y = samples[:, M]

# ---- label-pure clustering, exactly as reg_orig_p1 does (label_aware_kmeans):
# the K budget is split across the two labels and each is clustered separately,
# so every cluster is label-pure and the total cluster count is <= K. ----
centers, labels, weights = label_aware_kmeans(samples, K, DIM)
Kc = centers.shape[0]                          # actual number of clusters (<= K)
soft_label = centers[:, M]                     # y_k^c in [-1, 1] (now exactly +-1)

# per-cluster class counts and purity
pos_count = np.array([np.sum(y[labels == k] == 1) for k in range(Kc)])
neg_count = np.array([np.sum(y[labels == k] == -1) for k in range(Kc)])
size = pos_count + neg_count
frac_pos = np.divide(pos_count, size, out=np.full(Kc, 0.5), where=size > 0)
purity = np.maximum(frac_pos, 1 - frac_pos)

# ---- 2-D PCA of the covariates (numpy SVD, centered) for the scatter ----
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
PC = Xc @ Vt[:2].T                             # (N, 2)
cen_PC = (centers[:, :M] - X.mean(axis=0)) @ Vt[:2].T
ev = (S[:2] ** 2) / np.sum(S ** 2)             # explained variance ratio

# ===================== figure =====================
fig = plt.figure(figsize=(15, 7.5))
fig.suptitle(
    f"Current clustering and within-cluster labels  "
    f"(p=1 hinge / classification; n={N}, K={K} budget -> {Kc} label-pure clusters, m={M})",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                       wspace=0.22, hspace=0.32)
cmap = plt.cm.tab10 if Kc <= 10 else plt.cm.tab20
cluster_colors = cmap(np.arange(Kc) % cmap.N)

# ---- (left) PCA scatter ----
ax0 = fig.add_subplot(gs[:, 0])
for k in range(Kc):
    for lab, marker, name in ((1, "o", None), (-1, "X", None)):
        sel = (labels == k) & (y == lab)
        if np.any(sel):
            ax0.scatter(PC[sel, 0], PC[sel, 1], s=22, marker=marker,
                        color=cluster_colors[k], alpha=0.75, linewidths=0.3,
                        edgecolors="white")
# centroids: big diamond, face = cluster colour, ring = soft label (red->blue)
ring = plt.cm.coolwarm((soft_label + 1) / 2)   # -1 -> blue, +1 -> red
for k in range(Kc):
    ax0.scatter(cen_PC[k, 0], cen_PC[k, 1], s=320, marker="D",
                color=cluster_colors[k], edgecolors=ring[k], linewidths=3.0,
                zorder=5)
    ax0.annotate(str(k), (cen_PC[k, 0], cen_PC[k, 1]), ha="center", va="center",
                 fontsize=8, fontweight="bold", zorder=6)
ax0.set_title(f"Covariate PCA  (PC1 {ev[0]*100:.0f}%, PC2 {ev[1]*100:.0f}% var)\n"
              "marker = true label (o:+1,  x:-1);  diamond = centroid, ring = soft label")
ax0.set_xlabel("PC1")
ax0.set_ylabel("PC2")
ax0.grid(alpha=0.2)

# legend proxies
from matplotlib.lines import Line2D
leg = [
    Line2D([0], [0], marker="o", color="0.4", ls="", ms=7, label="label +1"),
    Line2D([0], [0], marker="X", color="0.4", ls="", ms=8, label="label -1"),
    Line2D([0], [0], marker="D", color="0.7", ls="", ms=11,
           markeredgecolor="k", label="cluster centroid"),
]
ax0.legend(handles=leg, loc="best", fontsize=9, framealpha=0.9)

# ---- (top right) per-cluster label composition (stacked bars) ----
ax1 = fig.add_subplot(gs[0, 1])
order = np.argsort(-size)                       # largest cluster first
xpos = np.arange(Kc)
ax1.bar(xpos, pos_count[order], color="#c0392b", label="+1")
ax1.bar(xpos, neg_count[order], bottom=pos_count[order], color="#2c6fbb",
        label="-1")
for i, k in enumerate(order):
    ax1.text(i, size[k] + max(size) * 0.01, f"{purity[k]*100:.0f}%",
             ha="center", va="bottom", fontsize=7.5, color="0.25")
ax1.set_xticks(xpos)
ax1.set_xticklabels([f"{k}\nw={weights[k]:.2f}" for k in order], fontsize=7.5)
ax1.set_title("Label composition within each cluster  (text = majority purity)")
ax1.set_xlabel("cluster (sorted by size)")
ax1.set_ylabel("# points")
ax1.legend(title="true label", loc="upper right", fontsize=9)
ax1.grid(alpha=0.2, axis="y")

# ---- (bottom right) centroid soft label ----
ax2 = fig.add_subplot(gs[1, 1])
bar_colors = plt.cm.coolwarm((soft_label[order] + 1) / 2)
ax2.bar(xpos, soft_label[order], color=bar_colors, edgecolor="0.3", linewidth=0.5)
ax2.axhline(0, color="0.4", lw=0.8)
ax2.set_ylim(-1.05, 1.05)
ax2.set_xticks(xpos)
ax2.set_xticklabels([str(k) for k in order], fontsize=8)
ax2.set_title(r"Centroid soft label  $y_k^c$  (weighted mean label; $\pm1$ = pure)")
ax2.set_xlabel("cluster (sorted by size)")
ax2.set_ylabel(r"$y_k^c \in [-1, 1]$")
ax2.grid(alpha=0.2, axis="y")

fig.subplots_adjust(top=0.89, left=0.05, right=0.98, bottom=0.08)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "clustering_labels_p1.png")
fig.savefig(out, dpi=130)

# ---- text summary ----
print(f"{'clu':>4}{'size':>6}{'w':>8}{'#+1':>6}{'#-1':>6}"
      f"{'frac+':>8}{'purity':>8}{'soft_y':>9}")
for k in order:
    print(f"{k:>4}{size[k]:>6}{weights[k]:>8.3f}{pos_count[k]:>6}{neg_count[k]:>6}"
          f"{frac_pos[k]:>8.2f}{purity[k]:>8.2f}{soft_label[k]:>9.3f}")
print(f"\nmean majority purity over clusters: {purity.mean():.3f}")
print(f"saved -> {out}")
