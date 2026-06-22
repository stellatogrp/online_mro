"""Visualize the synthetic regression dataset used by the DRO-BSS experiments.

Generates the same data the drivers consume (``generate_regression_data`` with
the default config: m=10 covariates, k_true=5 true-nonzero coefficients,
Toeplitz-correlated X, Gaussian noise) and draws a multi-panel summary:

  1. true sparse coefficient vector beta*       (which features matter)
  2. covariate correlation heatmap              (Toeplitz rho^|i-j| structure)
  3. response histogram                          (marginal of y)
  4. y vs. signal X beta*                         (linear fit + noise band)
  5. y vs. an ACTIVE covariate                    (clear slope)
  6. y vs. an INACTIVE covariate                  (no slope)

Run:  python visualize_data.py  ->  writes regression_data.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import generate_regression_data

# Match the driver defaults (reg_orig.py / reg_DRO_orig.py __main__).
M = 10
K_TRUE = 5
SEED = 12345
N_TOTAL = 20000
N_SHOW = 2000  # subsample for scatter readability

data, beta_true = generate_regression_data(n_total=N_TOTAL, m=M, k_true=K_TRUE, seed=SEED)
X = data[:, :M]
y = data[:, M]

active = np.flatnonzero(beta_true)
inactive = np.flatnonzero(beta_true == 0)
signal = X @ beta_true
resid = y - signal
noise_std = resid.std()

rng = np.random.default_rng(0)
idx = rng.choice(N_TOTAL, size=min(N_SHOW, N_TOTAL), replace=False)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    f"Synthetic regression data  (n={N_TOTAL}, d={M}, true support size={K_TRUE}, "
    f"noise std={noise_std:.2f})",
    fontsize=14, fontweight="bold",
)

# 1) true coefficients ------------------------------------------------------
ax = axes[0, 0]
colors = ["#d6604d" if b != 0 else "#bbbbbb" for b in beta_true]
ax.bar(np.arange(M), beta_true, color=colors)
ax.axhline(0, color="k", lw=0.6)
ax.set_title(r"True coefficients $\beta^\star$ (red = active)")
ax.set_xlabel("feature index"); ax.set_ylabel(r"$\beta^\star_j$")
ax.set_xticks(np.arange(M))

# 2) covariate correlation --------------------------------------------------
ax = axes[0, 1]
corr = np.corrcoef(X, rowvar=False)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_title(r"Covariate correlation (Toeplitz $\rho^{|i-j|}$)")
ax.set_xlabel("feature"); ax.set_ylabel("feature")
ax.set_xticks(np.arange(M)); ax.set_yticks(np.arange(M))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 3) response marginal ------------------------------------------------------
ax = axes[0, 2]
ax.hist(y, bins=60, color="#4393c3", edgecolor="white")
ax.set_title("Response $y$ distribution")
ax.set_xlabel("$y$"); ax.set_ylabel("count")

# 4) y vs signal ------------------------------------------------------------
ax = axes[1, 0]
ax.scatter(signal[idx], y[idx], s=6, alpha=0.3, color="#4393c3")
lims = [min(signal[idx].min(), y[idx].min()), max(signal[idx].max(), y[idx].max())]
ax.plot(lims, lims, "k--", lw=1, label="$y=\\beta^{\\star\\top}x$")
ax.set_title(r"$y$ vs. true signal $\beta^{\star\top}x$")
ax.set_xlabel(r"$\beta^{\star\top}x$"); ax.set_ylabel("$y$")
ax.legend(loc="upper left")

# 5) y vs an active feature -------------------------------------------------
ax = axes[1, 1]
ja = int(active[0])
ax.scatter(X[idx, ja], y[idx], s=6, alpha=0.3, color="#d6604d")
ax.set_title(fr"$y$ vs. ACTIVE feature $x_{{{ja}}}$  ($\beta^\star={beta_true[ja]:.1f}$)")
ax.set_xlabel(f"$x_{{{ja}}}$"); ax.set_ylabel("$y$")

# 6) y vs an inactive feature ----------------------------------------------
ax = axes[1, 2]
ji = int(inactive[0])
ax.scatter(X[idx, ji], y[idx], s=6, alpha=0.3, color="#999999")
ax.set_title(fr"$y$ vs. INACTIVE feature $x_{{{ji}}}$  ($\beta^\star=0$)")
ax.set_xlabel(f"$x_{{{ji}}}$"); ax.set_ylabel("$y$")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_data.png")
fig.savefig(out, dpi=130)
print(f"true support (active features): {active.tolist()}")
print(f"beta* on support: {np.round(beta_true[active], 3).tolist()}")
print(f"empirical noise std: {noise_std:.3f}  |  var(signal)={signal.var():.3f}  var(y)={y.var():.3f}")
print(f"signal-to-noise ratio (var ratio): {signal.var() / resid.var():.2f}")
print(f"saved -> {out}")
