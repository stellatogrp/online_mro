"""Radius selection for the p=1 hinge-loss DRO sparse-SVM (classification).

Hinge-loss sibling of ``visualize_radius.py``, tuned to the setting the driver
runs (``reg_orig_p1.py --m 20 --k 5 --k_true 5 --noise 3``).  It solves the
full-data DRO sparse-SVM MILP (``createproblem_hingeMIO``, MOSEK) and plots the
quantities that determine an appropriate radius.

Two things differ from the p=2 regression sweep:

  * the radius enters *linearly* and is the order-1 radius itself,
    ``radius = delta = init_eps / sqrt(n)`` (no square root), so the natural
    decision variable is the n-independent prefactor ``init_eps`` -- we sweep
    that on the x-axis and solve each n at ``delta = init_eps / sqrt(n)``, so the
    curves for different n are directly comparable in the quantity you choose;
  * the penalty is the sparsity-promoting ell_1 norm, so the shrinkage panel
    shows ||beta||_1.

Panels (vs init_eps):
  * out-of-sample hinge loss          (statistical optimum -> a min)
  * out-of-sample misclassification   (the classification metric; Bayes ref)
  * validity gap  DRO obj - test hinge  (>= 0 = valid upper bound / certificate)
  * ||beta||_1                        (coefficient shrinkage / selection)

Run:  python visualize_radius_p1.py  ->  writes radius_selection_p1.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from utils_p1 import (generate_classification_data, createproblem_hingeMIO,
                      evaluate_expected_cost_hinge, evaluate_misclassification_hinge,
                      MOSEK_PARAMS)

# Match the driver: reg_orig_p1.py --m 20 --k 5 --k_true 5 --noise 3
M, K_CARD = 20, 5
SEED = 12345
data, beta_true = generate_classification_data(
    n_total=20000, m=M, k_true=5, noise_std=3.0, seed=SEED)
test = data[15000:20000]

# Bayes references (the true sparse classifier on the test split).
bayes_hinge = evaluate_expected_cost_hinge(test, M, beta_true)
bayes_err = evaluate_misclassification_hinge(test, M, beta_true)
beta_true_l1 = np.linalg.norm(beta_true, 1)

# Sample sizes spanning the stream (N_init=5 .. T=2001), and the init_eps grid
# (radius = init_eps / sqrt(n)); 0 = non-robust SAA.
N_VALUES = [100, 500, 2000]
INIT_EPS = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
REC_BAND = (0.3, 0.7)   # recommended init_eps band for this (m, k) regime


def solve(n, delta):
    samp = data[:n]
    prob, beta, z, dp, eps, w = createproblem_hingeMIO(n, M, K_CARD)
    dp.value = samp
    eps.value = float(delta)
    w.value = np.ones(n) / n
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False, mosek_params=MOSEK_PARAMS)
    b = beta.value
    return (prob.objective.value,
            evaluate_expected_cost_hinge(test, M, b),
            evaluate_misclassification_hinge(test, M, b),
            np.linalg.norm(b, 1))


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    r"Radius selection (p=1 hinge / sparse-SVM): metrics vs. prefactor $\varepsilon_0$  "
    f"(m={M}, k={K_CARD}; radius $\\delta=\\varepsilon_0/\\sqrt{{n}}$)",
    fontsize=13, fontweight="bold",
)
colors = {100: "#d6604d", 500: "#9970ab", 2000: "#2166ac"}

summary = {}
for n in N_VALUES:
    res = np.array([solve(n, ie / np.sqrt(n)) for ie in INIT_EPS])
    objs, hinge, misclass, l1 = res.T
    c = colors[n]

    # panel (0,0): out-of-sample hinge
    axes[0, 0].plot(INIT_EPS, hinge, "o-", color=c, label=f"n={n}")
    ie_h = INIT_EPS[np.argmin(hinge)]
    axes[0, 0].scatter([ie_h], [hinge.min()], s=150, facecolors="none",
                       edgecolors=c, linewidths=2, zorder=5)
    # panel (0,1): misclassification
    axes[0, 1].plot(INIT_EPS, misclass, "o-", color=c, label=f"n={n}")
    ie_m = INIT_EPS[np.argmin(misclass)]
    axes[0, 1].scatter([ie_m], [misclass.min()], s=150, facecolors="none",
                       edgecolors=c, linewidths=2, zorder=5)
    # panel (1,0): validity gap (DRO obj - test hinge); >=0 -> valid upper bound
    gap = objs - hinge
    axes[1, 0].plot(INIT_EPS, gap, "o-", color=c, label=f"n={n}")
    valid = INIT_EPS[gap >= 0]
    ie_v = valid.min() if valid.size else np.nan
    if valid.size:
        axes[1, 0].scatter([ie_v], [0], s=150, facecolors="none",
                           edgecolors=c, linewidths=2, zorder=5)
    # panel (1,1): coefficient L1 norm
    axes[1, 1].plot(INIT_EPS, l1, "o-", color=c, label=f"n={n}")

    summary[n] = (ie_h, ie_m, ie_v)

axes[0, 0].axhline(bayes_hinge, color="0.5", ls="--", lw=1,
                   label=r"hinge($\beta^\star$)")
axes[0, 0].set_title("Out-of-sample hinge loss\n(circle = optimum)")
axes[0, 0].set_ylabel("test hinge loss")

axes[0, 1].axhline(bayes_err, color="0.5", ls="--", lw=1, label="Bayes error")
axes[0, 1].set_title("Out-of-sample misclassification\n(circle = optimum)")
axes[0, 1].set_ylabel("test 0-1 error")

axes[1, 0].axhline(0, color="k", lw=1)
axes[1, 0].set_title("Bound validity:  DRO obj - test hinge\n(>= 0 = valid upper bound; circle = threshold)")
axes[1, 0].set_ylabel("DRO objective - test hinge")

axes[1, 1].axhline(beta_true_l1, color="0.5", ls="--", lw=1,
                   label=r"$\|\beta^\star\|_1$")
axes[1, 1].set_title("Coefficient norm (shrinkage / selection)")
axes[1, 1].set_ylabel(r"$\|\hat\beta\|_1$")

for ax in axes.ravel():
    ax.axvspan(*REC_BAND, color="0.85", alpha=0.5, zorder=0,
               label=f"recommended $\\varepsilon_0\\in[{REC_BAND[0]},{REC_BAND[1]}]$")
    ax.set_xlabel(r"prefactor  $\varepsilon_0$  (init_eps)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radius_selection_p1.png")
fig.savefig(out, dpi=130)

print(f"Bayes (beta*) : hinge={bayes_hinge:.3f}  misclass={bayes_err:.3f}  "
      f"||beta*||_1={beta_true_l1:.2f}")
print("n      OOS-hinge-opt   misclass-opt   min-valid   (all as init_eps; "
      "radius = init_eps/sqrt(n))")
for n in N_VALUES:
    ie_h, ie_m, ie_v = summary[n]
    print(f"{n:<6} eps0={ie_h:<11.2f} eps0={ie_m:<10.2f} eps0={ie_v:<7.2f}")
print(f"saved -> {out}")
