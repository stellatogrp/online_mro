"""Radius (sqrt(delta)) selection for the DRO best-subset problem.

For the synthetic regression data, solves the full-data DRO-BSS MI-SOCP
(``createproblem_regMIO``, MOSEK) across a grid of radii sqrt(delta) and two
sample sizes, and plots the quantities that determine an appropriate radius:

  * out-of-sample MSE on a held-out test set      (statistical optimum -> a min)
  * worst-case DRO objective vs. test MSE          (the bound is VALID where
                                                     DRO obj >= test MSE)
  * ||beta||_2 (coefficient shrinkage)

The radius enters the drivers as  radius = init_eps / sqrt(n) = sqrt(delta),
so the top axis also marks the corresponding init_eps prefactor.

Run:  python visualize_radius.py  ->  writes radius_selection.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from utils import (generate_regression_data, createproblem_regMIO,
                   evaluate_expected_cost_reg, MOSEK_PARAMS)

M, K_CARD = 10, 5
SEED = 12345
data, beta_true = generate_regression_data(n_total=20000, m=M, k_true=5, seed=SEED)
test = data[15000:20000]
noise_mse = float(((data[:, M] - data[:, :M] @ beta_true) ** 2).mean())

N_VALUES = [100, 1000]
RADII = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])


def solve(n, sd):
    samp = data[:n]
    prob, beta, z, t, r, dp, eps, sqw = createproblem_regMIO(n, M, K_CARD)
    dp.value = samp
    eps.value = float(sd)
    sqw.value = np.sqrt(np.ones(n) / n)
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False, mosek_params=MOSEK_PARAMS)
    b = beta.value
    return (prob.objective.value, evaluate_expected_cost_reg(test, M, b), np.linalg.norm(b))


fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle(
    r"Radius selection: DRO best-subset metrics vs. $\sqrt{\delta}$  "
    f"(m={M}, k={K_CARD}, noise MSE={noise_mse:.1f})",
    fontsize=13, fontweight="bold",
)
colors = {100: "#d6604d", 1000: "#2166ac"}

for n in N_VALUES:
    objs, tmse, bn = np.array([solve(n, sd) for sd in RADII]).T
    c = colors[n]
    # panel 1: out-of-sample MSE
    axes[0].plot(RADII, tmse, "o-", color=c, label=f"n={n}")
    kbest = RADII[np.argmin(tmse)]
    axes[0].scatter([kbest], [tmse.min()], s=140, facecolors="none",
                    edgecolors=c, linewidths=2, zorder=5)
    # panel 2: validity gap (DRO obj - test MSE); >=0 means valid upper bound
    gap = objs - tmse
    axes[1].plot(RADII, gap, "o-", color=c, label=f"n={n}")
    # smallest radius with a valid bound
    valid = RADII[gap >= 0]
    if valid.size:
        sd_v = valid.min()
        axes[1].scatter([sd_v], [0], s=140, facecolors="none",
                        edgecolors=c, linewidths=2, zorder=5)
    # panel 3: coefficient norm
    axes[2].plot(RADII, bn, "o-", color=c, label=f"n={n}")

axes[0].axhline(noise_mse, color="0.5", ls="--", lw=1, label="irreducible MSE")
axes[0].set_title("Out-of-sample MSE\n(circle = optimum)")
axes[0].set_ylabel("test MSE")

axes[1].axhline(0, color="k", lw=1)
axes[1].set_title("Bound validity:  DRO obj - test MSE\n(>= 0 = valid upper bound; circle = threshold)")
axes[1].set_ylabel("DRO objective - test MSE")

axes[2].axhline(np.linalg.norm(beta_true), color="0.5", ls="--", lw=1,
                label=r"$\|\beta^\star\|_2$")
axes[2].set_title("Coefficient norm (shrinkage)")
axes[2].set_ylabel(r"$\|\hat\beta\|_2$")

for ax in axes:
    # shade the recommended init_eps band [0.5,3] mapped to sqrt(delta) at the
    # SMALLER n (widest band); just annotate verbally to avoid n-ambiguity.
    ax.set_xlabel(r"radius  $\sqrt{\delta}$")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radius_selection.png")
fig.savefig(out, dpi=130)

print(f"irreducible (noise) MSE = {noise_mse:.3f}")
for n in N_VALUES:
    objs, tmse, bn = np.array([solve(n, sd) for sd in RADII]).T
    gap = objs - tmse
    sd_opt = RADII[np.argmin(tmse)]
    sd_valid = RADII[gap >= 0].min() if (gap >= 0).any() else np.nan
    print(f"n={n:<5} OOS-optimal sqrt(delta)={sd_opt:.3f} (init_eps={sd_opt*np.sqrt(n):.2f}) | "
          f"min-valid sqrt(delta)={sd_valid:.3f} (init_eps={sd_valid*np.sqrt(n):.2f})")
print(f"saved -> {out}")
