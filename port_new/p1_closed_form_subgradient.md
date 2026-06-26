# Reducing the p1 Wasserstein-DRO solve to a closed-form subgradient step

This note documents how the type-1 (Wasserstein-1) portfolio DRO in `port_DRO.py`
replaces a per-timestep convex solve with a single closed-form projected
subgradient step. All formulas below are taken directly from `utils.py`
(`createproblem_portLP`, `createproblem_worstcase_p1_dro`, `worst_case_value`,
`dro_subgrad_step`, `project_simplex`).

## 1. What the original p1 did

`port_DRO_orig.py` rebuilds and solves the full conic LP `createproblem_portLP`
at every active interval:

```python
DRO_problem, DRO_x, DRO_s, DRO_tau, DRO_lmbda, DRO_data, DRO_eps, DRO_w = createproblem_portLP(num_dat, m)
...
safe_solve(DRO_problem, solver=cp.CLARABEL, ...)
```

That LP is the joint outer-minimization / inner-dual program

```
minimize    tau + eps*lam + w @ s
subject to  a*tau + a*(dat @ x) <= s,   s >= 0
            ||a*x||_2 <= lam
            sum(x) == 1,  0 <= x <= 1,  lam >= 0
```

with `a = -5`. Its variables `s` (per-sample epigraph), `lam` (the Wasserstein
dual multiplier), and the parameter `w` all scale with `num_dat`, which grows by
one every timestep — so the online cost is "factorize an LP whose size keeps
increasing," repeated thousands of times.

## 2. The DRO problem

The portfolio problem is the standard Esfahani–Kuhn type-1 data-driven DRO

$$\min_{x\in\Delta,\;\tau}\;\; \sup_{Q\,:\,W_1(Q,\hat P_N)\le\varepsilon}\;
\mathbb E_{Q}\big[\ell(x,\tau;\xi)\big],$$

with the simplex $\Delta=\{x\ge 0,\ \mathbf 1^\top x=1\}$, the empirical measure
$\hat P_N=\sum_i w_i\,\delta_{\xi_i}$ over `running_samples` (uniform
$w_i=1/N$ in the runs), Euclidean transport norm, and the **two-piece,
CVaR-style mean-risk loss**

$$\ell(x,\tau;\xi)\;=\;\tau \;+\; \max\!\big(0,\; a\tau + a\,\langle\xi,x\rangle\big),
\qquad a=-5.$$

The leading bare $\tau$ is the CVaR offset; the $\max(0,\cdot)$ piece is what gets
robustified. The only piece with nonzero $\xi$-gradient is $a\langle\xi,x\rangle$,
whose gradient is $a x$, so the loss is Lipschitz in $\xi$ with constant
$|a|\,\lVert x\rVert_2$.

## 3. The inner worst-case program and its closed form

For frozen $(x,\tau)$, the inner supremum is the worst-case-distribution program
`createproblem_worstcase_p1_dro`, written over transport variables — the
"$(p,z)$ dual" — as

$$\max_{p,\,z}\;\; \tau + a\tau\textstyle\sum_i p_i + a\sum_i \langle z_i,x\rangle
\qquad\text{s.t.}\qquad \sum_i\lVert z_i - p_i\,\xi_i\rVert_2 \le \varepsilon,\quad 0\le p\le w,$$

where $p_i$ is the mass placed on sample $i$ and $z_i$ its transported location.
This program has a **trivially tight optimum** (documented in
`worst_case_value`): the mass loads onto the active samples,

$$p_i^* = w_i \ \text{if}\ a\tau + a\langle\xi_i,x\rangle > 0,\quad \text{else } 0,$$

and the entire transport budget $\varepsilon$ is spent moving in the $x$
direction, which pins the multiplier to $\lambda^* = |a|\,\lVert x\rVert_2$.
The supremum therefore collapses to the **closed-form outer objective**

$$\boxed{\,F(x,\tau) \;=\; \tau \;+\; \sum_i w_i\,\max\!\big(0,\; a\tau + a\,\xi_i^\top x\big)
\;+\; \varepsilon\,|a|\,\lVert x\rVert_2\,}$$

This is the simplification: the inner sup over distributions (and its finite
$(p,z)$ dual) becomes the empirical objective **plus one regularizer term**
$\varepsilon|a|\lVert x\rVert_2$, with $\lambda$ no longer a decision variable.
There is nothing left to hand to a solver — `worst_case_value` evaluates this in
$O(Nm)$.

```python
scores = a * tau + a * (dat @ x)
F = tau + w @ np.maximum(scores, 0.0) + eps * abs(a) * np.linalg.norm(x)
```

Note this is **exact**, not an approximation. It is specific to p1: the type-2
version (`createproblem_portLP_p2`, `DRO_eps.value = radius**2`) couples
$\lambda$ quadratically in the dual, so $\lambda^*$ is not pinned to a dual norm
and the same collapse does not hold — which is why p2 keeps solving the LP every
interval.

## 4. The closed-form Danskin subgradient

Because $F$ is convex (bare $\tau$ + sum of convex PWL pieces + convex norm) and
$\Delta$ is convex and compact, the exact DRO solution is tracked with a
projected subgradient method. The Danskin (sub)gradient is read off at the
active set — no differentiation through an optimization:

$$g_x \;=\; a\,\big(w\odot\mathbf 1_{\text{active}}\big)^\top \mathrm{dat}
\;+\; \varepsilon\,|a|\,\frac{x}{\lVert x\rVert_2},
\qquad
g_\tau \;=\; 1 \;+\; a\sum_{i\in\text{active}} w_i.$$

The `1 +` in $g_\tau$ comes from the bare leading $\tau$; the norm term in $g_x$
uses the smooth $x/\lVert x\rVert_2$ with an $\lVert x\rVert_2>0$ guard.

```python
scores = a * tau + a * (dat @ x)
active = scores > 0
gx   = a * ((w * active) @ dat)
if xnorm > 0:
    gx = gx + eps * abs_a * x / xnorm
gtau = 1.0 + a * (w @ active)
```

## 5. The projected step

One step per timestep, with Euclidean projection of $x$ onto the simplex
(`project_simplex`, the Duchi et al. 2008 sort-and-threshold) and an
unconstrained step in $\tau$:

$$x_{t+1} = \Pi_\Delta\!\big(x_t - \eta_t\,g_x\big),\qquad
\tau_{t+1} = \tau_t - \eta_t\,g_\tau,\qquad
\eta_t = \frac{\eta_0}{\sqrt{t+1}}.$$

Optional Armijo backtracking (`line_search`) shrinks $\eta$ until
$F(x_{\text{new}},\tau_{\text{new}}) \le F_{\text{curr}} - \alpha\,\eta\,\lVert g\rVert^2$,
re-evaluating the *same* closed-form $F$ — so the line search costs only cheap
numpy re-evaluations, never re-solves.

The driver constants are the textbook subgradient scalings: `D_x = sqrt(2)` is
the $\ell_2$ diameter of the simplex, `R` the mean data norm
$\overline{\lVert\xi\rVert}$, and `L_x = |a|*R` the Lipschitz constant of the
objective in $x$ (since $\partial_x[a\langle x,\xi\rangle]=a\xi$). The
commented-out `eta_0 = D_x / L_x` is the classic $D/L$ step scale.

## 6. Why one step per timestep is enough

The problem barely moves between timesteps: `num_dat` grows by one sample and the
radius `eps = init_eps * num_dat**(-1/40)` shrinks only slightly, so the optimum
drifts slowly. A single warm-started subgradient step — seeded once at `t == 0`
by the one genuine LP solve in `createproblem_portLP` — tracks it. The online
per-step cost drops from "solve a growing-size LP" to $O(Nm)$ arithmetic plus an
$O(m\log m)$ simplex projection.
