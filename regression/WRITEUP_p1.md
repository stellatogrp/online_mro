# Distributionally robust best-subset SVM under perturbed covariates ($p=1$, hinge loss)

This note documents the **$p=1$ hinge-loss** (sparse-SVM) variant of the
perturbed-covariates DRO best-subset problem, and the scripts that implement it
([reg_orig_p1.py](reg_orig_p1.py), [reg_DRO_orig_p1.py](reg_DRO_orig_p1.py),
[utils_p1.py](utils_p1.py)). It is the *classification* counterpart of the
$p=2$ squared-loss regression note in [WRITEUP.md](WRITEUP.md): the response is
replaced by a label $y\in\{-1,+1\}$, the squared loss by the hinge loss, the
order-2 Wasserstein ball by an order-1 ball, and the ridge-type $\ell_2$ penalty
by the sparsity-promoting $\ell_1$ penalty. The clustering, streaming, and
bookkeeping machinery is reused **verbatim** from `utils.py`.

---

## 1. Problem

### 1.1 Distributionally robust best-subset SVM

We observe i.i.d. labelled pairs $(x_i, y_i) \in \mathbb{R}^d \times \{-1,+1\}$
and form the empirical measure

$$\hat P_n = \frac{1}{n}\sum_{i=1}^n \delta_{(x_i,\,y_i)}.$$

We seek a sparse linear classifier $x \mapsto \operatorname{sign}(\beta^\top x)$
using at most $k$ features, robust against a worst-case perturbation of the
data-generating distribution within a Wasserstein ball. Only the **covariates**
may be perturbed; the labels are held fixed (flipping a label costs $\infty$).
Because the hinge loss is Lipschitz, the natural ball is the **order-1**
Wasserstein ball with transport cost

$$c\big((x,y),(x',y')\big) = \|x - x'\|_q \;+\; \infty\cdot \mathbf 1[\,y \neq y'\,],$$

$$\mathcal B_\delta(\hat P_n) = \big\{ P : W_c(P, \hat P_n) \le \delta \big\}.$$

The distributionally robust best-subset SVM (DRO-BSS-SVM) is

$$\boxed{\;\min_{\beta:\;\|\beta\|_0 \le k}\;\; \sup_{P \in \mathcal B_\delta(\hat P_n)}\; \mathbb E_P\big[\,(1 - y\,\beta^\top x)_+\,\big],\;}$$

where $(a)_+ = \max(0,a)$ is the hinge loss. This is the cardinality-constrained,
distributionally robust analogue of the soft-margin SVM.

### 1.2 Closed-form inner maximization

The hinge loss $x \mapsto (1 - y\,\beta^\top x)_+$ is Lipschitz in $x$ with
modulus $\|\beta\|_p$ relative to $\|\cdot\|_q$ (its gradient, where active, is
$-y\beta$ and $|y|=1$, so the $\|\cdot\|_q$-Lipschitz constant is the dual norm
$\|\beta\|_p$). For a Lipschitz loss and an order-1 Wasserstein ball, the
Wasserstein-DRO duality of Shafieezadeh-Abadeh–Kuhn–Esfahani (regularization via
mass transportation), Gao–Kleywegt, and Blanchet–Kang–Murthy gives the exact
closed form, with $\tfrac1p + \tfrac1q = 1$:

$$\sup_{P \in \mathcal B_\delta(\hat P_n)} \mathbb E_P\big[(1 - y\,\beta^\top x)_+\big]
   \;=\; \frac1n\sum_{i=1}^n (1 - y_i\,\beta^\top x_i)_+ \;+\; \delta\,\|\beta\|_p .$$

The worst-case expected loss is "empirical hinge + dual-norm penalty," and the
Wasserstein radius $\delta$ is *itself* the regularization weight — there is no
square root, in contrast to the $p=2$ regression case where $\sqrt\delta$
appears and the objective is squared. Substituting back, the DRO-BSS-SVM is

$$\min_{\beta:\;\|\beta\|_0 \le k}\;\Big(\tfrac1n\sum_{i=1}^n (1 - y_i\,\beta^\top x_i)_+ + \delta\,\|\beta\|_p\Big) .$$

**We use $p = 1$ (hence $q = \infty$):** the covariate transport norm is
$\ell_\infty$ and the induced penalty is the $\ell_1$ (lasso-type) penalty
$\delta\,\|\beta\|_1$. This is the robustness reading of $\ell_1$-regularized
SVM, derived from distributional uncertainty. Unlike the $p=2$ ridge penalty —
which only *shrinks* — the $\ell_1$ penalty also *selects*, so here the radius
$\delta$ and the cardinality budget $k$ **both** push toward sparsity (see §2.4).
(Taking $q=2,\,p=2$ would instead give an $\ell_2$-regularized robust SVM.)

### 1.3 Mixed-integer linear reformulation

Both the hinge loss and the $\ell_1$ penalty are piecewise linear, so for a
fixed support the objective is a linear program; the cardinality constraint is
modeled exactly with binaries, giving a mixed-integer **LP**
([`createproblem_hingeMIO`](utils_p1.py)):

$$
\begin{aligned}
\min_{\beta,\,z,\,\xi}\quad & \tfrac1n\sum_{i=1}^n \xi_i \;+\; \delta\,\|\beta\|_1 \\
\text{s.t.}\quad
& \xi_i \ge 1 - y_i\,\beta^\top x_i, \qquad \xi_i \ge 0, \quad i = 1,\dots,n, \\
& -M z_j \le \beta_j \le M z_j, \quad j = 1,\dots,d, \\
& \textstyle\sum_{j} z_j \le k, \qquad z \in \{0,1\}^d .
\end{aligned}
$$

(In code the hinge epigraph $\xi$ and the $\ell_1$ epigraph are formed
automatically by CVXPY; the objective is written directly as
`w @ pos(1 - y .* (X beta)) + eps * norm(beta, 1)`.) The optimal objective value
is exactly the **worst-case expected hinge loss**, directly comparable to an
out-of-sample hinge loss. We solve with **MOSEK** (`solver=cp.MOSEK`), big-$M = 10$,
and a hard 3000 s wall-clock cap (`MOSEK_PARAMS`). This is a MILP, whereas the
$p=2$ regression model is a MI-SOCP.

### 1.4 Mean-robust (clustered) variant

Mean robust optimization replaces the empirical measure by a reduced measure
supported on $K$ weighted centroids $\{(x_k^c, y_k^c)\}_{k=1}^K$ with weights
$w_k \ge 0$, $\sum_k w_k = 1$. The DRO-BSS-SVM objective with the same closed
form becomes the weighted-hinge problem

$$\min_{\beta:\;\|\beta\|_0\le k}\;\Big(\textstyle\sum_{k} w_k\,(1 - y_k^c\,\beta^\top x_k^c)_+ + \delta\,\|\beta\|_1\Big) ,$$

the same MILP with the $K$ centroids as data and the weights $w_k$ entering
**linearly** (parameter `w` $=w$ — passed directly, *not* square-rooted as in
the SOCP). Full-data DRO is the special case $K = n$, $w_k = 1/n$.

Clustering is **label-pure**: every cluster contains points of a single label,
so each centroid label $y_k^c$ is exactly $\pm1$ (not a fractional average). The
two labels *share* the cluster budget — the number of macro-clusters never
exceeds $K$ and is split across the labels in proportion to their counts — so
each centroid is an unambiguous prototype of one class and the hinge surrogate
$(1 - y_k^c\,\beta^\top x_k^c)_+$ behaves exactly like a per-class data point.

### 1.5 Worst-case evaluation

For a *fixed* $\beta$ the worst-case expected hinge over the ball is available in
closed form (no solver needed, [`worst_case_hinge`](utils_p1.py)):

$$V(\beta) = \textstyle\sum_i w_i\,(1 - y_i\,\beta^\top x_i)_+ \;+\; \delta\,\|\beta\|_1 .$$

---

## 2. Experimental setting

The data arrive in a **streaming** fashion. At each timestep $t$ the sample size
$n$ grows by one (starting from `N_init`), and the methods are re-solved on the
data seen so far. Each sample/centroid is stored as a single row of length
$d+1 = m+1$: the first $m$ entries are the covariates $x$, the last is the
label $y$. All clustering operates in this joint $(x,y)$ space.

### 2.1 Methods compared

| Method | Data used | Robust? | Where |
|---|---|---|---|
| **Online MRO** | $K$ online streaming centroids | yes ($\delta$) | [reg_orig_p1.py](reg_orig_p1.py) |
| **Batch MRO** | $K$ kmeans centroids (recomputed) | yes ($\delta$) | [reg_orig_p1.py](reg_orig_p1.py) |
| **cluster_SAA** | same $K$ kmeans centroids | no ($\delta=0$) | [reg_orig_p1.py](reg_orig_p1.py) |
| **Full DRO** | all $n$ samples | yes ($\delta$) | [reg_DRO_orig_p1.py](reg_DRO_orig_p1.py) |
| **SAA** | all $n$ samples | no ($\delta=0$) | [reg_DRO_orig_p1.py](reg_DRO_orig_p1.py) |

`cluster_SAA` reuses the batch-MRO clusters but drops the robustness term,
isolating the effect of *clustering* from the effect of *distributional
robustness*. SAA (`create_scenario_hinge`) is the non-robust
cardinality-constrained hinge minimization.

### 2.2 Online clustering

[reg_orig_p1.py](reg_orig_p1.py) maintains the same two-layer online clustering
as the regression code, but made **label-aware** in [utils_p1.py](utils_p1.py)
(the loss-agnostic distance / regret machinery of [utils.py](utils.py) is still
reused):

* a fine layer of up to $Q$ micro-clusters (`q_dict`), updated point-by-point:
  an incoming point is **absorbed** into the nearest *same-label* micro-cluster
  when its distance is $\le 2\,\mathrm{rmse}$, otherwise it **spawns** a new one;
  once $Q$ is exceeded the closest *same-label* micro-pair is merged;
* a coarse layer of up to $K$ macro-clusters (`k_dict`) obtained by clustering
  the micro-centers of each label separately.

Crucially the two labels **share** both budgets: the total micro-cluster count
never exceeds $Q$ and the total macro-cluster count never exceeds $K$, each split
across the labels in proportion to their counts (`_split_budget`, largest-
remainder apportionment with at least one cluster per present label). Because
absorption, spawning, and merging are all restricted to a single label, every
micro- and macro-cluster stays **label-pure**.

The singleton-radius floor is data-adaptive ($0.3\times$ the median
nearest-neighbour distance of the init centroids). After a fixed time the centers
are frozen and only updated by nearest *same-label* centroid assignment
(`fixed_cluster`). Batch MRO instead re-runs label-pure $k$-means each interval
(`label_aware_kmeans`: the $K$ budget split by label, each class clustered
separately).

### 2.3 Radius schedule

The code variable `radius` **is** $\delta$ (the penalty coefficient itself,
*not* $\sqrt\delta$). The RWPI theory of Blanchet–Kang–Murthy chooses an order-1
radius $\delta_n \asymp 1/\sqrt n$, so

$$\delta_n \;=\; \texttt{init\_eps}\,\big/\sqrt{n},$$

with `init_eps` the swept prefactor. The default sweep is
`eps_init = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3]`; the range brackets the validity
threshold (the smallest radius for which the worst-case hinge objective is a
valid upper bound on the out-of-sample hinge loss) up through the
out-of-sample optimum; large `init_eps` over-shrinks $\beta$ toward $0$.

### 2.4 Metrics recorded

Per timestep and per method the drivers log: the DRO/MRO objective (worst-case
hinge), out-of-sample hinge loss on a held-out test split, a **satisfaction**
flag (objective $\ge$ out-of-sample hinge, i.e. the bound is valid), the
closed-form worst-case value, $K$-dependent clustering value
($W_1(\hat P_n,\hat P_K)$, within-cluster RMSE), an averaged regret bound, and
solve times. The CSV schema is identical to the $p=2$ regression drivers, so the
existing [plots.py](plots.py) consumes it unchanged.
[`evaluate_misclassification_hinge`](utils_p1.py) provides the $0$–$1$ test
error as a convenience metric (not used by the hinge-based satisfaction check).

A note on $p=1$: because the $\ell_1$ penalty *selects* (drives coefficients to
exactly zero), increasing the radius $\delta$ both improves out-of-sample
performance / bound validity **and** sparsifies $\hat\beta$ — so the radius can
aid support recovery even before the cardinality constraint $k$ binds. This is
the qualitative difference from the $p=2$ ridge case, where the radius only
shrinks.

---

## 3. Data generation

Synthetic sparse linear-**classification** data
([`generate_classification_data`](utils_p1.py)). One large dataset is generated
once with a fixed seed; each experiment seed then draws its own train/test split
(`train_size = 19000`, `test_size = 1000`).

**Covariates.** $x_i \sim \mathcal N(0, \Sigma)$ with a Toeplitz correlation

$$\Sigma_{ij} = \rho^{\,|i-j|}, \qquad \rho = 0.5,$$

so neighbouring features are correlated.

**True coefficients.** $\beta^\star$ has exactly $k_{\text{true}}$ nonzero
entries at random positions, each $\pm\,\texttt{beta\_scale}$ with random sign
($\texttt{beta\_scale}=2$).

**Labels.** $y_i = \operatorname{sign}(\beta^{\star\top} x_i + \varepsilon_i)$,
$\varepsilon_i \sim \mathcal N(0, \texttt{noise\_std}^2)$ — the noisy linear
(probit-style) labelling rule, so `noise_std` controls the label-flip rate
(lower noise $\Rightarrow$ wider margin $\Rightarrow$ easier separation). Because
$x$ is zero-mean and the noise is symmetric, the Bayes boundary passes through
the origin and the **intercept-free** classifier $\operatorname{sign}(\beta^\top x)$
is appropriate (matching the intercept-free regression model; an unpenalized bias
would be a straightforward extension).

Each row is stored as $[\,x_i^\top \;\; y_i\,] \in \mathbb R^{m+1}$.

### 3.1 Key parameters

| Parameter | Symbol | Default | Meaning |
|---|---|---|---|
| `m` | $d$ | 10 | number of covariates |
| `k` | $k$ | 5 | cardinality budget $\|\beta\|_0 \le k$ (estimator) |
| `k_true` | $k_{\text{true}}$ | 5 | true number of nonzero coefficients (data) |
| `K` | $K$ | 10/25 | number of clusters (MRO) |
| `Q` | $Q$ | 500 | micro-cluster cap (online layer) |
| `N_init` | $n_0$ | 50 | initial sample size |
| `noise_std` | $\sigma$ | 3.0 | label-noise std (flip rate) |
| `beta_scale` | — | 2.0 | magnitude of nonzero $\beta^\star_j$ |
| `rho` | $\rho$ | 0.5 | covariate Toeplitz correlation |
| `T`, `R` | — | 3001, 5 | timesteps, random seeds |

### 3.2 Margin / difficulty

With $\beta^\star = \pm 2$ on $k_{\text{true}}$ features and `noise_std` $=3$ the
Bayes error is moderate (a few tens of percent), so small-$n$ classifiers are
genuinely imperfect while large $n$ recovers — the regime where the methods
separate. Lowering `noise_std` widens the margin (easier, near-separable);
raising it (or setting $k>k_{\text{true}}$) makes recovery harder.

---

## 4. Relation to the $p=2$ regression note

| | $p=2$ regression ([WRITEUP.md](WRITEUP.md)) | $p=1$ classification (this note) |
|---|---|---|
| Response / label | $y\in\mathbb R$ | $y\in\{-1,+1\}$ |
| Per-sample loss | squared $(y-\beta^\top x)^2$ | hinge $(1-y\,\beta^\top x)_+$ |
| Wasserstein order | 2 (cost $\|\cdot\|_q^2$) | 1 (cost $\|\cdot\|_q$) |
| Penalty | $\sqrt\delta\,\|\beta\|_2$ (ridge, shrinks) | $\delta\,\|\beta\|_1$ (lasso, selects) |
| `radius` variable | $\sqrt\delta$ | $\delta$ |
| Worst-case form | $(\text{RMSE}+\sqrt\delta\|\beta\|_2)^2$ | $\text{hinge}+\delta\|\beta\|_1$ |
| Reformulation | MI-SOCP | MILP |
| Weights in solver | $\sqrt{w}$ (inside cone) | $w$ (linear) |

The streaming protocol, online/batch clustering, regret bookkeeping, CSV schema,
and the five compared methods are otherwise identical, so the two notes can be
plotted and compared with the same tooling.
