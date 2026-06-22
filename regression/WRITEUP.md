# Distributionally robust best-subset selection under perturbed covariates

This note documents the regression problem, the experimental setting, and the
synthetic data generation used by the scripts in `regression/`
([reg_orig.py](reg_orig.py), [reg_DRO_orig.py](reg_DRO_orig.py),
[utils.py](utils.py)). It is the regression counterpart of the portfolio
mean-robust-optimization (MRO) experiments in `port_new/`.

---

## 1. Problem

### 1.1 Distributionally robust best-subset selection

We observe i.i.d. covariate–response pairs $(x_i, y_i) \in \mathbb{R}^d \times
\mathbb{R}$ and form the empirical measure

$$\hat P_n = \frac{1}{n}\sum_{i=1}^n \delta_{(x_i,\,y_i)}.$$

We seek a sparse linear predictor $x \mapsto \beta^\top x$ using at most $k$
features, chosen to be robust against a worst-case perturbation of the
data-generating distribution within a Wasserstein ball. Only the **covariates**
may be perturbed; the responses are held fixed. The transport cost is therefore

$$c\big((x,y),(x',y')\big) = \|x - x'\|_q^{\,2} \;+\; \infty\cdot \mathbf 1[\,y \neq y'\,],$$

and the order-2 Wasserstein ambiguity set of radius $\delta$ is

$$\mathcal B_\delta(\hat P_n) = \big\{ P : W_c(P, \hat P_n) \le \delta \big\}.$$

The distributionally robust best-subset-selection (DRO-BSS) problem is

$$\boxed{\;\min_{\beta:\;\|\beta\|_0 \le k}\;\; \sup_{P \in \mathcal B_\delta(\hat P_n)}\; \mathbb E_P\big[(y - \beta^\top x)^2\big].\;}$$

This is the *distributional* analogue of the Bertsimas–Copenhaver
perturbed-design-matrix result: instead of an adversary moving $X$ inside an
operator-norm ball, the adversary moves the data-generating distribution inside
a Wasserstein ball whose cost is built from a covariate norm.

### 1.2 Closed-form inner maximization

By the Wasserstein-DRO duality of Blanchet–Kang–Murthy and Gao–Chen–Kleywegt,
the inner worst-case expectation has a closed form: with $\tfrac1p + \tfrac1q =
1$,

$$\sup_{P \in \mathcal B_\delta(\hat P_n)} \sqrt{\mathbb E_P\big[(y-\beta^\top x)^2\big]}
   \;=\; \sqrt{\tfrac1n \|y - X\beta\|_2^2}\;+\;\sqrt{\delta}\,\|\beta\|_p .$$

The worst-case expected loss is "empirical RMSE + dual-norm penalty," and the
Wasserstein radius $\sqrt\delta$ plays the role of a regularization weight.
Substituting back, the DRO-BSS objective becomes

$$\min_{\beta:\;\|\beta\|_0 \le k}\;\Big(\sqrt{\tfrac1n\|y-X\beta\|_2^2} + \sqrt\delta\,\|\beta\|_p\Big)^2 .$$

**We use $p = 2$ (hence $q = 2$):** the covariate transport norm is Euclidean and
the induced penalty is the $\ell_2$ (ridge-type) penalty $\sqrt\delta\,\|\beta\|_2$.
This recovers the robustness reading of a Tikhonov term, now derived from
distributional rather than fixed-matrix uncertainty. (Taking $q=\infty,\,p=1$
would instead give a cardinality-constrained square-root *lasso*; that variant
promotes sparsity, whereas $p=2$ only shrinks — see §2.4.)

### 1.3 Mixed-integer second-order-cone reformulation

For a fixed support the objective is convex in $\beta$, and the cardinality
constraint is modeled exactly with binaries, giving a mixed-integer SOCP
([`createproblem_regMIO`](utils.py)):

$$
\begin{aligned}
\min_{\beta,\,z,\,t,\,r}\quad & (t + \sqrt\delta\, r)^2 \\
\text{s.t.}\quad
& \tfrac{1}{\sqrt n}\,\|y - X\beta\|_2 \le t, \qquad \|\beta\|_2 \le r, \\
& -M z_j \le \beta_j \le M z_j, \quad j = 1,\dots,d, \\
& \textstyle\sum_{j} z_j \le k, \qquad z \in \{0,1\}^d .
\end{aligned}
$$

The optimal objective value is exactly the **worst-case mean-squared error**, so
it is directly comparable to an out-of-sample MSE. We solve with **MOSEK**
(`solver=cp.MOSEK`), big-$M = 10$, and a hard 3000 s wall-clock cap on both the
continuous and mixed-integer optimizers (`MOSEK_PARAMS`).

### 1.4 Mean-robust (clustered) variant

Mean robust optimization replaces the empirical measure by a reduced measure
supported on $K$ weighted centroids $\{(x_k^c, y_k^c)\}_{k=1}^K$ with weights
$w_k \ge 0$, $\sum_k w_k = 1$. The DRO-BSS objective with the same closed form
becomes the weighted least-squares problem

$$\min_{\beta:\;\|\beta\|_0\le k}\;\Big(\sqrt{\textstyle\sum_{k} w_k\,(y_k^c - \beta^\top x_k^c)^2} + \sqrt\delta\,\|\beta\|_2\Big)^2 ,$$

which is the same MI-SOCP with the $K$ centroids as data and $\sqrt{w_k}$ folded
inside the second-order cone (parameter `sqw` $=\sqrt{w}$). Full-data DRO is the
special case $K = n$, $w_k = 1/n$.

### 1.5 Worst-case evaluation

For a *fixed* $\beta$ the $p=2$ worst-case MSE over the ball is available in
closed form (no solver needed, [`worst_case_reg`](utils.py)):

$$V(\beta) = \Big(\sqrt{\textstyle\sum_i w_i\,(y_i-\beta^\top x_i)^2}\,+\,\sqrt\delta\,\|\beta\|_2\Big)^2 .$$

---

## 2. Experimental setting

The data arrive in a **streaming** fashion. At each timestep $t$ the sample size
$n$ grows by one (starting from `N_init`), and the methods are re-solved on the
data seen so far. Each sample/centroid is stored as a single row of length
$d+1 = m+1$: the first $m$ entries are the covariates $x$, the last is the
response $y$. All clustering operates in this joint $(x,y)$ space.

### 2.1 Methods compared

| Method | Data used | Robust? | Where |
|---|---|---|---|
| **Online MRO** | $K$ online streaming centroids | yes ($\sqrt\delta$) | [reg_orig.py](reg_orig.py) |
| **Batch MRO** | $K$ kmeans centroids (recomputed) | yes ($\sqrt\delta$) | [reg_orig.py](reg_orig.py) |
| **cluster_SAA** | same $K$ kmeans centroids | no ($\delta=0$) | [reg_orig.py](reg_orig.py) |
| **Full DRO** | all $n$ samples | yes ($\sqrt\delta$) | [reg_DRO_orig.py](reg_DRO_orig.py) |
| **SAA** | all $n$ samples | no ($\delta=0$) | [reg_DRO_orig.py](reg_DRO_orig.py) |

`cluster_SAA` reuses the batch-MRO clusters but drops the robustness term,
isolating the effect of *clustering* from the effect of *distributional
robustness*. SAA (`create_scenario_reg`) is the non-robust cardinality-constrained
least squares.

### 2.2 Online clustering

[reg_orig.py](reg_orig.py) maintains two layers (the portfolio machinery,
reused verbatim in [utils.py](utils.py)):

* a fine layer of up to $Q$ micro-clusters (`q_dict`), updated point-by-point:
  an incoming point is **absorbed** into the nearest micro-cluster when its
  distance is $\le 2\,\mathrm{rmse}$, otherwise it **spawns** a new one (with
  pairwise merging once $Q$ is exceeded);
* a coarse layer of $K$ macro-clusters (`k_dict`) obtained by clustering the
  micro-centers.

The singleton-radius floor is **data-adaptive** ($0.3\times$ the median
nearest-neighbour distance of the init centroids) so the absorption rule tracks
the data scale. After a fixed time the centers are frozen and only updated by
nearest-centroid assignment (`fixed_cluster`). Batch MRO instead re-runs
$k$-means (warm-started) each interval.

### 2.3 Radius schedule

The code variable `radius` **is** $\sqrt\delta$ (the penalty coefficient). The
RWPI theory of Blanchet–Kang–Murthy chooses $\sqrt{\delta_n}\asymp 1/\sqrt n$, so

$$\sqrt{\delta_n} \;=\; \texttt{init\_eps}\,\big/\sqrt{n},$$

with `init_eps` the swept prefactor. Empirically (for $m=10$, $k=k_{\text{true}}=5$,
moderate SNR) the validity threshold — the smallest radius for which the
worst-case objective is a valid upper bound on out-of-sample MSE — sits at
`init_eps` $\approx 1$, and the out-of-sample-MSE optimum at `init_eps`
$\approx 2$–$3$; `init_eps` $\gtrsim 5$ over-shrinks. The default sweep is
`eps_init = [3.0, 2.0, 1.5, 1.0, 0.7, 0.5]`.

### 2.4 Metrics recorded

Per timestep and per method the drivers log: the DRO/MRO objective (worst-case
MSE), out-of-sample MSE on a held-out test split, a **satisfaction** flag
(objective $\ge$ out-of-sample MSE, i.e. the bound is valid), the closed-form
worst-case value, $K$-dependent clustering value
($W_1(\hat P_n,\hat P_K)$, within-cluster RMSE), an averaged regret bound, and
solve times.

A note on $p=2$: because the $\ell_2$ penalty *shrinks* but does not *select*,
the radius improves out-of-sample MSE and bound validity but does not by itself
improve support recovery (the cardinality constraint $k$ does the selection).

---

## 3. Data generation

Synthetic sparse linear-regression data ([`generate_regression_data`](utils.py)),
the regression analogue of `port_new/synthetic_200_1.csv`. One large dataset is
generated once with a fixed seed; each experiment seed then draws its own
train/test split (`train_size = 19000`, `test_size = 1000`).

**Covariates.** $x_i \sim \mathcal N(0, \Sigma)$ with a Toeplitz correlation

$$\Sigma_{ij} = \rho^{\,|i-j|}, \qquad \rho = 0.5,$$

so neighbouring features are correlated (this is what makes support recovery
non-trivial and gives the $\ell_2$ penalty something to do).

**True coefficients.** $\beta^\star$ has exactly $k_{\text{true}}$ nonzero
entries at random positions, each $\pm\,\texttt{beta\_scale}$ with random sign
($\texttt{beta\_scale}=2$, so $\|\beta^\star\|_2 = 2\sqrt{k_{\text{true}}}$).

**Response.** $y_i = \beta^{\star\top} x_i + \varepsilon_i$, $\varepsilon_i \sim
\mathcal N(0, \texttt{noise\_std}^2)$.

Each row is stored as $[\,x_i^\top \;\; y_i\,] \in \mathbb R^{m+1}$.

### 3.1 Key parameters

| Parameter | Symbol | Default | Meaning |
|---|---|---|---|
| `m` | $d$ | 10 | number of covariates |
| `k` | $k$ | 5 | cardinality budget $\|\beta\|_0 \le k$ (estimator) |
| `k_true` | $k_{\text{true}}$ | 5 | true number of nonzero coefficients (data) |
| `K` | $K$ | 15/25/30 | number of clusters (MRO) |
| `Q` | $Q$ | 500 | micro-cluster cap (online layer) |
| `N_init` | $n_0$ | 50 | initial sample size |
| `noise_std` | $\sigma$ | 3.0 | response noise std |
| `beta_scale` | — | 2.0 | magnitude of nonzero $\beta^\star_j$ |
| `rho` | $\rho$ | 0.5 | covariate Toeplitz correlation |
| `T`, `R` | — | 3001, 5 | timesteps, random seeds |

### 3.2 Signal-to-noise and difficulty

With $\beta^\star = \pm 2$ on $k_{\text{true}}$ features and unit-variance
covariates, $\operatorname{var}(\beta^{\star\top}x)$ is on the order of a few
$\times 10$, so the variance SNR is $\operatorname{var}(\beta^{\star\top}x)/\sigma^2$.

* **Correct specification** $k = k_{\text{true}}$: the estimator's budget matches
  the truth (the default). At moderate-to-high SNR support recovery is easy, so
  the methods differ mainly in out-of-sample MSE / bound validity.
* **Harder regimes**: set $k > k_{\text{true}}$ (the estimator may over-select)
  and/or raise `noise_std` to lower the SNR. With $m=10, k=5$, SNR $\approx 0.2$–$0.4$
  (i.e. `noise_std` $\approx 7$–$10$) makes small-$n$ support recovery genuinely
  imperfect while large $n$ still recovers — the regime where the methods
  separate.

---

## 4. Visualizations

* [`visualize_data.py`](visualize_data.py) → `regression_data.png`: coefficients,
  covariate correlation, response, signal vs. fit, active/inactive scatters.
* [`visualize_clustering.py`](visualize_clustering.py) → `clustering_value.png`:
  clustering value ($W_1$, within-cluster RMSE) vs. $K$ for several $n$.
* [`visualize_radius.py`](visualize_radius.py) → `radius_selection.png`:
  out-of-sample MSE, bound validity, and $\|\hat\beta\|_2$ vs. $\sqrt\delta$.
