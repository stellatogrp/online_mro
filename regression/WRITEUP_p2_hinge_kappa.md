# DRO sparse-SVM with finite label transport cost $\kappa$

This note documents the **finite-$\kappa$** variant of the perturbed-covariates
DRO best-subset SVM.  It keeps the hinge loss, the order-1 Wasserstein ball, and
the $\ell_2$ covariate transport norm of [WRITEUP_p2_hinge.md](WRITEUP_p2_hinge.md),
but **replaces the infinite label transport cost** with a finite penalty $\kappa$:

$$c\big((x,y),(x',y')\big) = \|x - x'\|_2 \;+\; \kappa\,|y - y'|.$$

When $\kappa \to \infty$ labels cannot move and we recover the $p=2$ hinge model
exactly.  For finite $\kappa$ the inner supremum no longer has the simple
closed form $\text{hinge} + \delta\|\beta\|_2$; instead the Wasserstein dual
yields a mixed-integer SOCP (Corollary 3.12).

---

## 1. Problem

### 1.1 Transport cost and ambiguity set

Labelled data $(x_i, y_i) \in \mathbb{R}^d \times \{-1,+1\}$ with empirical
measure $\hat P_n$.  The order-1 Wasserstein ball with cost $c$ above is

$$\mathcal B_\delta(\hat P_n) = \big\{ P : W_c(P,\hat P_n) \le \delta \big\}.$$

The DRO best-subset SVM is

$$\min_{\beta:\;\|\beta\|_0 \le k}\;\; \sup_{P \in \mathcal B_\delta(\hat P_n)}\; \mathbb E_P\big[(1 - y\,\beta^\top x)_+\big].$$

### 1.2 Wasserstein dual — splitting by label

The dual of the inner supremum assigns, for each sample $(x_i, y_i)$, a worst-case
point $(x', y')$ at transport cost $\le \lambda$ (the Wasserstein dual multiplier).
Since $y' \in \{-1,+1\}$ there are exactly two cases:

**Same label** ($y' = y_i$, transport cost $\|x_i - x'\|_2$):

$$g_i^+(\beta, \lambda) = \sup_{x'}\bigl[(1 - y_i\,\beta^\top x')_+ - \lambda\|x_i - x'\|_2\bigr].$$

**Label flip** ($y' = -y_i$, transport cost $\|x_i - x'\|_2 + 2\kappa$):

$$g_i^-(\beta, \lambda) = \sup_{x'}\bigl[(1 + y_i\,\beta^\top x')_+ - \lambda\|x_i - x'\|_2\bigr] - 2\kappa\lambda.$$

Both suprema equal their linear part minus the SOC bound when $\|\beta\|_2 \le \lambda$
(and $+\infty$ otherwise):

$$g_i^+(\beta,\lambda) = (1 - y_i\,\beta^\top x_i)_+, \quad
  g_i^-(\beta,\lambda) = (1 + y_i\,\beta^\top x_i)_+ - 2\kappa\lambda,
  \quad \text{if } \|\beta\|_2 \le \lambda.$$

The per-sample contribution is $g_i = \max(g_i^+, g_i^-)$, and the full dual is

$$\min_{\lambda \ge \|\beta\|_2}\; \lambda\delta \;+\; \frac{1}{n}\sum_{i=1}^n
  \max\!\bigl((1 - m_i)_+,\;(1 + m_i - 2\kappa\lambda)_+\bigr),$$

where $m_i = y_i\,\beta^\top x_i$ is the margin of sample $i$.

**Comparison with $\kappa = \infty$.**  As $\kappa \to \infty$ the flip term
vanishes for any finite $\lambda$, so the min over $\lambda$ is attained at
$\lambda = \|\beta\|_2$ and V reduces to the $p=2$ closed form
$\delta\|\beta\|_2 + \frac{1}{n}\sum_i (1-m_i)_+$.

### 1.3 MI-SOCP reformulation (Corollary 3.12)

Introducing epigraph slacks $s_i$ and treating $\lambda$ as a decision variable,
the combined minimization over $(\beta, \lambda, s)$ with the cardinality
constraint gives the mixed-integer SOCP:

$$
\begin{aligned}
\min_{\beta,\,z,\,\lambda,\,s}\quad & \lambda\,\delta \;+\; \frac{1}{N}\textstyle\sum_{i=1}^N s_i \\
\text{s.t.}\quad
& s_i \;\ge\; 1 - y_i\,\beta^\top x_i, \qquad i \in [N], \\
& s_i \;\ge\; 1 + y_i\,\beta^\top x_i \;-\; 2\kappa\lambda, \qquad i \in [N], \\
& s_i \;\ge\; 0, \qquad i \in [N], \\
& \|\beta\|_2 \;\le\; \lambda, \qquad \lambda \ge 0, \\
& -M z_j \;\le\; \beta_j \;\le\; M z_j, \quad j = 1,\dots,d, \\
& \textstyle\sum_j z_j \;\le\; k, \qquad z \in \{0,1\}^d.
\end{aligned}
$$

The only SOCP constraint is $\|\beta\|_2 \le \lambda$; everything else is linear,
so the problem is an MI-SOCP solved with MOSEK.

**Weighted (clustered) variant.**  For the mean-robust / clustered formulations,
replace $(1/N)\sum_i s_i$ by $\sum_k w_k s_k$ where $w_k$ are the cluster weights.
The problem structure is unchanged; $N$ becomes the number of centroids $K$.

### 1.4 Worst-case evaluation for a fixed $\beta$

For a fixed $\beta$ the worst-case value is a scalar convex minimization over
$\lambda \ge \|\beta\|_2$:

$$V(\beta,\delta,\kappa) \;=\; \min_{\lambda \ge \|\beta\|_2} \;
   \lambda\,\delta \;+\;
   \sum_i w_i\,\max\!\bigl((1-m_i)_+,\;(1+m_i-2\kappa\lambda)_+\bigr).$$

The objective is piecewise linear and convex in $\lambda$; `worst_case_hinge_kappa`
in [`utils_p1.py`](utils_p1.py) finds the minimum with `scipy.optimize.minimize_scalar`
(bounded 1-D search, no solver needed).

---

## 2. Comparison with the other models

| | $p=1$ hinge ([WRITEUP_p1.md](WRITEUP_p1.md)) | $p=2$ hinge, $\kappa=\infty$ ([WRITEUP_p2_hinge.md](WRITEUP_p2_hinge.md)) | **$p=2$ hinge, finite $\kappa$ (this note)** |
|---|---|---|---|
| Label transport cost | $\infty$ | $\infty$ | $\kappa < \infty$ |
| Covariate transport norm | $\ell_\infty$ | $\ell_2$ | $\ell_2$ |
| Inner sup closed form | yes: $\delta\|\beta\|_1$ | yes: $\delta\|\beta\|_2$ | no — MI-SOCP needed |
| Reformulation | MILP | MI-SOCP | MI-SOCP |
| Label flips penalized? | $\infty$ (impossible) | $\infty$ (impossible) | $2\kappa\lambda$ (finite) |
| Recovers $\kappa=\infty$ case? | — | — | yes |

The key practical difference is that the closed-form penalty disappears; the
scalar $\lambda$ is now an additional decision variable that trades off radius
cost against the benefit of allowing label flips.

---

## 3. Experimental setting

Identical to [WRITEUP_p2_hinge.md](WRITEUP_p2_hinge.md) with the following
additions.

### 3.1 New parameter: $\kappa$

| Parameter | Symbol | Default | Meaning |
|---|---|---|---|
| `kappa` | $\kappa$ | 1.0 | Label transport cost per unit $|y - y'|$ |

Large $\kappa$ (e.g., $\kappa = 100$) effectively prohibits label flips and
matches the $\kappa = \infty$ (p=2 hinge) behavior.  Small $\kappa$ allows
the adversary to flip labels cheaply, yielding a more conservative classifier.

### 3.2 Radius schedule

Same RWPI schedule as the other hinge notes:
$\delta_n = \texttt{init\_eps} / \sqrt{n}$ (i.e., `power = 0.5`).

### 3.3 Methods compared

| Method | Data used | Robust? | File |
|---|---|---|---|
| **Online MRO** | $K$ online streaming centroids | yes ($\delta, \kappa$) | [reg_orig_p2.py](reg_orig_p2.py) |
| **Batch MRO** | $K$ kmeans centroids (recomputed) | yes ($\delta, \kappa$) | [reg_orig_p2.py](reg_orig_p2.py) |
| **cluster\_SAA** | same $K$ kmeans centroids | no ($\delta=0$) | [reg_orig_p2.py](reg_orig_p2.py) |
| **Full DRO** | all $n$ samples | yes ($\delta, \kappa$) | [reg_DRO_orig_p2.py](reg_DRO_orig_p2.py) |
| **SAA** | all $n$ samples | no ($\delta=0$) | [reg_DRO_orig_p2.py](reg_DRO_orig_p2.py) |

`cluster_SAA` and `SAA` are unchanged from the $\kappa=\infty$ case (they have
no robustness term and so are independent of $\kappa$).

---

## 4. Implementation notes

`createproblem_hingeMIO_kappa(N, m, k)` in [`utils_p1.py`](utils_p1.py) builds
the MI-SOCP above.  It exposes four CVXPY `Parameter` objects:
- `dat` — data matrix $(N, m+1)$
- `eps` — Wasserstein radius $\delta$
- `kappa_param` — label transport cost $\kappa$
- `w` — sample/cluster weights

`worst_case_hinge_kappa(dat, m, beta, delta, kappa, weights=None)` evaluates
$V(\beta,\delta,\kappa)$ via a bounded 1-D scipy search (no CVXPY call needed).

The drivers [`reg_orig_p2.py`](reg_orig_p2.py) and
[`reg_DRO_orig_p2.py`](reg_DRO_orig_p2.py) are direct adaptations of their
`_p1` counterparts with:
1. `createproblem_hingeMIO` $\to$ `createproblem_hingeMIO_kappa` (sets `kappa_param.value = kappa`)
2. `worst_case_hinge` $\to$ `worst_case_hinge_kappa`
3. `--kappa` added as a CLI argument (default 1.0)
4. `--p` removed (the norm is fixed to $\ell_2$ on covariates)
