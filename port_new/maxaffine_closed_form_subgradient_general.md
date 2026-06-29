# Closed-form W1-DRO subgradient for a max-of-affine loss

This note describes a closed-form projected-subgradient scheme for type-1
Wasserstein DRO whenever the loss is a finite max of functions that are affine in
the uncertain parameter. It is written for a general decision variable and a
general convex feasible set; the portfolio problem with simplex constraints is
one special case (Section 7). The conclusion: the worst-case **value** is
available in closed form, and the **subgradient step** needs no inner convex
solve — only a per-sample argmax and a max over pieces.

## 1. Setup

Let the decision be $\theta\in\Theta$, where $\Theta\subseteq\mathbb R^D$ is a
closed convex set onto which Euclidean projection $\Pi_\Theta$ is available
(closed-form or cheap). Let $\xi\in\mathbb R^d$ be the uncertain parameter with
**unbounded support** $\mathbb R^d$. The loss is a finite max of affine-in-$\xi$
pieces,

$$g(\xi;\theta) \;=\; \max_{k\in[K]} \big(b_k(\theta)^\top \xi + c_k(\theta)\big),$$

where the slopes $b_k(\theta)\in\mathbb R^d$ and intercepts $c_k(\theta)\in\mathbb R$
may depend on the decision. The ambiguity set is the type-1 Wasserstein ball
$B_1(\hat P_N,\varepsilon)$ of radius $\varepsilon>0$ around the empirical measure
$\hat P_N=\sum_i w_i\,\delta_{\hat\xi_i}$ ($w_i\ge 0$, $\sum_i w_i=1$), built from
a ground norm $\lVert\cdot\rVert$ on $\mathbb R^d$ with dual norm $\lVert\cdot\rVert_*$.
The DRO problem is

$$\min_{\theta\in\Theta}\;\;
\sup_{Q\,:\,W_1(Q,\hat P_N)\le\varepsilon}\;\mathbb E_Q\big[g(\xi;\theta)\big].$$

Any part of the objective that does not depend on $\xi$ (a deterministic decision
term) passes straight through the expectation and is carried along unchanged.

## 2. The worst-case value is closed form

For fixed $\theta$, the integrand $g(\cdot;\theta)$ is convex and Lipschitz in
$\xi$, with Lipschitz constant equal to the largest dual-norm slope,

$$\mathrm{Lip}_\xi(g) \;=\; \max_{k\in[K]} \lVert b_k(\theta)\rVert_*,$$

(the gradient of $g$ at any point is one of the slopes $b_k$; see Section 4 for
why the *dual* norm). The type-1 worst-case expectation of a convex Lipschitz
loss over unbounded support is exactly empirical risk plus a linear penalty:

$$\boxed{\;
\sup_{Q\,:\,W_1(Q,\hat P_N)\le\varepsilon}\mathbb E_Q[g(\xi;\theta)]
\;=\;
\underbrace{\sum_i w_i\,\max_{k}\big(b_k(\theta)^\top\hat\xi_i + c_k(\theta)\big)}_{\text{empirical risk}}
\;+\;
\varepsilon\,\max_{k}\lVert b_k(\theta)\rVert_*
\;}$$

This is the clean form stated as Proposition 6.17 in Kuhn, Shafiee & Wiesemann
(2025), $\;\mathbb E_{\hat P}[\ell(\hat Z)] + r\,\mathrm{lip}(\ell)$, specialized
to $\mathrm{lip}(\ell)=\max_k\lVert b_k\rVert_*$. For the max-of-affine case it is
also the original Mohajerin Esfahani–Kuhn (2018) worst-case reformulation, whose
dual LP carries exactly one norm constraint per piece. The unbounded-support
assumption is what makes this an equality rather than an upper bound; on a
constrained support one falls back to the $\inf_\lambda$ form of Section 9 with a
soft per-sample penalty.

## 3. Worst-case distribution and attainment

The value identity of Section 2 is exact unconditionally. Whether a worst-case
distribution $Q^*$ **attaining** that supremum exists is a separate question, and
the answer has a clean characterization for the affine case.

### 3.1 Structure of the optimal transport plan

From the dual derivation (Section 4), at $\lambda^*=\max_k\lVert b_k\rVert_*$
the adversary moves each atom $\hat\xi_i$ by a displacement $s_i u_i^*$ along the
**dual-norm-steepest direction** of its active piece,

$$\zeta_i^* = \hat\xi_i + s_i\,u_i^*,
\qquad
u_i^* \in \operatorname*{arg\,max}_{\lVert u\rVert\le 1} b_{k(i)}^\top u,$$

where $k(i)=\operatorname*{arg\,max}_k(b_k^\top\hat\xi_i+c_k)$ is the active piece
at sample $i$ and the non-negative displacements $s_i\ge 0$ exhaust the budget:
$\sum_i w_i s_i=\varepsilon$. The worst-case $Q^*$ is therefore a **deterministic
shift** of each empirical atom — when attainment holds.

### 3.2 Three failure modes

**1. Polyhedral ground norm — non-uniqueness but attainment still holds.**
$u_i^*$ exists for any norm (dual-ball is compact), but need not be unique. Under
a smooth/strictly-convex norm ($\ell_2$) the dual-norm-attaining direction is
unique: $u^*=b/\lVert b\rVert_2$, so $Q^*$ is unique given the active set. Under
a polyhedral norm ($\ell_1$, $\ell_\infty$) the argmax is a face of the dual
ball; any selection from that face, or any mixture, attains the same value.
Attainment holds, but $Q^*$ is not unique.

**2. Ties in the active piece — mass splitting.**
At a sample $\hat\xi_i$ where two pieces tie,
$b_j^\top\hat\xi_i+c_j=b_k^\top\hat\xi_i+c_k$, the loss has a kink and the two
slope directions $u_j^*,u_k^*$ may point in different directions. The
supremum may require splitting atom $i$'s mass and sending fractions toward each
slope direction; a single deterministic shift cannot attain it. A two-point (or
multi-point) conditional $Q_i^*$ does, so attainment holds with a discrete $Q^*$,
but the worst-case distribution is generically not a pure shift at kink samples.

**3. Genuine non-attainment — the maximizing sequence escapes to infinity.**
This does not occur for max-of-affine losses on $\mathbb R^d$ (see §3.3), but
arises for losses that only *asymptotically* approach their Lipschitz slope. In
that case the value formula still gives the correct supremum, but the supremum is
a limit approached by a sequence $Q_m^*$ that sends mass $\varepsilon/m$ to
distance $m\to\infty$; no weak-limit $Q^*$ inside the Wasserstein ball attains it.

### 3.3 Why max-of-affine losses always attain $Q^*$

The key condition is whether the loss attains its Lipschitz slope at **finite
displacement** or only in the limit. Huang, Li, and Mao (2026) distinguish two
regimes (their conditions (8)/(11) vs. (20)):

- **Asymptotic-slope condition** (necessary for the value formula; holds for any
  convex Lipschitz $f$):
  $$\lim_{m\to\infty}\frac{1}{m}\big(f(\xi_0+m v)-f(\xi_0)\big)=\mathrm{Lip}(f).$$
- **Finite-step condition** (sufficient for attainment; strictly stronger):
  $$\sup_{\lVert y\rVert\le\varepsilon}\big(f(\xi_0+y)-f(\xi_0)\big)
  =\mathrm{Lip}(f)\cdot\varepsilon
  \qquad\text{for every }\varepsilon>0.$$

A max-of-affine loss satisfies the finite-step condition: each piece $b_k^\top\xi$
has constant slope $b_k$ everywhere, so $b_k^\top(u^*\varepsilon)=\lVert b_k\rVert_*\varepsilon$
is realized at the finite displacement $y=u^*\varepsilon$ for any $\varepsilon>0$.
The Lipschitz slope is therefore attained immediately, not merely approached, and
the worst-case $Q^*$ — a finite shift of the atoms — is an exact maximizer.

Losses that only satisfy the asymptotic condition but not the finite-step condition
(e.g. smooth convex losses such as softplus or log-sum-exp that are asymptotically
linear, or the "convex tail" losses in Huang–Li–Mao Example 4) still have the same
closed-form value but fail to attain $Q^*$: realizing the full slope requires
$\lVert u\rVert\to\infty$, so the value is a supremum, not a maximum.

**Bottom line.** For a max-of-affine loss on $\mathbb R^d$, the worst-case
distribution is always attained. It is a finite deterministic shift of each atom
along the dual-norm-steepest direction of its active piece, with mass splitting
permitted at kink ties. Uniqueness holds under a strictly convex ground norm and
can fail (a face of optimal transport plans) under polyhedral norms.

## 4. Derivation notes

The penalty $\varepsilon\max_k\lVert b_k\rVert_*$ is $\lambda^*\varepsilon$, where
$\lambda^*$ is the optimal value of the **dual multiplier on the single
Wasserstein-budget constraint** $W_1(Q,\hat P_N)\le\varepsilon$. Writing $Q$
through its transport plan from the atoms $\hat\xi_i$ and attaching $\lambda\ge 0$
to that one budget constraint, strong duality (Esfahani–Kuhn) gives

$$\sup_{Q\,:\,W_1\le\varepsilon}\mathbb E_Q[g]
=\inf_{\lambda\ge 0}\Big\{\lambda\varepsilon
+\sum_i w_i\,\sup_{\zeta}\big[\,g(\zeta;\theta)-\lambda\lVert\zeta-\hat\xi_i\rVert\,\big]\Big\}.$$

The $\lambda\varepsilon$ term is "multiplier $\times$ right-hand side." Plugging in
the max-of-affine loss and shifting $\zeta=\hat\xi_i+u$, each inner sup splits per
piece into

$$\sup_u\big[b_k^\top u - \lambda\lVert u\rVert\big]
=\begin{cases}0,&\lVert b_k\rVert_*\le\lambda\\[2pt]+\infty,&\lVert b_k\rVert_*>\lambda,\end{cases}$$

so finiteness forces the per-piece constraints $\lambda\ge\lVert b_k\rVert_*$ for
all $k$, and on that set the inner sup reduces to $g(\hat\xi_i;\theta)$. The dual
becomes $\inf_{\lambda\ge\max_k\lVert b_k\rVert_*}\{\lambda\varepsilon+\sum_i w_i\,g(\hat\xi_i)\}$,
which is increasing in $\lambda$, so

$$\lambda^*=\max_k\lVert b_k(\theta)\rVert_*=\mathrm{Lip}_\xi(g),
\qquad\text{penalty}=\lambda^*\varepsilon.$$

(At general $p$ the budget is $W_p\le\varepsilon$, the constant term is
$\lambda\varepsilon^p$, and the conjugate becomes a soft penalty instead of the
hard $0/\infty$ switch — which is why the clean collapse is a $p=1$ phenomenon.)

## 5. Closed-form Danskin subgradient

The outer objective is

$$F(\theta) \;=\; \big(\text{deterministic term}\big)
\;+\; \sum_i w_i\,\max_{k}\big(b_k(\theta)^\top\hat\xi_i + c_k(\theta)\big)
\;+\; \varepsilon\,\max_{k}\lVert b_k(\theta)\rVert_*,$$

convex in $\theta$ whenever each piece and each $\lVert b_k(\theta)\rVert_*$ is
convex in $\theta$ (in particular when $b_k,c_k$ are affine in $\theta$). The
subgradient is read off at the active pieces (Danskin), with no differentiation
through any optimization.

**Empirical term — per-sample argmax.** For each sample select the dominating
piece $k_i^* = \arg\max_k(b_k(\theta)^\top\hat\xi_i + c_k(\theta))$ and use its
gradient in $\theta$:

$$g_\theta^{\text{emp}} = \sum_i w_i\Big(\partial_\theta b_{k_i^*}(\theta)^\top\hat\xi_i + \partial_\theta c_{k_i^*}(\theta)\Big).$$

This is the $K$-way generalization of a binary active/inactive mask; cost
$O(NK)$ evaluations, no solve.

**Regularizer term — max over pieces.** Select $k^*=\arg\max_k\lVert b_k(\theta)\rVert_*$
and take

$$g_\theta^{\text{reg}} = \varepsilon\,\partial_\theta\lVert b_{k^*}(\theta)\rVert_*,$$

e.g. for the Euclidean dual norm, $\partial\lVert b\rVert_2=b/\lVert b\rVert_2$
($b\neq 0$) composed with $\partial_\theta b_{k^*}(\theta)$ via the chain rule.

**Projected step.** With $g_\theta = (\text{det. term gradient}) + g_\theta^{\text{emp}} + g_\theta^{\text{reg}}$,

$$\theta_{t+1} = \Pi_\Theta\!\big(\theta_t - \eta_t\,g_\theta\big),
\qquad \eta_t = \frac{\eta_0}{\sqrt{t+1}},$$

with $\Pi_\Theta$ the Euclidean projection onto the feasible set and optional
Armijo backtracking evaluated against the same closed-form $F$ (cheap
re-evaluations, never re-solves). Diminishing steps give standard subgradient
convergence on the convex $F$ over convex $\Theta$.

## 6. Smoothness in the decision

The value identity is exact for any $K$. The only geometric subtlety is in
$\theta$:

- The empirical term is piecewise-smooth, with kinks where two pieces tie for a
  sample, $b_j^\top\hat\xi_i+c_j=b_k^\top\hat\xi_i+c_k$.
- The regularizer $\varepsilon\max_k\lVert b_k(\theta)\rVert_*$ is piecewise-smooth,
  with kinks where two pieces tie for the largest dual norm,
  $\lVert b_j\rVert_*=\lVert b_k\rVert_*$, plus the norm kink at $b_{k^*}=0$.

A subgradient method needs no smoothness, so this is not an obstacle; one only
loses uniqueness of the gradient at the ties.

**Fast path — shared slope direction.** If all slopes are scalar multiples of a
common decision-dependent vector, $b_k(\theta)=a_k\,v(\theta)$ with scalars
$a_k$, then $\max_k\lVert a_k v(\theta)\rVert_* = (\max_k|a_k|)\,\lVert v(\theta)\rVert_*$:
the max over pieces collapses to a single constant times one norm, the
regularizer has no extra kinks, and only the empirical-term argmax is needed.
The general nonsmooth regularizer appears only when the pieces depend on $\theta$
through structurally different maps (e.g. $b_k(\theta)=A_k\theta$ with distinct
$A_k$) so that the functions $\lVert b_k(\theta)\rVert_*$ actually cross.

## 7. Special case: portfolio with simplex constraints

Take $\theta=(x,\tau)$ with $x\in\Delta=\{x\ge 0,\ \mathbf 1^\top x=1\}$ and
$\tau\in\mathbb R$, ground norm $\ell_2$, and the two-piece CVaR-style loss

$$g(\xi;x,\tau)=\tau+\max\big(0,\;a\tau+a\langle\xi,x\rangle\big),$$

i.e. $b_1=0,\ c_1=0$ and $b_2=ax,\ c_2=a\tau$, with a bare $\tau$ as the
deterministic term. Then $\max_k\lVert b_k\rVert_2=\lVert ax\rVert_2=|a|\lVert x\rVert_2$,
so $\lambda^*=|a|\lVert x\rVert_2$ and the value identity reads
$\tau+\sum_i w_i\max(0,a\tau+a\hat\xi_i^\top x)+\varepsilon|a|\lVert x\rVert_2$.
The empirical argmax reduces to `active = scores > 0`, the regularizer is the
smooth $\varepsilon|a|\,x/\lVert x\rVert_2$ (shared-direction fast path), and
$\Pi_\Theta$ is the Euclidean simplex projection (Duchi et al. 2008) in $x$ with
an unconstrained step in $\tau$. This is exactly the scheme implemented in
`dro_subgrad_step` / `worst_case_value`.

## 8. Relation to projection equivalence and other risk measures

A convex max-of-affine loss with pieces normalized to a common dual-norm slope,
$\lVert b_k\rVert_*=c_f$, is precisely the canonical representation
$f=\max_{i}\{c_f\beta_i^\top\xi+b_i\}$ that characterizes the complete convex
class admitting exact regularization (Huang–Li–Mao 2026, Prop. 3; type-1
coherent regime of Thm. 4). The same closed-form scheme extends from the
expectation to worst-case CVaR or convex distortion risk at $p=1$, with the
penalty constant scaled by $\tfrac{1}{1-\alpha}$ for CVaR$_\alpha$ or
$\lVert h'\rVert_\infty$ for a distortion $\rho_h$ — the empirical-risk-plus-
$\varepsilon\cdot$Lip structure, and hence the subgradient step, is unchanged.

## 9. Support constraints

When the uncertain parameter is restricted to a closed convex set
$\Xi\subsetneq\mathbb R^d$ (a box, a polytope, a norm ball, etc.), the clean
collapse of Section 2 fails: the supremum in the dual is over $\Xi$, not over
all of $\mathbb R^d$, so the $0/\infty$ switch that pinned $\lambda^*$ to the
Lipschitz constant no longer operates.

### 8.1 How the dual changes

Strong duality (Esfahani–Kuhn 2018, Thm. 4.2) still gives

$$\sup_{Q\,:\,W_1(Q,\hat P_N)\le\varepsilon,\,\text{supp}(Q)\subseteq\Xi}
\mathbb E_Q[g]
\;=\;
\inf_{\lambda\ge 0}\Big\{
  \lambda\varepsilon
  +\sum_i w_i\,\phi_\lambda(\hat\xi_i)
\Big\},$$

where the per-sample oracle is now the **constrained** augmented loss

$$\phi_\lambda(\hat\xi_i)
\;=\;\sup_{\zeta\in\Xi}\big[\,g(\zeta;\theta)-\lambda\lVert\zeta-\hat\xi_i\rVert\,\big].$$

Because $g$ is bounded on the compact set $\Xi$, $\phi_\lambda$ is finite for
every $\lambda\ge 0$ and is convex and nonincreasing in $\lambda$. Consequently
the 1D dual objective $h(\lambda)=\lambda\varepsilon+\sum_i w_i\phi_\lambda(\hat\xi_i)$
is convex in $\lambda$ and the infimum is attained at some $\lambda^*\in[0,\max_k\lVert
b_k\rVert_*]$. Evaluating the dual at $\lambda=\max_k\lVert b_k\rVert_*$ recovers
exactly the unbounded formula (since $\phi_{\max_k\lVert b_k\rVert_*}(\hat\xi_i)=g(\hat\xi_i;\theta)$
for any $\hat\xi_i\in\Xi$, by the same $0/\infty$ argument), so

$$\text{exact worst-case value} \;\le\;
\underbrace{\sum_i w_i\,\max_k(b_k^\top\hat\xi_i+c_k)
+\varepsilon\max_k\lVert b_k\rVert_*}_{\text{unbounded formula, now an upper bound.}}$$

The bound is tight when the adversary's optimal perturbation in the unconstrained
problem already lands inside $\Xi$ — concretely, when $\varepsilon$ is small
relative to the slack between each $\hat\xi_i$ and $\partial\Xi$.

### 8.2 Inner problems for max-of-affine losses

Plugging in the max-of-affine form,

$$\phi_\lambda(\hat\xi_i)
=\sup_{\zeta\in\Xi}\max_k\big(\,b_k^\top\zeta+c_k-\lambda\lVert\zeta-\hat\xi_i\rVert\big)
=\max_k\;\sup_{\zeta\in\Xi}\big(\,b_k^\top\zeta+c_k-\lambda\lVert\zeta-\hat\xi_i\rVert\big).$$

The exchange of max and sup holds because the inner problems share no coupling
across pieces. Each per-piece problem $\sup_{\zeta\in\Xi}[b_k^\top\zeta-\lambda\lVert\zeta-\hat\xi_i\rVert]$
is a DC (difference-of-convex) program; its tractability depends on $\Xi$ and
the ground norm.

**Box support, $\ell_1$ ground norm.** $\Xi=[\ell,u]$, $\lVert\cdot\rVert=\lVert\cdot\rVert_1$,
dual norm $\lVert\cdot\rVert_\infty$. The problem decouples by coordinate:
for each $j$,

$$\sup_{\xi_j\in[\ell_j,u_j]}\big[(b_k)_j\,\xi_j-\lambda|\xi_j-\hat\xi_{i,j}|\big].$$

This 1D problem has the closed-form maximizer
$\xi_j^*=\hat\xi_{i,j}$ when $|(b_k)_j|\le\lambda$ (no movement pays), and
$\xi_j^*=u_j$ or $\ell_j$ (whichever gives the larger value) when $|(b_k)_j|>\lambda$.

**Polyhedral support, $\ell_1$ ground norm.** $\Xi=\{A\zeta\le a\}$,
$\lVert\cdot\rVert=\lVert\cdot\rVert_1$. Writing $\lVert\zeta-\hat\xi_i\rVert_1=\mathbf 1^\top t$
with $-t\le\zeta-\hat\xi_i\le t$, $t\ge 0$, each per-piece inner sup becomes
an LP in $(\zeta,t)$. The full dual is then $NK$ LPs (one per sample per piece)
plus a 1D convex search over $\lambda$.

**Polyhedral support, $\ell_2$ ground norm.** Each inner problem is a QP over
$\Xi$. The KKT stationarity condition is $b_k=\lambda(\zeta^*-\hat\xi_i)/\lVert\zeta^*-\hat\xi_i\rVert_2+A^\top\mu$
(with $\mu\ge 0$ the LP multiplier); for a box, the problem again separates
by coordinate and yields a clipped analogue of the unconstrained optimal
$\hat\xi_i+b_k/\lambda$.

### 8.3 Solving the 1D dual

The dual objective $h(\lambda)$ is convex and its subgradient is

$$h'(\lambda)=\varepsilon-\sum_i w_i\,\lVert\zeta_i^*(\lambda)-\hat\xi_i\rVert,$$

where $\zeta_i^*(\lambda)$ is any maximizer of $\phi_\lambda(\hat\xi_i)$. The
optimal $\lambda^*$ equates the marginal budget cost $\varepsilon$ to the
weighted average transport distance used by the adversary. Special cases:
- $\lambda^*=0$: the adversary's budget exceeds the diameter of $\Xi$ (it can
  move every atom to the global worst point within $\Xi$ for free); the worst-case
  value is simply $\max_{\zeta\in\Xi}g(\zeta;\theta)$.
- $\lambda^*=\max_k\lVert b_k\rVert_*$: the support constraint never binds
  (tight budget or large slack), recovering the unbounded formula exactly.

The search can be carried out by bisection or golden section on the compact
interval $[0,\max_k\lVert b_k\rVert_*]$.

### 8.4 Subgradient under support constraints

**Exact route.** At each $\theta$, solve the 1D problem for $\lambda^*(\theta)$
and obtain $\zeta_i^*(\lambda^*)$ for every sample. By Danskin's theorem applied
to both the inner sup over $\Xi$ and the outer inf over $\lambda$, the subgradient
of the exact worst-case value w.r.t. $\theta$ is

$$g_\theta = \sum_i w_i\,\partial_\theta g\!\left(\zeta_i^*;\theta\right)\big|_{\text{active piece}},$$

with $\zeta_i^*=\zeta_i^*(\lambda^*(\theta))$. The regularizer-style term $\varepsilon\,\partial_\theta\lambda^*$ vanishes because $\lambda^*$ is the minimizer of $h$ (envelope theorem). The computational overhead versus the unconstrained scheme is the $\lambda$-search and the constrained inner sups; the outer projected step $\theta_{t+1}=\Pi_\Theta(\theta_t-\eta_t g_\theta)$ is unchanged.

**Conservative (upper-bound) route.** Use the closed-form scheme of Sections 2–5
verbatim. The resulting objective is the upper bound above, which is a valid
DRO surrogate and converges to the same optimum whenever the support constraint
does not bind at optimality (e.g. $\varepsilon$ small or $\Xi$ large relative to
the optimal perturbation).

## 10. Comparison with Chen–Fattahi–Shafiee (2026)

Chen, Fattahi, and Shafiee (2026) study Wasserstein DRO in an online (sequential)
setting and develop an efficient oracle for the inner worst-case expectation problem
for **piecewise-concave** losses. This section clarifies how their setting and method
differ structurally from the max-of-affine framework developed above.

### 9.1 The key distinction: convex vs. concave pieces in $\xi$

Both frameworks consider a max-of-pieces loss, but the pieces play opposite roles:

- **This note**: $g(\xi;\theta)=\max_k(b_k(\theta)^\top\xi+c_k(\theta))$, each piece
  **affine** (hence convex) in $\xi$.
- **Chen–Fattahi–Shafiee**: $\ell(x,\xi)=\max_{k\in[K]}\ell_k(x,\xi)$, each piece
  $\ell_k$ **concave and differentiable** in $\xi$ (their Assumption 2).

This single difference drives every computational distinction. For an affine piece, the
adversarial shift $\sup_\zeta[b_k^\top(\zeta-\hat\xi_i)-\lambda\lVert\zeta-\hat\xi_i\rVert]$
is the norm-conjugate operation and evaluates to $0$ if $\lVert b_k\rVert_*\le\lambda$
and $+\infty$ otherwise — the $0/\infty$ switch. For a concave piece, no such switch
exists: $\sup_\zeta[\ell_k(\zeta)-\lambda\lVert\zeta-\hat\xi_i\rVert^p]$ is finite and
genuinely nontrivial for every $\lambda\ge 0$ because the adversary always gains by
moving $\zeta$ toward the peak of $\ell_k$, regardless of how large the transport
penalty $\lambda$ is.

### 9.2 Support and growth conditions

- **This note (Sections 1–8)**: unbounded support $\Xi=\mathbb R^d$; linear growth
  in $\xi$ (affine pieces). The $0/\infty$ switch pins $\lambda^*=\max_k\lVert b_k\rVert_*$
  and yields the closed form of Section 2.
- **This note (Section 9)**: bounded support $\Xi\subsetneq\mathbb R^d$; affine
  pieces. Bounded support prevents the $0/\infty$ switch, requiring a 1D search.
- **Chen–Fattahi–Shafiee**: unbounded support $\Xi=\mathbb R^m$ (their Assumption 5);
  **sublinear growth** $\ell(x,\xi)\le g(1+\lVert\xi\rVert^r)$, $r\in(0,1)$. The
  sublinear growth is what keeps the dual inner sup finite on all of $\mathbb R^m$
  without needing a compact $\Xi$. Affine losses (linear growth, $r=1$) explicitly
  violate their Assumption 5 — their oracle is not designed for the max-of-affine
  case.

### 9.3 Inner problem structure

The three regimes produce qualitatively different per-sample subproblems:

| Setting | Per-sample inner problem | Character | Cost |
|---|---|---|---|
| Affine pieces, $\Xi=\mathbb R^d$ (§2) | Collapses to $g(\hat\xi_i;\theta)$ | Closed form | $O(K)$ |
| Affine pieces, bounded $\Xi$ (§9) | $\sup_{\zeta\in\Xi}[g(\zeta;\theta)-\lambda\lVert\zeta-\hat\xi_i\rVert]$ | DC program; LP for polyhedral $\Xi$+$\ell_1$ | $NK$ LPs + 1D search |
| Concave pieces, $\Xi=\mathbb R^m$ (CFS) | $S_i^{(k_1,k_2)}(b)$: concave max over conic set | Concave program; golden section | $O(K^2\cdot\mathrm{Cost}_k\cdot\mathrm{polylog}(1/\delta))$ |

The concave-piece inner problems in Chen–Fattahi–Shafiee are concave
maximizations (perspective of a concave function over a conic set), which
golden-section methods handle efficiently. The affine-piece problems in Section 9
are DC maximizations (convex function minus convex function), which require LP or
QP structure to be tractable.

### 9.4 Dual decomposition structure

- **Section 9 (affine, bounded $\Xi$)**: a single 1D bisection over
  $\lambda\in[0,\max_k\lVert b_k\rVert_*]$.
- **Chen–Fattahi–Shafiee**: a two-level search. The worst-case expectation is
  reformulated as a **budget allocation problem** — allocate total transport budget
  $\rho t$ as $(b_1,\ldots,b_t)$ across the $t$ empirical samples, each $b_i\ge0$, to
  maximize $\frac{1}{t}\sum_i S_i(b_i)$ (their Theorem 7). Dual decomposition then
  bisects over the shadow price $\lambda$ of the budget constraint, solving the $t$
  decoupled problems $\max_{b\ge0}\{S_i(b)-\lambda b\}$ via a nested golden-section
  search. The budget-allocation structure is specific to piecewise-concave losses:
  the concave utility functions $S_i(b)$ are strictly increasing in budget, so the
  adversary non-trivially spreads transport mass across all samples. For max-of-affine
  losses on unbounded support the degenerate analogue would concentrate all budget on
  the sample/piece with the largest slope, recovering the closed form directly.

### 9.5 Summary of differences

| Dimension | This note (§1–8) | This note (§9) | Chen–Fattahi–Shafiee |
|---|---|---|---|
| Loss type | Max-of-affine (convex in $\xi$) | Max-of-affine | Max-of-concave |
| Support $\Xi$ | $\mathbb R^d$ (unbounded) | Bounded convex | $\mathbb R^m$ (unbounded) |
| Growth | Linear ($r=1$) | Linear ($r=1$) | Sublinear ($r<1$) |
| Dual collapse | Full (closed form) | Partial (1D search) | None (budget allocation) |
| Inner solve | None | LP/QP per piece | Concave QP per pair per sample |
| Gradient step | $O(NK)$ argmax | $O(NK)$ LPs + bisection | $O(K^2 t\,\mathrm{polylog})$ per oracle call |

## 11. Summary

For a max-of-affine loss over a general convex feasible set, the worst-case value
is the empirical risk plus $\varepsilon\max_k\lVert b_k(\theta)\rVert_*$, with the
Wasserstein multiplier pinned to $\lambda^*=\max_k\lVert b_k(\theta)\rVert_*=\mathrm{Lip}_\xi(g)$
(Kuhn–Shafiee–Wiesemann 2025, Prop. 6.17; Esfahani–Kuhn 2018). The projected
subgradient step needs only a per-sample argmax and a max over pieces — no inner
solve — and reduces to the portfolio/simplex implementation as one special case.

### References
- P. Mohajerin Esfahani and D. Kuhn (2018). Data-driven distributionally robust optimization using the Wasserstein metric. *Mathematical Programming* 171:115–166.
- R. Gao and A. Kleywegt (2023). Distributionally robust stochastic optimization with Wasserstein distance. *Mathematics of Operations Research*.
- S. Shafieezadeh-Abadeh, D. Kuhn, P. Mohajerin Esfahani (2019). Regularization via mass transportation. *JMLR* 20(103).
- D. Kuhn, S. Shafiee, W. Wiesemann (2025). Distributionally robust optimization. *Acta Numerica* 34:579–804. (Prop. 6.17.)
- C. Huang, J. Y.-M. Li, T. Mao (2026). When Wasserstein DRO reduces exactly: complete characterizations of projection equivalence and regularization.
- J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra (2008). Efficient projections onto the ℓ1-ball for learning in high dimensions. *ICML*.
