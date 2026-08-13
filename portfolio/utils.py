"""Utilities for the box-support W1-DRO portfolio-CVaR example.

Problem (continuous decision, type-1 Wasserstein with the L2 ground norm,
**box support** on the uncertain parameter)
--------------------------------------------------------------------------
    decision   theta = (x, tau),   x in the simplex  Delta = {x>=0, 1^T x = 1},
                                    tau in R
    uncertain  xi = asset returns, restricted to the box  Xi = [lb, ub]
    loss       g(xi; x, tau) = tau + max(0, a*tau + a * <xi, x>),   a = -1/(1-alpha)

``g`` is the CVaR-style max-of-affine loss of Section 7 of
``maxaffine_closed_form_subgradient_general.md`` (two affine pieces
b_1 = 0, c_1 = 0 and b_2 = a*x, c_2 = a*tau, plus a deterministic tau).  Unlike
the portfolio example in ``port_new`` -- which has *unbounded* support and hence
the clean closed-form penalty  eps*|a|*||x||_2 -- here the support is a **box**,
so that closed form is only an UPPER BOUND (note Section 9, eq. after 9.1).

The exact worst-case value uses:
  * Section 9.5  -- box support, L2 ground norm: the per-sample inner problem
    sup_{zeta in [lb,ub]} [b^T zeta + c - lam ||zeta - xi||_2] solved by the
    clipped-step parametric bisection on the multiplier mu;
  * Section 9.9  -- the outer 1D dual  min_{lam>=0} lam*eps + sum_i w_i phi_lam(xi_i)
    solved by bisection on  h'(lam) = eps - sum_i w_i ||zeta_i^*(lam) - xi_i||_2;
  * Section 9.10 (exact route) -- the projected subgradient step: move each atom
    to its box-constrained worst-case location zeta_i^*(lam^*) and read off the
    Danskin subgradient at the active piece (the eps*d(lam^*) term vanishes by the
    envelope theorem).

``createproblem_box_DRO`` / ``worst_case_value_box_socp`` give the exact
Esfahani-Kuhn SOCP reformulation as an independent reference for validation.
"""
import json
import os
import time

import cvxpy as cp
import joblib
import numpy as np

# Optional C fast path for worst_case_value_box (portfolio/cworst/).  The numpy
# body below stays untouched as the verification oracle; the dispatch at the
# top of worst_case_value_box uses the C kernel when the compiled library is
# available unless env PORT_WORSTCASE_IMPL == "python".
try:
    from . import cworst as _cworst
except ImportError:      # utils imported outside the package
    try:
        import cworst as _cworst
    except ImportError:
        _cworst = None


# --------------------------------------------------------------------------- #
# Generic helpers (shared with the port_new / regression style drivers)
# --------------------------------------------------------------------------- #
def get_n_processes(max_n=np.inf):
    try:
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()
    return max(min(max_n, n_cpus), 1)


def save_run_metadata(metadata, paths):
    lines = ["# Run metadata", ""]
    for k, v in metadata.items():
        if isinstance(v, (list, tuple)):
            lines.append(f"- {k}: {', '.join(str(x) for x in v)}")
        else:
            lines.append(f"- {k}: {v}")
    txt = "\n".join(lines) + "\n"
    for entry in paths:
        if isinstance(entry, str):
            d, jname = entry, 'metadata.json'
        else:
            d, jname = entry
            jname = jname or 'metadata.json'
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, jname), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        with open(os.path.join(d, 'metadata.txt'), 'w') as f:
            f.write(txt)


def remove_files(paths):
    """Remove exactly the given files, skipping any that don't exist.

    Used at the end of a run to discard the specific now-redundant
    intermediate files that *this* run generated, without touching anything
    else that may share the same directory (e.g. another concurrent run's
    output, or files left over from an unrelated experiment)."""
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def safe_solve(problem, name='?', t=None, **solve_kwargs):
    try:
        problem.solve(**solve_kwargs)
    except Exception as e:
        print(f"[safe_solve] {name} t={t} raised: {type(e).__name__}: {e}", flush=True)
        return False
    status = getattr(problem, 'status', None)
    if status not in ('optimal', 'optimal_inaccurate'):
        print(f"[safe_solve] {name} t={t} status={status}", flush=True)
        return False
    return True


def project_simplex(v):
    """Euclidean projection onto {x >= 0, sum x = 1} (Duchi et al. 2008)."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, n + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


# --------------------------------------------------------------------------- #
# Synthetic bounded-return data
# --------------------------------------------------------------------------- #
def generate_returns(n_total, m, seed=0, n_factors=3, box_q=0.40, n_hist=5000):
    """A factor-plus-idiosyncratic return panel, clipped to a per-asset box.

    Returns ``(data, lb, ub)`` where ``data`` is ``(n_total, m)`` and
    ``[lb, ub]`` is the per-asset support box.

    The box bounds are estimated from a separate *historical* draw of ``n_hist``
    observations (same model parameters, earlier in the RNG stream) and then
    applied to the ``n_total`` experiment observations.  This mimics the
    real-world practice of inferring price-limit bands from a historical window
    before running the online experiment — the box is not data-snooped from the
    same samples seen by the algorithm.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-0.01, 0.04, m)                  # per-asset mean return
    fac_load = rng.uniform(0.3, 1.2, (m, n_factors))  # factor loadings
    idio_std = rng.uniform(0.02, 0.06, m)             # per-asset idio. vol

    # --- historical window: used only to set lb / ub ---
    fac_hist = rng.standard_normal((n_hist, n_factors)) * 0.03
    idio_hist = rng.standard_normal((n_hist, m)) * idio_std
    hist = mu + fac_hist @ fac_load.T + idio_hist
    lb = np.quantile(hist, box_q, axis=0)
    ub = np.quantile(hist, 1.0 - box_q, axis=0)

    # --- experiment window: fresh draws, clipped to the historical box ---
    fac = rng.standard_normal((n_total, n_factors)) * 0.03
    idio = rng.standard_normal((n_total, m)) * idio_std
    data = mu + fac @ fac_load.T + idio
    data = np.clip(data, lb, ub)
    return data, lb, ub


# --------------------------------------------------------------------------- #
# Section 9.5: box + L2 inner worst-case solve for the single affine piece
# --------------------------------------------------------------------------- #
def _worstcase_piece2(dat, b, c2, lam, lb, ub, mu_iter=80):
    """Section 9.5 per-sample inner problem for the affine piece  b^T zeta + c2:

        phi(i) = sup_{zeta in [lb,ub]}  b^T zeta + c2 - lam * ||zeta - dat_i||_2.

    ``b`` (d,), scalars ``c2``, ``lam`` are shared across the ``N`` samples
    ``dat`` (N, d); ``lb``/``ub`` (d,) are the box.  Solved by the clipped-step
    bisection on the per-sample multiplier mu (self-consistency
    mu ||zeta*(mu) - xi|| = lam), vectorized across samples.

    Returns ``(phi (N,), zeta_star (N, d), r (N,))`` with r = ||zeta*-xi||_2.
    """
    dat = np.asarray(dat, dtype=float)
    N, d = dat.shape
    nb = float(np.linalg.norm(b))
    # ||b||_2 <= lam  (Cauchy-Schwarz): moving never beats the transport cost.
    if nb == 0.0 or lam >= nb:
        return dat @ b + c2, dat.copy(), np.zeros(N)
    zbar = np.where(b > 0.0, ub, lb)                  # gradient-favorable corner
    rmax = np.linalg.norm(zbar - dat, axis=1)         # (N,)
    if lam <= 1e-15:
        # zero transport cost: jump straight to the favorable corner
        zeta = np.broadcast_to(zbar, (N, d)).copy()
        return zeta @ b + c2 - lam * rmax, zeta, rmax
    # g(mu) = mu ||zeta*(mu)-xi|| - lam is nondecreasing; root in [lam/rmax, hi]
    mu_lo = lam / np.maximum(rmax, 1e-12)
    mu_hi = np.full(N, nb / 1e-9)
    mu_hi = np.maximum(mu_hi, mu_lo)                  # guard atoms already at the corner
    for _ in range(mu_iter):
        mu = 0.5 * (mu_lo + mu_hi)
        zeta = np.clip(dat + b[None, :] / mu[:, None], lb, ub)
        r = np.linalg.norm(zeta - dat, axis=1)
        pos = (mu * r - lam) > 0.0
        mu_hi = np.where(pos, mu, mu_hi)
        mu_lo = np.where(pos, mu_lo, mu)
    mu = 0.5 * (mu_lo + mu_hi)
    zeta = np.clip(dat + b[None, :] / mu[:, None], lb, ub)
    r = np.linalg.norm(zeta - dat, axis=1)
    return zeta @ b + c2 - lam * r, zeta, r


def worst_case_value_box(x, tau, dat, w, eps, lb, ub, a=-5.0,
                         lam_iter=60, mu_iter=80, return_state=False):
    """Exact box-support W1 worst-case value of the CVaR loss at (x, tau).

    Sections 9.5 + 9.9:  min_{lam in [0, ||a x||_2]} lam*eps + sum_i w_i phi_lam(xi_i),
    where phi_lam(xi_i) = max(0, [piece-2 box inner value]).  Adds the
    deterministic tau.  With ``return_state`` also returns ``(lam*, zeta*, active)``
    for the Section-9.10 subgradient (``active_i`` = piece 2 active at zeta_i^*).
    """
    # C kernel is THE implementation (see portfolio/cworst/); the numpy body
    # below is retained as the verification oracle only, reachable via env
    # PORT_WORSTCASE_IMPL=python (used by the unit tests).  A missing C build
    # is an error, not a silent 240x slowdown.
    if os.environ.get("PORT_WORSTCASE_IMPL") != "python":
        if _cworst is None or not _cworst.available():
            raise RuntimeError(
                "C worst-case kernel not built: run `sh portfolio/cworst/"
                "build.sh` (or set PORT_WORSTCASE_IMPL=python to use the "
                "verification-only numpy oracle).")
        return _cworst.worst_case_value_box_c(
            x, tau, dat, w, eps, lb, ub, a=a, lam_iter=lam_iter,
            mu_iter=mu_iter, return_state=return_state, inner_mode=1)
    x = np.asarray(x, dtype=float)
    dat = np.asarray(dat, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    b = a * x
    c2 = a * tau
    nb = float(np.linalg.norm(b))

    if eps <= 0.0 or nb == 0.0:                       # empirical (non-robust) value
        phi2 = dat @ b + c2
        active = phi2 > 0.0
        F = tau + float(w @ np.maximum(phi2, 0.0))
        if return_state:
            return F, 0.0, dat.copy(), active
        return F

    def eval_lam(lam):
        phi2, zeta, r = _worstcase_piece2(dat, b, c2, lam, lb, ub, mu_iter)
        active = phi2 > 0.0                            # piece 2 (move) beats piece 1 (0)
        hval = lam * eps + float(w @ np.where(active, phi2, 0.0))
        hprime = eps - float(w @ np.where(active, r, 0.0))
        return hval, hprime, zeta, active

    # h convex on [0, ||b||]; h' nondecreasing.  h'(||b||)=eps>0, so root in [0,||b||)
    # unless the budget already exceeds the box diameter (then lam*=0).
    _, hp0, _, _ = eval_lam(0.0)
    if hp0 >= 0.0:
        lam_star = 0.0
    else:
        lo, hi = 0.0, nb
        for _ in range(lam_iter):
            mid = 0.5 * (lo + hi)
            _, hpm, _, _ = eval_lam(mid)
            if hpm > 0.0:
                hi = mid
            else:
                lo = mid
        lam_star = 0.5 * (lo + hi)

    hval, _, zeta, active = eval_lam(lam_star)
    F = tau + hval
    if return_state:
        return F, lam_star, zeta, active
    return F


def worst_case_value_unbounded(x, tau, dat, w, eps, a=-5.0):
    """Closed-form UNBOUNDED-support value (Section 2) -- a valid upper bound on
    the box value; equal to it when the box never binds."""
    x = np.asarray(x, dtype=float)
    scores = a * tau + a * (dat @ x)
    return tau + float(w @ np.maximum(scores, 0.0)) + eps * abs(a) * float(np.linalg.norm(x))


# --------------------------------------------------------------------------- #
# Section 9.10: projected-subgradient step (exact route, box support)
# --------------------------------------------------------------------------- #
def box_dro_subgrad_step(x, tau, dat, w, eps, lb, ub, eta, a=-5.0,
                         line_search=False, ls_alpha=1e-4, ls_shrink=0.5,
                         ls_max_iters=20, eta_min=1e-12, lam_iter=60, mu_iter=80):
    """One projected-subgradient step on the box-support W1-DRO objective.

    Exact route (Section 9.10): solve for lam^*(theta) and the box-constrained
    worst-case atoms zeta_i^*; the Danskin subgradient reads off the active piece
    at each zeta_i^* (the eps*d(lam^*) term vanishes by the envelope theorem):

        grad_x   = a * sum_i w_i active_i * zeta_i^*
        grad_tau = 1 + a * sum_i w_i active_i.

    Returns ``(x_new, tau_new, F)`` with ``F`` the worst-case value at the
    *current* iterate (before the step).
    """
    x = np.asarray(x, dtype=float)
    dat = np.asarray(dat, dtype=float)
    F, lam_star, zeta, active = worst_case_value_box(
        x, tau, dat, w, eps, lb, ub, a=a, lam_iter=lam_iter, mu_iter=mu_iter,
        return_state=True)
    xnorm = float(np.linalg.norm(x))
    # Regime split.  When the support box does not bind, lam* sits at the upper
    # bracket ||a x|| and the inner maximizer is non-unique (Cauchy-Schwarz tie):
    # the box solver returns the "stay" atom, which would drop the regularizer
    # from the Danskin subgradient.  Detect this by comparing the exact box value
    # to the unbounded closed form; if they agree, use the unbounded subgradient
    # a * sum_i w_i active_i xi_i + eps|a| x/||x||_2 (Section 5).  Otherwise the
    # box binds (interior lam*) and the moved atoms zeta_i^* give the exact
    # Danskin subgradient with the eps*d(lam*) term killed by the envelope theorem.
    F_unb = worst_case_value_unbounded(x, tau, dat, w, eps, a=a)
    box_slack = (eps <= 0.0) or (xnorm == 0.0) or (F >= F_unb - 1e-9 * (1.0 + abs(F_unb)))
    if box_slack and eps > 0.0 and xnorm > 0.0:
        scores = a * tau + a * (dat @ x)
        act = scores > 0.0
        grad_x = a * ((w * act) @ dat) + eps * abs(a) * x / xnorm
        grad_tau = 1.0 + a * float(w @ act)
    else:
        grad_x = a * ((w * active) @ zeta)
        grad_tau = 1.0 + a * float(w @ active)

    def _step(eta_val):
        return project_simplex(x - eta_val * grad_x), tau - eta_val * grad_tau

    if not line_search:
        x_new, tau_new = _step(eta)
        return x_new, tau_new, F

    grad_sq = float(grad_x @ grad_x + grad_tau ** 2)
    if grad_sq == 0.0:
        return x.copy(), tau, F
    x_new, tau_new = _step(eta)
    for _ in range(ls_max_iters):
        F_new = worst_case_value_box(x_new, tau_new, dat, w, eps, lb, ub, a=a,
                                     lam_iter=lam_iter, mu_iter=mu_iter)
        if F_new <= F - ls_alpha * eta * grad_sq:
            break
        eta *= ls_shrink
        if eta < eta_min:
            break
        x_new, tau_new = _step(eta)
    return x_new, tau_new, F


# --------------------------------------------------------------------------- #
# Non-DRO baseline (SAA) and exact SOCP references
# --------------------------------------------------------------------------- #
def create_scenario(dat, m, num_dat, a=-5.0):
    """Sample-average (non-robust) CVaR problem: eps -> 0 limit of the DRO one.

        min_{x in Delta, tau}  tau + (1/n) sum_i max(0, a*tau + a*<dat_i, x>).
    """
    tau = cp.Variable()
    x = cp.Variable(m)
    objective = cp.sum(tau + cp.maximum(a * tau + a * (dat @ x), 0.0)) / num_dat
    constraints = [cp.sum(x) == 1, x >= 0, x <= 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau


def createproblem_box_DRO(N, m, lb, ub, a=-5.0):
    """Exact box-support W1-DRO as an Esfahani-Kuhn SOCP (reference solver).

    For each sample the two affine pieces give:  piece 1 (b=0, c=0) -> s_i >= 0;
    piece 2 (b=a*x, c=a*tau) -> s_i >= a*tau + a*<xi_i,x> + gamma+^T(ub-xi_i)
    + gamma-^T(xi_i-lb) with ||a*x - (gamma+ - gamma-)||_2 <= lam.  Objective
    tau + eps*lam + sum_i w_i s_i.  Parameters ``dat``, ``eps``, ``w`` are set by
    the caller.  ``lb``/``ub`` are baked in as constants.
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dat = cp.Parameter((N, m))
    eps = cp.Parameter(nonneg=True)
    w = cp.Parameter(N, nonneg=True)

    x = cp.Variable(m)
    tau = cp.Variable()
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(N)
    gpos = cp.Variable((N, m), nonneg=True)
    gneg = cp.Variable((N, m), nonneg=True)

    box_term = (cp.sum(cp.multiply(ub[None, :] - dat, gpos), axis=1)
                + cp.sum(cp.multiply(dat - lb[None, :], gneg), axis=1))
    constraints = [cp.sum(x) == 1, x >= 0, x <= 1, s >= 0,
                   s >= a * tau + a * (dat @ x) + box_term]
    for i in range(N):
        constraints += [cp.norm(a * x - (gpos[i, :] - gneg[i, :]), 2) <= lam]
    objective = tau + eps * lam + w @ s
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau, dat, eps, w


def worst_case_value_box_socp(x, tau, dat, w, eps, lb, ub, a=-5.0, solver=cp.CLARABEL):
    """Exact inner worst-case value at a FIXED (x, tau) via the Esfahani-Kuhn
    SOCP dual -- an independent reference for ``worst_case_value_box``."""
    dat = np.asarray(dat, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    N, m = dat.shape
    b = a * np.asarray(x, dtype=float)
    c2 = a * tau
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(N)
    gpos = cp.Variable((N, m), nonneg=True)
    gneg = cp.Variable((N, m), nonneg=True)
    box_term = (cp.sum(cp.multiply(ub[None, :] - dat, gpos), axis=1)
                + cp.sum(cp.multiply(dat - lb[None, :], gneg), axis=1))
    constraints = [s >= 0, s >= c2 + dat @ b + box_term]
    for i in range(N):
        constraints += [cp.norm(b - (gpos[i, :] - gneg[i, :]), 2) <= lam]
    problem = cp.Problem(cp.Minimize(eps * lam + w @ s), constraints)
    problem.solve(solver=solver)
    return tau + float(problem.value)


# --------------------------------------------------------------------------- #
# Out-of-sample evaluation / satisfaction (mirrors port_new)
# --------------------------------------------------------------------------- #
def _expected_cost(d_eval, x, tau, a=-5.0):
    """Mean CVaR loss  E[g] = E[max(tau, (1+a)tau + a <d,x>)]."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.maximum(tau, (1.0 + a) * tau + a * (d_eval @ x))))


def compute_cumulative_regret(history, dateval, m, a=-5.0):
    """Out-of-sample cost and satisfaction (worst-case obj >= OOS cost) for the
    DRO and SAA iterates, over two held-out evaluation windows."""
    DRO_e, DRO_s, SA_e, SA_s = [], [], [], []
    T = len(history['t'])
    for j in range(2):
        DRO_eval = np.zeros(T)
        SA_eval = np.zeros(T)
        eval_samples = dateval[(j * 200):(j + 1) * 200, :m]
        for t in range(T):
            DRO_eval[t] = _expected_cost(eval_samples, history['DRO_x'][t], history['DRO_tau'][t], a)
            SA_eval[t] = _expected_cost(eval_samples, history['SA_x'][t], history['SA_tau'][t], a)
        DRO_s.append(np.array(history['DRO_obj_values'] >= DRO_eval).astype(float))
        SA_s.append(np.array(history['SA_obj_values'] >= SA_eval).astype(float))
        DRO_e.append(DRO_eval)
        SA_e.append(SA_eval)
    return DRO_e, DRO_s, SA_e, SA_s


def compute_cumulative_regret_dro_only(history, dateval, m, a=-5.0):
    """DRO-only out-of-sample cost and satisfaction (no SAA), over two held-out
    windows.  Returns ``(DRO_e, DRO_s)``."""
    DRO_e, DRO_s = [], []
    T = len(history['t'])
    for j in range(2):
        DRO_eval = np.zeros(T)
        eval_samples = dateval[(j * 200):(j + 1) * 200, :m]
        for t in range(T):
            DRO_eval[t] = _expected_cost(eval_samples, history['DRO_x'][t], history['DRO_tau'][t], a)
        DRO_s.append(np.array(history['DRO_obj_values'] >= DRO_eval).astype(float))
        DRO_e.append(DRO_eval)
    return DRO_e, DRO_s


# ==========================================================================
# Online / batch mean-robust (MRO) clustering machinery.
# Copied verbatim from ../port_new/utils.py (problem-agnostic: it clusters
# points in R^m); used by port_box.py / port_box_orig.py for the online-MRO and
# batch-MRO variants.  The subgradient / worst-case pieces stay box-aware via
# box_dro_subgrad_step / worst_case_value_box above.
# ==========================================================================
from scipy.spatial import distance          # noqa: E402
from scipy.spatial.distance import cdist     # noqa: E402
import ot                                     # noqa: E402
from sklearn.cluster import KMeans            # noqa: E402


def calc_rmse(dat,mean):
    rmse = 0
    for d in dat:
        rmse += np.linalg.norm(d-mean,2)**2
    return rmse


def find_min_pairwise_distance(data):
    distances = distance.cdist(data, data)
    np.fill_diagonal(distances, np.inf)  # set diagonal to infinity to ignore self-distances
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    return min_indices


def w2_dist(k1, k2, m):
    K = k2['K']
    val = 0
    for k in range(K):
        val += np.abs(k1["w"][k] - k2["w"][k])*np.linalg.norm(k1["d"][k] - k2["d"][k])
    if k1['K']>K:
        dists = cdist(k1['d'][K].reshape((1,m)),k2['d'][:K])
        val += dists@np.abs(k2['w'][:K] - k1['w'][:K])
    # val can become a size-1 array in the branch above; np.sum keeps this a
    # plain float under numpy >= 2.0 (float() on a 1-element array now raises).
    return float(np.sum(val))


def wasserstein(samples_p, samples_q):
    """
    Compute the Wasserstein-1 distance between two multi-dimensional empirical distributions.

    Parameters:
        samples_p (np.array): Samples from distribution P, shape (N, D).
        samples_q (np.array): Samples from distribution Q, shape (M, D).

    Returns:
        float: The Wasserstein-1 distance.
    """
    # Ensure the input arrays are 2D
    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)

    # Number of samples in each distribution
    N = samples_p.shape[0]
    M = samples_q.shape[0]

    # Create uniform weights for the samples
    weights_p = np.ones(N) / N  # Uniform weights for P
    weights_q = np.ones(M) / M  # Uniform weights for Q

    # Compute the cost matrix (pairwise Euclidean distances)
    cost_matrix = ot.dist(samples_p, samples_q, metric='euclidean')

    # Compute the Wasserstein-1 distance
    w_distance = ot.emd2(weights_p, weights_q, cost_matrix)

    return w_distance


def fixed_cluster(k_dict, new_dat, num_dat, m):
    new_dat = np.reshape(new_dat,(1,m))
    start_time = time.time()
    dists = cdist(new_dat,k_dict['a'])
    min_ind = np.argmin(dists)
    k_dict['d'][min_ind] = (k_dict['d'][min_ind]*k_dict['w'][min_ind]*num_dat + new_dat)/(k_dict['w'][min_ind]*num_dat + 1)
    w_k_temp = k_dict['w']*num_dat/(num_dat+1)
    increased_w = (k_dict['w'][min_ind]*num_dat + 1)/(num_dat+1)
    k_dict['w'] = w_k_temp
    k_dict['w'][min_ind] = increased_w
    total_time = time.time() - start_time
    # Membership bookkeeping. The online clusters store row *indices* into the
    # running sample array ('idx', O(1) append); the batch clusters built in the
    # drivers still store the raw rows ('data', vstack). Branch on which exists.
    if 'idx' in k_dict:
        k_dict['idx'][min_ind].append(num_dat)
    else:
        k_dict['data'][min_ind] = np.vstack([k_dict['data'][min_ind], new_dat])
    return k_dict, total_time


def calc_cluster_val_online(K,k_dict, num_dat,x,running_samples):
    square_val = 0.0
    sig_val = 0.0
    cur_K = int(np.minimum(K,num_dat))
    # Online clusters store row indices ('idx') into running_samples; batch
    # clusters store the raw rows ('data'). Either way, gather the member rows
    # per macro-cluster and vectorize the squared-distance / hinge-gap sums.
    use_idx = 'idx' in k_dict
    for k in range(cur_K):
        centroid = k_dict['d'][k]
        if use_idx:
            members = np.asarray(k_dict['idx'][k], dtype=int)
            if members.size == 0:
                continue
            pts = running_samples[members]
        else:
            pts = np.asarray(k_dict['data'][k])
            if pts.shape[0] == 0:
                continue
        diff = pts - centroid                                   # (n_k, m)
        square_val += float(np.einsum('ij,ij->i', diff, diff).sum())
        sig_val += float(np.maximum(0.0, diff @ x).sum())
    cost_matrix = ot.dist(running_samples, k_dict['d'][:cur_K], metric='euclidean')
    w_distance = ot.emd2(np.ones(num_dat)/num_dat, k_dict['w'][:cur_K], cost_matrix)
    return w_distance, square_val/num_dat, sig_val/num_dat


def cluster_k_online(K,q_dict, k_dict, init=False):
    start_time = time.time()
    cur_K = int(np.minimum(K,q_dict['cur_Q']))
    cur_Q = q_dict['cur_Q']
    k_dict['K'] = cur_K
    if cur_Q <= K:
        # Each micro-cluster is its own macro-cluster; no KMeans needed.
        for k in range(cur_K):
            k_dict[k] = np.array([k])
            k_dict['a'][k] = q_dict['a'][k]
            k_dict['d'][k] = q_dict['d'][k]
            k_dict['w'][k] = q_dict['w'][k]
        total_time = time.time() - start_time
        k_dict['idx'] = {k: list(q_dict['idx'][k]) for k in range(cur_K)}
        return k_dict, total_time
    if init:
        kmeans = KMeans(n_clusters=cur_K, init='k-means++', n_init=1).fit(q_dict['d'][:cur_Q,:])
    else:
        kmeans = KMeans(n_clusters=cur_K, init=k_dict['a'], n_init=1).fit(q_dict['d'][:cur_Q,:])
    k_dict['a'] = kmeans.cluster_centers_
    for k in range(cur_K):
        k_dict[k]= np.where(kmeans.labels_ == k)[0]
        d_cur = q_dict['d'][:cur_Q,:][kmeans.labels_ == k]
        w_cur = q_dict['w'][:cur_Q][kmeans.labels_ == k]
        k_dict['w'][k] = np.sum(w_cur)
        w_cur_norm = w_cur/(k_dict['w'][k])
        k_dict['d'][k] = np.sum(d_cur*w_cur_norm[:,np.newaxis],axis=0)
    total_time = time.time() - start_time
    k_dict['idx'] = {}
    for k in range(cur_K):
        idx_k = []
        for q in k_dict[k]:
            idx_k.extend(q_dict['idx'][int(q)])
        k_dict['idx'][k] = idx_k
    return k_dict, total_time


def assign_k_online(K, q_dict, k_dict):
    """Cheap macro-cluster update used between full KMeans re-clusters when
    ``cluster_interval > 1``.

    Keeps the existing macro centers ``k_dict['a']`` (from the last full
    re-cluster) fixed and assigns each micro-cluster centroid to its nearest
    macro center, then recomputes the macro weights / centroids / membership.
    This is a single assignment step -- O(cur_Q * K) -- instead of a full
    Lloyd's iteration, and is only invoked once there are more micro-clusters
    than macro-clusters (cur_Q > K), so ``k_dict['a']`` is already populated.
    """
    start_time = time.time()
    cur_Q = q_dict['cur_Q']
    cur_K = int(np.minimum(K, cur_Q))
    k_dict['K'] = cur_K
    micro = q_dict['d'][:cur_Q]
    labels = np.argmin(cdist(micro, k_dict['a'][:cur_K]), axis=1)
    for k in range(cur_K):
        k_dict[k] = np.where(labels == k)[0]
        w_cur = q_dict['w'][:cur_Q][k_dict[k]]
        k_dict['w'][k] = np.sum(w_cur)
        if k_dict['w'][k] > 0:
            w_cur_norm = w_cur / k_dict['w'][k]
            k_dict['d'][k] = np.sum(micro[k_dict[k]] * w_cur_norm[:, np.newaxis], axis=0)
    total_time = time.time() - start_time
    k_dict['idx'] = {}
    for k in range(cur_K):
        idx_k = []
        for q in k_dict[k]:
            idx_k.extend(q_dict['idx'][int(q)])
        k_dict['idx'][k] = idx_k
    return k_dict, total_time


def online_cluster_init_online(K, Q, data, m):
    start_time = time.time()
    k_dict = {}
    q_dict = {}
    init_num = data.shape[0]
    cur_Q =np.minimum(Q,init_num)
    q_dict['cur_Q'] = cur_Q
    qmeans = KMeans(n_clusters=q_dict['cur_Q']).fit(data)
    q_dict['a'] = np.zeros((Q+1,m))
    q_dict['d'] = np.zeros((Q+1,m))
    q_dict['w'] = np.zeros(Q+1)
    q_dict['rmse'] = np.zeros(Q+1)
    q_dict['a'][:cur_Q,:] = qmeans.cluster_centers_
    q_dict['d'][:cur_Q,:] = qmeans.cluster_centers_
    q_dict['w'][:cur_Q] = np.bincount(qmeans.labels_) / init_num
    q_dict['rmse'][:cur_Q] = np.zeros(q_dict['cur_Q'])
    # Data-adaptive floor for singleton micro-cluster radii.  The absorption
    # rule (online_cluster_update) admits a new point only when its distance to
    # the nearest micro-center is <= 2*rmse, so the floor must track the data
    # scale rather than the fixed constant 0.02.  We set it to 0.3 * median
    # nearest-neighbour distance of the init centroids so it auto-rescales.
    if cur_Q > 1:
        _dd = cdist(qmeans.cluster_centers_, qmeans.cluster_centers_)
        np.fill_diagonal(_dd, np.inf)
        rmse_floor = 0.3 * np.median(_dd.min(axis=1))
    else:
        rmse_floor = 0.02
    if not np.isfinite(rmse_floor) or rmse_floor <= 1e-6:
        rmse_floor = 0.02
    total_time = time.time() - start_time
    # Micro-cluster membership stored as row *indices* into the running sample
    # array (positions 0..init_num-1 here), not copied rows, so absorption is an
    # O(1) list append instead of an O(cluster_size) vstack.
    q_dict['idx'] = {}
    for q in range(q_dict['cur_Q']):
        members = np.where(qmeans.labels_ == q)[0]
        q_dict['idx'][q] = list(members)
        cluster_data = data[members]
        rmse = np.sqrt(calc_rmse(cluster_data,np.reshape(q_dict['d'][q],(1,m))))
        if rmse <= 1e-6:
            rmse = rmse_floor
        q_dict['rmse'][q] = rmse
    # Persistent pairwise-distance matrix between the micro-cluster anchors
    # q_dict['a'] (diagonal/inactive slots = inf).  The anchors only change on
    # spawn (one new row) and merge (two rows), so this is patched incrementally
    # instead of recomputing the full Q x Q matrix every merge.
    cq = q_dict['cur_Q']
    q_dict['D'] = np.full((Q + 1, Q + 1), np.inf)
    if cq > 1:
        sub = cdist(q_dict['a'][:cq], q_dict['a'][:cq])
        np.fill_diagonal(sub, np.inf)
        q_dict['D'][:cq, :cq] = sub
    k_dict = {}
    k_dict['a'] = np.zeros((K,m))
    k_dict['w'] = np.zeros(K)
    k_dict['d'] = np.zeros((K,m))
    k_dict['idx'] = {}
    k_dict['K'] = np.minimum(K,init_num)
    k_dict, t_time = cluster_k_online(K,q_dict, k_dict, init=True)
    return q_dict, k_dict, total_time + t_time


def online_cluster_update_online(K, new_dat, q_dict, k_dict, num_dat, t, fix_time, m, Q, rmse_mult=1.25, cluster_interval=1):
    cur_K = k_dict['K']
    new_dat = np.reshape(new_dat,(1,m))
    if t >= fix_time:
        k_dict, total_time = fixed_cluster(k_dict, new_dat, num_dat, m)
        return q_dict, k_dict, total_time
    cur_Q = q_dict['cur_Q']
    start_time = time.time()
    dists = cdist(new_dat,q_dict['d'][:cur_Q,:])
    min_dist = np.min(dists)
    min_ind = np.argmin(dists)
    if min_dist <= rmse_mult*q_dict['rmse'][min_ind] and cur_K == K:
        q_dict['d'][min_ind] = (q_dict['d'][min_ind]*q_dict['w'][min_ind]*num_dat + new_dat)/(q_dict['w'][min_ind]*num_dat + 1)
        q_dict['rmse'][min_ind] = np.sqrt((q_dict['rmse'][min_ind]**2*q_dict['w'][min_ind]*num_dat + np.linalg.norm(new_dat - q_dict['d'][min_ind],2)**2)/(q_dict['w'][min_ind]*num_dat + 1))
        w_q_temp = q_dict['w'][:cur_Q]*num_dat/(num_dat+1)
        increased_w = (q_dict['w'][min_ind]*num_dat + 1)/(num_dat+1)
        q_dict['w'][:cur_Q] = w_q_temp
        q_dict['w'][min_ind] = increased_w
        for k in range(cur_K):
            if min_ind in k_dict[k]:
                k_dict['d'][k] = (k_dict['d'][k]*k_dict['w'][k]*num_dat + new_dat)/(k_dict['w'][k]*num_dat + 1)
                k_dict['w'][k] = (k_dict['w'][k]*num_dat + 1)/(num_dat + 1)
                # absorb into this macro-cluster's membership (O(1) append)
                k_dict['idx'][k].append(num_dat)
            else:
                k_dict['w'][k] = (k_dict['w'][k]*num_dat)/(num_dat + 1)
        total_time = time.time() - start_time
        q_dict['idx'][min_ind].append(num_dat)
    else:
        start_time = time.time()
        cur_Q = q_dict['cur_Q'] + 1
        q_dict['cur_Q'] = cur_Q
        q_dict['a'][cur_Q-1] = new_dat
        q_dict['d'][cur_Q-1] = new_dat
        q_dict['rmse'][cur_Q-1] = min_dist
        q_dict['w'][:cur_Q-1] = (q_dict['w'][:cur_Q-1]*num_dat)/(num_dat+1)
        q_dict['w'][cur_Q-1] = 1/(num_dat+1)
        # patch the new anchor's distances into D (row/col cur_Q-1)
        ni = cur_Q - 1
        if ni > 0:
            row = cdist(q_dict['a'][ni:ni+1], q_dict['a'][:ni]).ravel()
            q_dict['D'][ni, :ni] = row
            q_dict['D'][:ni, ni] = row
        q_dict['D'][ni, ni] = np.inf
        total_time = time.time() - start_time
        q_dict['idx'][cur_Q-1] = [num_dat]
        if cur_Q > Q:
            start_time = time.time()
            q_dict['cur_Q'] = Q
            # closest active pair from the maintained matrix (all Q+1 slots are
            # active at this point), replacing the full O(Q^2) cdist recompute.
            min_pair = np.unravel_index(np.argmin(q_dict['D']), q_dict['D'].shape)
            merged_weight = np.sum(q_dict['w'][min_pair[0]]+q_dict['w'][min_pair[1]])
            merged_center = (q_dict['a'][min_pair[0]]*q_dict['w'][min_pair[0]] + q_dict['a'][min_pair[1]]*q_dict['w'][min_pair[1]])/merged_weight
            merged_centroid = (q_dict['d'][min_pair[0]]*q_dict['w'][min_pair[0]] + q_dict['d'][min_pair[1]]*q_dict['w'][min_pair[1]])/merged_weight
            merged_rmse = np.sqrt((q_dict['rmse'][min_pair[0]]**2*q_dict['w'][min_pair[0]] + q_dict['rmse'][min_pair[1]]**2*q_dict['w'][min_pair[1]])/merged_weight + (q_dict['w'][min_pair[0]]*np.linalg.norm( q_dict['d'][min_pair[0]]- merged_centroid)**2 + q_dict['w'][min_pair[1]]*np.linalg.norm(q_dict['d'][min_pair[1]]- merged_centroid)**2)/(merged_weight ))
            q_dict['a'][min_pair[0]] = merged_center
            q_dict['d'][min_pair[0]] = merged_centroid
            q_dict['w'][min_pair[0]] = merged_weight
            q_dict['rmse'][min_pair[0]] = merged_rmse
            q_dict['a'][min_pair[1]] = q_dict['a'][Q]
            q_dict['d'][min_pair[1]] = q_dict['d'][Q]
            q_dict['w'][min_pair[1]] = q_dict['w'][Q]
            q_dict['rmse'][min_pair[1]] = q_dict['rmse'][Q]
            # patch D: slot min_pair[0] holds the merged anchor and slot
            # min_pair[1] now holds the (moved) slot-Q anchor, so recompute those
            # two rows against the active set 0..Q-1; blank the freed slot Q.
            for s in (min_pair[0], min_pair[1]):
                drow = cdist(q_dict['a'][s:s+1], q_dict['a'][:Q]).ravel()
                q_dict['D'][s, :Q] = drow
                q_dict['D'][:Q, s] = drow
                q_dict['D'][s, s] = np.inf
            q_dict['D'][Q, :] = np.inf
            q_dict['D'][:, Q] = np.inf
            total_time += time.time() - start_time
            # merge the two micro-clusters' index lists; move slot Q into the
            # vacated slot (mirrors the centroid/weight bookkeeping above)
            q_dict['idx'][min_pair[0]] = list(q_dict['idx'][min_pair[0]]) + list(q_dict['idx'][min_pair[1]])
            q_dict['idx'][min_pair[1]] = q_dict['idx'][Q]
        # Macro re-cluster: a full KMeans every ``cluster_interval`` steps,
        # otherwise a cheap nearest-macro-center assignment of the micro-clusters
        # (keeping the last full re-cluster's centers).  cluster_interval=1 (the
        # default) reproduces the original every-spawn full re-cluster.
        if q_dict['cur_Q'] <= K or (t % cluster_interval == 0):
            k_dict, time_temp = cluster_k_online(K, q_dict, k_dict)
        else:
            k_dict, time_temp = assign_k_online(K, q_dict, k_dict)
        total_time += time_temp
    return q_dict, k_dict, total_time



def compute_cumulative_regret_online(history, dateval, m):
    """
    Compute cumulative regret by comparing online decisions against optimal DRO solution in hindsight.
    At each time t, use the same samples that were available to the online policy.

    Args:
        history (dict): History of online decisions and parameters
        dro_params (DROParameters): Problem parameters
        online_samples (np.array): Array of observed samples
        num_eval_samples (int): Number of samples to use for SAA evaluation
        seed (int): Random seed for reproducibility
    """
    def evaluate_expected_cost(d_eval, x, tau):
        return np.mean(
            np.maximum(-5*d_eval@x - 4*tau, tau))

    MRO_e = []
    MRO_s = []
    online_e = []
    online_s = []
    online_ws = []
    MRO_ws = []

    T = len(history['t'])
    # Generate evaluation samples from true distribution for cost computation
    for j in range(2):
        eval_values = np.zeros(T)
        MRO_eval_values = np.zeros(T)
        eval_samples = dateval[(j*200):(j+1)*200,:m]
    # For each timestep t
        for t in range(T):
            # Compute instantaneous regret at time t using true distribution
            online_cost = evaluate_expected_cost(eval_samples, history['x'][t],history['tau'][t])
            MRO_cost = evaluate_expected_cost(eval_samples, history['MRO_x'][t],history['MRO_tau'][t])
            eval_values[t] = online_cost
            MRO_eval_values[t] = MRO_cost

        MRO_satisfy = np.array(history['MRO_obj_values'] >= MRO_eval_values).astype(float)
        satisfy = np.array(history['obj_values'] >= eval_values).astype(float)
        worst_satisfy = np.array( np.array(history['obj_values']) + 5*np.array(history["sig_val"])>= eval_values).astype(float)
        MRO_worst_satisfy = np.array(np.array(history['MRO_obj_values']) + 5*np.array(history["sig_val_MRO"])>= MRO_eval_values).astype(float)

        MRO_e.append(MRO_eval_values)
        MRO_s.append(MRO_satisfy)
        online_e.append(eval_values)
        online_s.append(satisfy)
        online_ws.append(worst_satisfy)
        MRO_ws.append(MRO_worst_satisfy)

    return MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws


def create_scenario_cluster(dat, m, num_dat, w, a=-5.0):
    """Non-robust CVaR on the K weighted cluster centroids (cluster-SAA):
        min_{x in Delta, tau}  sum_k w_k (tau + max(0, a*tau + a*<c_k, x>)).
    ``dat`` are the centroids (K, m), ``w`` their masses (sum to 1)."""
    tau = cp.Variable()
    x = cp.Variable(m)
    objective = w @ (tau + cp.maximum(a * tau + a * (dat @ x), 0.0))
    constraints = [cp.sum(x) == 1, x >= 0, x <= 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau
