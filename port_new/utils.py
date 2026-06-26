"""Shared utilities for port_new experiment scripts.

Extracted from port.py, port_p2.py, portnew.py, portnew_p2.py,
port_DRO.py, port_DRO_orig.py, port_DRO_orig_p2.py, port_DRO_p2.py.

Functions that were byte-identical across files keep their original name.
Functions that diverged between the "port" family (port*.py / portnew*.py)
and the "DRO" family (port_DRO*.py) are exported in two variants with
_online and _dro suffixes respectively. Each experiment file imports the
variant it currently uses with an `as` alias so call sites stay unchanged.
"""
import json
import os
import time

import cvxpy as cp
import joblib
import numpy as np
import ot
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def save_run_metadata(metadata, paths):
    """Write a run-metadata JSON + human-readable .txt into one or more dirs.

    Called as the *first* thing in each port_*.py ``__main__`` block so the
    metadata is persisted even if the run later crashes mid-experiment.

    Parameters
    ----------
    metadata : dict
        Run parameters / config.  Non-JSON-serializable values are coerced
        via ``default=str``.
    paths : iterable
        Each entry is either:
          * a directory path (str) -- writes ``metadata.json`` + ``metadata.txt`` there, or
          * a ``(dir, json_name)`` pair -- writes ``json_name`` (e.g.
            ``metadata_K5.json``) plus ``metadata.txt`` there.

        Directories are created with ``os.makedirs(..., exist_ok=True)``.
    """
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


def safe_solve(problem, name='?', t=None, **solve_kwargs):
    """Try ``problem.solve(**solve_kwargs)``.

    Returns True iff a usable solution exists -- no exception, and the
    solver's reported status is one of ``{'optimal', 'optimal_inaccurate'}``.
    Any caught exception OR non-optimal status prints a single warning line
    to stdout (with ``flush=True`` so it survives joblib workers) and
    returns False.

    Used by the port_new experiment scripts to tolerate individual solve
    failures without aborting the whole epsilon block.  The caller is
    responsible for falling back to a sentinel (np.nan / prior iterate)
    when the return value is False.
    """
    try:
        problem.solve(**solve_kwargs)
    except Exception as e:
        print(
            f"[safe_solve] {name} t={t} raised: {type(e).__name__}: {e}",
            flush=True,
        )
        return False
    status = getattr(problem, 'status', None)
    if status not in ('optimal', 'optimal_inaccurate'):
        print(f"[safe_solve] {name} t={t} status={status}", flush=True)
        return False
    return True


def get_n_processes(max_n=np.inf):
    """Get number of processes from current cps number
    Parameters
    ----------
    max_n: int
        Maximum number of processes.
    Returns
    -------
    float
        Number of processes to use.
    """

    try:
        # Check number of cpus if we are on a SLURM server
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc


def createproblem_portLP(N, m):
    """Continuous relaxation of createproblem_portMIP (no cardinality constraint).

    Used as a one-shot warm-start to seed (x, tau) before switching to
    worst-case + gradient-step iterates.
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5

    # VARIABLES #
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    tau = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*dat@x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*x, 2) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w


def pca(data, k):
    """PCA projection of ``data`` (shape (N, m)).

    Parameters
    ----------
    data : array-like of shape (N, m)
        Samples to fit PCA on.
    k : int or float
        * If ``int`` (>=1): keep the top-``k`` principal directions.
        * If ``float`` in ``(0, 1]``: auto-pick the smallest number of
          components whose cumulative explained-variance ratio is
          ``>= k`` (e.g. ``k=0.8`` keeps enough components to cover
          80% of the variance).

    Returns
    -------
    A : ndarray of shape (k_eff, m)
        Top-``k_eff`` right singular vectors (principal directions, as rows).
        The low-dim image of a sample xi is  ``A @ (xi - b)``.
    b : ndarray of shape (m,)
        Sample mean of the input rows.
    """
    data = np.asarray(data, dtype=float)
    b = data.mean(axis=0)
    centered = data - b
    # SVD: centered = U @ diag(S) @ Vt; rows of Vt are right singular
    # vectors (principal directions).
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    if isinstance(k, float) and 0 < k <= 1:
        var = S ** 2
        total = var.sum()
        if total == 0:
            k_eff = 1
        else:
            ratio = np.cumsum(var) / total
            # smallest index whose cumulative ratio is >= k
            k_eff = int(np.searchsorted(ratio, k) + 1)
            k_eff = max(1, min(k_eff, len(S)))
    else:
        k_eff = int(k)

    A = Vt[:k_eff]
    return A, b


def createproblem_portLP_lr(N, m,A,b):
    """Continuous relaxation of createproblem_portMIP (no cardinality constraint).

    Used as a one-shot warm-start to seed (x, tau) before switching to
    worst-case + gradient-step iterates.
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5

    # VARIABLES #
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    tau = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*((dat - b) @ A.T @ A + b) @ x<= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*A@x, 2) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w



def find_min_pairwise_distance(data):
    distances = distance.cdist(data, data)
    np.fill_diagonal(distances, np.inf)  # set diagonal to infinity to ignore self-distances
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    return min_indices

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


def createproblem_portLP_p2(N, m):
    """Continuous W_2-DRO relaxation (no cardinality constraint).

    Differences vs. the W_1 version:
      * objective gets a quadratic perspective term  ||a*x||_2^2 / (4*lam)
      * Wasserstein budget enters as  eps^2 * lam   (not  eps * lam)
      * the Lipschitz-type constraint  ||a*x||_2 <= lam  is dropped
        (it is the W_1 dual; W_2's dual is the quad-over-lin penalty above)
    """
    # PARAMETERS #
    dat  = cp.Parameter((N, m))
    eps2 = cp.Parameter(nonneg=True)        # set eps2.value = eps_val ** 2
    w    = cp.Parameter(N, nonneg=True)
    a    = -5

    # VARIABLES #
    x   = cp.Variable(m)
    s   = cp.Variable(N)
    lam = cp.Variable(nonneg=True)
    tau = cp.Variable()

    # OBJECTIVE #
    objective = (tau
                 + eps2 * lam
                 + w @ s
                 + cp.quad_over_lin(a * x, 4 * lam))

    # CONSTRAINTS #
    constraints = [
        a * tau + a * dat @ x <= s,
        s >= 0,
        cp.sum(x) == 1,
        x >= 0, x <= 1,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps2, w

def createproblem_worstcase_p2(N, m, a=-5):
    dat      = cp.Parameter((N, m))
    eps      = cp.Parameter(nonneg=True)
    w        = cp.Parameter(N, nonneg=True)
    x_star   = cp.Parameter(m)
    tau_star = cp.Parameter()

    p = cp.Variable(N, nonneg=True)
    z = cp.Variable((N, m))

    objective = (tau_star
                 + a * tau_star * cp.sum(p)
                 + a * cp.sum(z @ x_star))

    diff = z - cp.multiply(cp.reshape(p, (N, 1)), dat)   # (N, m)

    # rotated-SOC perspective: ||diff_i||^2 / p_i, summed
    wass2 = cp.sum(
        cp.hstack([cp.quad_over_lin(diff[i], p[i]) for i in range(N)])
    )

    constraints = [
        wass2 <= eps,
        p   <= w,
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    return problem, p, z, x_star, tau_star, dat, eps, w

def calc_rmse(dat,mean):
    rmse = 0
    for d in dat:
        rmse += np.linalg.norm(d-mean,2)**2
    return rmse

def gradient_step(x_curr, tau_curr, p_opt, z_opt, eta, a=-5,
                  line_search=False, inner_eval=None, F_curr=None,
                  ls_alpha=1e-4, ls_shrink=0.5, ls_max_iters=20, eta_min=1e-12):
    """One projected (sub)gradient step on (x, tau) using Danskin gradients.

    When ``line_search`` is False (default) this is a single fixed-``eta`` step
    and behavior is identical to the old signature.

    When ``line_search`` is True, an Armijo backtracking line search is run on
    the outer objective F(x, tau):

        F(x_trial, tau_trial) <= F_curr - ls_alpha * eta * ||grad||**2

    ``eta`` is shrunk by ``ls_shrink`` (default 0.5) up to ``ls_max_iters``
    times, or until ``eta < eta_min`` -- whichever comes first.  If no trial
    satisfies the condition, the last trial point is returned anyway (we never
    refuse to move on a subgradient method).

    Parameters
    ----------
    inner_eval : callable (x, tau) -> float
        Required when ``line_search=True``.  Returns the inner-max objective
        F(x, tau), typically by re-solving the worst-case dual with
        ``x_star.value``/``tau_star.value`` set to the trial point.
    F_curr : float, optional
        F(x_curr, tau_curr).  The caller has just solved the inner problem at
        the current iterate (to obtain ``p_opt``/``z_opt``), so passing in
        ``wc_problem.objective.value`` saves one re-solve.  If omitted, it is
        computed by calling ``inner_eval(x_curr, tau_curr)``.
    ls_alpha, ls_shrink, ls_max_iters, eta_min :
        Standard backtracking knobs.  Defaults are textbook Armijo.
    """
    # Danskin gradients
    grad_x   = a * z_opt.sum(axis=0)            # (m,)
    grad_tau = 1.0 + a * p_opt.sum()            # scalar

    def _step(eta_val):
        x_tilde = x_curr   - eta_val * grad_x
        tau_new = tau_curr - eta_val * grad_tau     # tau unconstrained
        x_new   = project_simplex(x_tilde)          # x onto {x>=0, sum x = 1}
        return x_new, tau_new

    if not line_search:
        return _step(eta)

    if inner_eval is None:
        raise ValueError("gradient_step: line_search=True requires inner_eval.")

    grad_sq = float(grad_x @ grad_x + grad_tau ** 2)
    if grad_sq == 0.0:
        return x_curr, tau_curr                     # zero subgradient -> no move

    if F_curr is None:
        F_curr = inner_eval(x_curr, tau_curr)

    x_new, tau_new = _step(eta)
    for _ in range(ls_max_iters):
        F_new = inner_eval(x_new, tau_new)
        if F_new <= F_curr - ls_alpha * eta * grad_sq:
            return x_new, tau_new
        eta *= ls_shrink
        if eta < eta_min:
            break
        x_new, tau_new = _step(eta)
    return x_new, tau_new

def dro_subgrad_step(x_curr, tau_curr, dat, w, eps, eta, a=-5,
                     line_search=False, ls_alpha=1e-4, ls_shrink=0.5,
                     ls_max_iters=20, eta_min=1e-12):
    """One projected-subgradient step on the W1-DRO outer objective, using the
    closed-form inner worst-case value/gradient (no CVXPY inner solve).

    Strong duality collapses the (p, z) worst-case dual to a closed form, so
    the inner max equals the primal DRO objective

        F(x, tau) = tau + sum_i w_i * max(0, a*tau + a*d_i^T x) + eps*|a|*||x||_2

    with worst-case dual optimisers  p_i = w_i if a*tau + a*d_i^T x > 0 else 0
    and the transport budget eps loaded entirely onto the ||x|| direction.
    The Danskin (sub)gradient is therefore

        grad_x   = a * (w restricted to the active set) @ dat + eps*|a|*x/||x||
        grad_tau = 1 + a * sum_{active} w_i

    Optional Armijo backtracking is evaluated against the same closed form, so
    the line search costs only cheap numpy re-evaluations rather than re-solves.

    Returns
    -------
    (x_new, tau_new, F_curr) where F_curr = F(x_curr, tau_curr).
    """
    abs_a = abs(a)

    def _F_and_grad(x, tau):
        scores = a * tau + a * (dat @ x)          # (N,)
        active = scores > 0
        xnorm = float(np.linalg.norm(x))
        F = tau + float(w @ np.where(active, scores, 0.0)) + eps * abs_a * xnorm
        gx = a * ((w * active) @ dat)             # (m,)
        if xnorm > 0:
            gx = gx + eps * abs_a * x / xnorm
        gtau = 1.0 + a * float(w @ active)
        return F, gx, gtau

    F_curr, grad_x, grad_tau = _F_and_grad(x_curr, tau_curr)

    def _step(eta_val):
        x_new = project_simplex(x_curr - eta_val * grad_x)   # x onto {x>=0, sum=1}
        tau_new = tau_curr - eta_val * grad_tau              # tau unconstrained
        return x_new, tau_new

    if not line_search:
        x_new, tau_new = _step(eta)
        return x_new, tau_new, F_curr

    grad_sq = float(grad_x @ grad_x + grad_tau ** 2)
    if grad_sq == 0.0:
        return x_curr, tau_curr, F_curr             # zero subgradient -> no move

    x_new, tau_new = _step(eta)
    for _ in range(ls_max_iters):
        F_new, _, _ = _F_and_grad(x_new, tau_new)
        if F_new <= F_curr - ls_alpha * eta * grad_sq:
            break
        eta *= ls_shrink
        if eta < eta_min:
            break
        x_new, tau_new = _step(eta)
    return x_new, tau_new, F_curr


def project_simplex(v):
    """Euclidean projection onto {x >= 0, sum x = 1} (Duchi et al. 2008)."""
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, n + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)

def w2_dist(k1, k2, m):
    K = k2['K']
    val = 0
    for k in range(K):
        val += np.abs(k1["w"][k] - k2["w"][k])*np.linalg.norm(k1["d"][k] - k2["d"][k])
    if k1['K']>K:
        dists = cdist(k1['d'][K].reshape((1,m)),k2['d'][:K])
        val += dists@np.abs(k2['w'][:K] - k1['w'][:K])
    return float(val)

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

def worst_case_value(x, tau, dat, w, eps, a=-5):
    """Closed-form optimal value of the worst_case LP (x, tau frozen).

    The LP has a trivially tight dual: s_i* = max(0, a*tau + a*dat_i@x)
    and lam* = |a|*||x||_2, giving this O(N) expression.
    """
    scores = a * tau + a * (dat @ x)
    return tau + float(w @ np.maximum(scores, 0.0)) + eps * abs(a) * np.linalg.norm(x)


def worst_case(N,m,dat):
    # PARAMETERS #
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5
    tau = cp.Parameter()
    x = cp.Parameter(m)

    # VARIABLES #
    # weights, s_i, lambda, tau
    s = cp.Variable(N)
    lam = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # + cp.quad_over_lin(a*x, 4*lam)
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*dat@x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*x, 2) <= lam]
    constraints += [lam >= 0]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, s, lam, x, tau, eps, w

def worst_case_p2(N, m, dat):
    """Evaluate V(x, tau) under W_2-DRO by solving the (frozen-x,tau) dual.

    Returns the same value as the (p, z) worst-case program, but parametrised
    over the dual variables (lam, s) so that x and tau enter as parameters.
    """
    # PARAMETERS #
    eps2 = cp.Parameter(nonneg=True)        # set eps2.value = eps_val ** 2
    w    = cp.Parameter(N, nonneg=True)
    tau  = cp.Parameter()
    x    = cp.Parameter(m)
    a    = -5

    # VARIABLES #
    s   = cp.Variable(N)
    lam = cp.Variable(nonneg=True)

    # OBJECTIVE #
    objective = (tau
                 + eps2 * lam
                 + w @ s
                 + cp.quad_over_lin(a * x, 4 * lam))

    # CONSTRAINTS #
    constraints = [
        a * tau + a * dat @ x <= s,
        s >= 0,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, s, lam, x, tau, eps2, w

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
    cur_K = np.minimum(K,q_dict['cur_Q'])
    cur_Q = q_dict['cur_Q']
    k_dict['K'] = cur_K
    if init or (cur_Q<=K):
        kmeans = KMeans(n_clusters=cur_K, init='k-means++', n_init=1).fit(q_dict['d'][:cur_Q,:])
    else:
        kmeans = KMeans(n_clusters=cur_K, init=k_dict['a'], n_init=1).fit(q_dict['d'][:cur_Q,:])
    k_dict['a'] = kmeans.cluster_centers_
    # k_dict['w'] = np.zeros(cur_K)
    # k_dict['d'] = np.zeros((cur_K,m))
    # k_dict['data'] = {}
    for k in range(cur_K):
        k_dict[k]= np.where(kmeans.labels_ == k)[0]
        d_cur = q_dict['d'][:cur_Q,:][kmeans.labels_ == k]
        w_cur = q_dict['w'][:cur_Q][kmeans.labels_ == k]
        k_dict['w'][k] = np.sum(w_cur)
        w_cur_norm = w_cur/(k_dict['w'][k])
        k_dict['d'][k] = np.sum(d_cur*w_cur_norm[:,np.newaxis],axis=0)
    total_time = time.time() - start_time
    # Macro-cluster membership as concatenated micro-cluster index lists (cheap
    # int concatenation, no row copies). Excluded from total_time as before.
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

def compute_cumulative_regret_dro(history, dateval, m):
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

    DRO_e = []
    DRO_s = []
    SA_e = []
    SA_s = []
    T = len(history['t'])
    # Generate evaluation samples from true distribution for cost computation
    for j in range(2):
        DRO_eval_values = np.zeros(T)
        SA_eval_values = np.zeros(T)
        eval_samples = dateval[(j*200):(j+1)*200,:m]
    # For each timestep t
        for t in range(T):
            # Compute instantaneous regret at time t using true distribution
            optimal_cost = evaluate_expected_cost(eval_samples, history['DRO_x'][t],history['DRO_tau'][t])
            SA_cost = evaluate_expected_cost(eval_samples, history['SA_x'][t],history['SA_tau'][t])
            DRO_eval_values[t] = optimal_cost
            SA_eval_values[t] = SA_cost

        DRO_satisfy = np.array(history['DRO_obj_values'] >= DRO_eval_values).astype(float)
        SA_satisfy = np.array(history['SA_obj_values'] >= SA_eval_values).astype(float)

        DRO_e.append(DRO_eval_values)
        DRO_s.append(DRO_satisfy)
        SA_e.append(SA_eval_values)
        SA_s.append(SA_satisfy)

    return DRO_e, DRO_s, SA_e, SA_s


def create_scenario_dro(dat,m,num_dat):
    tau = cp.Variable()
    x = cp.Variable(m)
    # z = cp.Variable(m, boolean=True)
    objective = cp.sum(tau + 5*cp.maximum(-dat@x - tau,0))/num_dat
    constraints = []
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    # constraints += [x - z <= 0, cp.sum(z) <= 8]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau


def create_scenario_cluster(dat, m, K, w):
    """SAA scenario/CVaR problem (same formulation as ``create_scenario_dro``)
    but over the ``K`` clustered points ``dat`` weighted by the cluster masses
    ``w`` (which sum to 1) instead of a uniform ``1/num_dat`` weighting."""
    tau = cp.Variable()
    x = cp.Variable(m)
    objective = w @ (tau + 5*cp.maximum(-dat@x - tau, 0))
    constraints = []
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau

def createproblem_worstcase_p1_online(N, m, a=-5):
    """Worst-case distribution problem with parameters for warm re-solving.

    Parameters
    ----------
    N : int     number of empirical samples
    m : int     dimension of each sample
    a : scalar  same coefficient as the primal (default -5)

    Returns
    -------
    problem, p, z, x_star, tau_star, dat, eps, w
        p, z          : decision variables
        x_star, tau_star : cp.Parameter, set before each solve
        dat, eps, w   : cp.Parameter, set once (or whenever data changes)
    """
    # PARAMETERS #
    dat      = cp.Parameter((N, m))
    eps      = cp.Parameter(nonneg=True)
    w        = cp.Parameter(N, nonneg=True)
    x_star   = cp.Parameter(m)
    tau_star = cp.Parameter()

    # VARIABLES #
    p = cp.Variable(N, nonneg=True)
    z = cp.Variable((N, m))

    # OBJECTIVE #
    objective = (tau_star
                 + a * tau_star * cp.sum(p)
                 + a * cp.sum(z @ x_star))

    # CONSTRAINTS #
    diff = z - cp.multiply(cp.reshape(p, (N, 1)), dat)   # (N, m)
    wass = cp.sum(cp.norm(diff, 2, axis=1))

    constraints = [
        wass <= eps,
        p   <= w,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    return problem, p, z, x_star, tau_star, dat, eps, w

def createproblem_worstcase_p1_dro(N, m, a=-5):
    """Worst-case distribution problem with parameters for warm re-solving.

    Returns
    -------
    problem, p, z, x_star, tau_star, dat, eps, w
        p, z          : decision variables
        x_star, tau_star : cp.Parameter, set before each solve
        dat, eps, w   : cp.Parameter, set once (or whenever data changes)
    """
    # PARAMETERS #
    dat      = cp.Parameter((N, m))
    eps      = cp.Parameter(nonneg=True)
    w        = cp.Parameter(N, nonneg=True)
    x_star   = cp.Parameter(m)
    tau_star = cp.Parameter()

    # VARIABLES #
    p = cp.Variable(N, nonneg=True)
    z = cp.Variable((N, m))

    # OBJECTIVE #
    objective = (tau_star
                 + a * tau_star * cp.sum(p)
                 + a * cp.sum(z @ x_star))

    # CONSTRAINTS #
    diff = z - cp.multiply(cp.reshape(p, (N, 1)), dat)   # (N, m)
    wass = cp.sum(cp.norm(diff, 2, axis=1))

    constraints = [
        wass <= eps,
        p   <= w,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    return problem, p, z, x_star, tau_star, dat, eps, w

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


