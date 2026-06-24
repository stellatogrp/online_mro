"""Shared utilities for the regression (MIO perturbed-covariates) experiments.

This is the regression sibling of ``port_new/utils.py``.  It mirrors that
file's structure and reuses its (problem-agnostic) online-clustering and
bookkeeping machinery verbatim, swapping the *problem-specific* pieces for
the distributionally-robust best-subset-selection (DRO-BSS) model.

Problem (perturbed covariates, p = 2)
-------------------------------------
Empirical measure  Phat_n = (1/n) sum_i delta_(x_i, y_i).  Only covariates may
be perturbed, with order-2 Wasserstein transport cost built from the ell_q
covariate norm:

    c((x,y),(x',y')) = ||x - x'||_q^2  +  inf * 1[y != y'].

For ambiguity radius delta the DRO best-subset problem is

    min_{beta : ||beta||_0 <= k}  sup_{P in B_delta(Phat_n)}  E_P[(y - beta^T x)^2].

Blanchet-Kang-Murthy / Gao-Chen-Kleywegt give the closed-form inner sup

    sup_P sqrt(E_P[(y-beta^T x)^2]) = sqrt((1/n)||y - X beta||_2^2) + sqrt(delta) ||beta||_p,
    1/p + 1/q = 1.

With **p = 2** (hence q = 2) the dual norm is the ell_2 (ridge-type) penalty.
Substituting back, the DRO-BSS problem becomes the mixed-integer SOCP

    min_{beta, z, t, r}   (t + sqrt(delta) r)^2
    s.t.  (1/sqrt(n)) ||y - X beta||_2 <= t,   ||beta||_2 <= r,
          -M z_j <= beta_j <= M z_j,  j = 1..d,
          sum_j z_j <= k,             z in {0,1}^d.

The objective value is exactly the worst-case mean-squared error.

For the (mean-robust) clustered variants the empirical average is replaced by
the cluster-weighted average: sqrt(sum_k w_k (y_k - beta^T x_k)^2), with the
K weighted cluster centroids (x_k, y_k) playing the role of the data.

Data layout
-----------
Every sample / cluster centroid is stored as a single row of length ``m + 1``:
the first ``m`` entries are the covariates x, the last entry is the response y.
The generic clustering helpers therefore operate in dimension ``m + 1``, while
the regression problem builders take the covariate dimension ``m`` and split
each row internally.
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

    Called as the *first* thing in each experiment ``__main__`` block so the
    metadata is persisted even if the run later crashes mid-experiment.
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


def get_n_processes(max_n=np.inf):
    """Number of processes to use (SLURM-aware)."""
    try:
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()
    return max(min(max_n, n_cpus), 1)


# MOSEK solve options shared by every experiment solve: a hard 3000 s wall-clock
# cap on both the continuous and mixed-integer optimizers (returns the best
# incumbent found so far if the cap is hit).
MOSEK_TIME_LIMIT = 2000.0
MOSEK_PARAMS = {
    'MSK_DPAR_OPTIMIZER_MAX_TIME': MOSEK_TIME_LIMIT,
    'MSK_DPAR_MIO_MAX_TIME': MOSEK_TIME_LIMIT,
}


# --------------------------------------------------------------------------- #
# Regression problem builders
# --------------------------------------------------------------------------- #
def createproblem_regMIO(N, m, k, M_big=10.0):
    """DRO best-subset-selection as a mixed-integer SOCP (perturbed covariates, p=2).

    Parameters
    ----------
    N : int   number of (weighted) samples / cluster centroids.
    m : int   covariate dimension.  Each data row has length ``m + 1`` (x then y).
    k : int   cardinality budget ||beta||_0 <= k.
    M_big : float   big-M bound on |beta_j|.

    The empirical / clustered RMSE enters as a weighted ell_2 norm of the
    residuals; pass the SQUARE ROOT of the sample weights through ``sqw`` so the
    weighting sits inside the second-order cone.  For full-data DRO use
    ``sqw.value = sqrt((1/N) ones)``; for MRO use ``sqw.value = sqrt(w_k)``.

    Returns
    -------
    problem, beta, z, t, r, dat, eps, sqw
        ``dat`` : Parameter (N, m+1)        the data (covariates | response)
        ``eps`` : Parameter (scalar)        set to sqrt(delta) (the penalty coeff)
        ``sqw`` : Parameter (N,)            set to sqrt(sample/cluster weights)
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m + 1))
    eps = cp.Parameter(nonneg=True)          # = sqrt(delta)
    sqw = cp.Parameter(N, nonneg=True)       # = sqrt(weights), weights sum to 1

    # VARIABLES #
    beta = cp.Variable(m)
    z = cp.Variable(m, boolean=True)
    t = cp.Variable(nonneg=True)
    r = cp.Variable(nonneg=True)

    X = dat[:, :m]
    y = dat[:, m]
    resid = cp.multiply(sqw, y - X @ beta)

    # OBJECTIVE #  worst-case MSE = (RMSE + sqrt(delta)||beta||_2)^2
    objective = cp.square(t + eps * r)

    # CONSTRAINTS #
    constraints = [
        cp.norm(resid, 2) <= t,
        cp.norm(beta, 2) <= r,
        beta <= M_big * z,
        beta >= -M_big * z,
        cp.sum(z) <= k,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, beta, z, t, r, dat, eps, sqw


def create_scenario_reg(dat, m, num_dat, k, M_big=10.0, weights=None):
    """Sample-average (non-robust) best subset: cardinality-constrained least squares.

    The delta -> 0 limit of the DRO-BSS; analogue of ``create_scenario_dro``.
    With ``weights=None`` the objective is the uniform empirical MSE; pass a
    weight vector (summing to 1) to fit the weighted centroid least squares used
    by the clustered-SAA variant.  Returns ``(problem, beta, z)``.
    """
    X = dat[:, :m]
    y = dat[:, m]
    beta = cp.Variable(m)
    z = cp.Variable(m, boolean=True)
    resid = y - X @ beta
    if weights is None:
        objective = cp.sum_squares(resid) / num_dat
    else:
        objective = cp.sum(cp.multiply(weights, cp.square(resid)))
    constraints = [
        beta <= M_big * z,
        beta >= -M_big * z,
        cp.sum(z) <= k,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, beta, z


def worst_case_reg(dat, m, beta, sqrt_delta, weights=None):
    """Closed-form worst-case MSE of a fixed ``beta`` over the Wasserstein ball.

    worst MSE = (sqrt(sum_i w_i (y_i - beta^T x_i)^2) + sqrt(delta) ||beta||_2)^2.

    For p = 2 this replaces the portfolio ``worst_case`` dual program (the inner
    sup is available in closed form), so no solver call is needed.
    """
    X = dat[:, :m]
    y = dat[:, m]
    n = dat.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    emp_rmse = np.sqrt(np.sum(weights * (y - X @ beta) ** 2))
    return float((emp_rmse + sqrt_delta * np.linalg.norm(beta, 2)) ** 2)


def evaluate_expected_cost_reg(d_eval, m, beta):
    """Out-of-sample mean squared error of ``beta`` on ``d_eval`` (rows x|y)."""
    X = d_eval[:, :m]
    y = d_eval[:, m]
    return float(np.mean((y - X @ beta) ** 2))


# --------------------------------------------------------------------------- #
# Synthetic data generation (regression analogue of synthetic_200_1.csv)
# --------------------------------------------------------------------------- #
def generate_regression_data(n_total, m, k_true, noise_std=3.0, rho=0.5,
                             beta_scale=2.0, seed=0):
    """Generate a sparse linear-regression dataset.

    X ~ N(0, Sigma) with Toeplitz correlation Sigma_ij = rho^|i-j|; the true
    beta has ``k_true`` nonzero entries (random signs, magnitude ~beta_scale);
    y = X beta_true + N(0, noise_std^2).

    Returns ``(data, beta_true)`` where ``data`` is ``(n_total, m + 1)`` with the
    response in the last column -- the same layout the experiments consume.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(m)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n_total, m)) @ L.T
    beta_true = np.zeros(m)
    support = rng.choice(m, size=k_true, replace=False)
    beta_true[support] = beta_scale * rng.choice([-1.0, 1.0], size=k_true)
    y = X @ beta_true + noise_std * rng.standard_normal(n_total)
    data = np.column_stack([X, y])
    return data, beta_true


# --------------------------------------------------------------------------- #
# Generic clustering / distance machinery (identical to port_new/utils.py,
# operating in the joint (x, y) space of dimension m + 1)
# --------------------------------------------------------------------------- #
def find_min_pairwise_distance(data):
    distances = distance.cdist(data, data)
    np.fill_diagonal(distances, np.inf)
    return np.unravel_index(np.argmin(distances), distances.shape)


def fixed_cluster(k_dict, new_dat, num_dat, m):
    new_dat = np.reshape(new_dat, (1, m))
    start_time = time.time()
    dists = cdist(new_dat, k_dict['a'])
    min_ind = np.argmin(dists)
    k_dict['d'][min_ind] = (k_dict['d'][min_ind] * k_dict['w'][min_ind] * num_dat + new_dat) / (k_dict['w'][min_ind] * num_dat + 1)
    w_k_temp = k_dict['w'] * num_dat / (num_dat + 1)
    increased_w = (k_dict['w'][min_ind] * num_dat + 1) / (num_dat + 1)
    k_dict['w'] = w_k_temp
    k_dict['w'][min_ind] = increased_w
    total_time = time.time() - start_time
    k_dict['data'][min_ind] = np.vstack([k_dict['data'][min_ind], new_dat])
    return k_dict, total_time


def calc_rmse(dat, mean):
    rmse = 0
    for d in dat:
        rmse += np.linalg.norm(d - mean, 2) ** 2
    return rmse


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
        val += np.abs(k1["w"][k] - k2["w"][k]) * np.linalg.norm(k1["d"][k] - k2["d"][k])
    if k1['K'] > K:
        dists = cdist(k1['d'][K].reshape((1, m)), k2['d'][:K])
        val += dists @ np.abs(k2['w'][:K] - k1['w'][:K])
    return float(val)


def wasserstein(samples_p, samples_q):
    """Wasserstein-1 distance between two empirical distributions."""
    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)
    N = samples_p.shape[0]
    M = samples_q.shape[0]
    weights_p = np.ones(N) / N
    weights_q = np.ones(M) / M
    cost_matrix = ot.dist(samples_p, samples_q, metric='euclidean')
    return ot.emd2(weights_p, weights_q, cost_matrix)


def calc_cluster_val_reg(K, k_dict, num_dat, beta, running_samples, m):
    """Clustering diagnostics for a fixed ``beta``.

    Returns ``(w_distance, square_val, sig_val)`` where
      * ``w_distance`` : Wasserstein-1 between the full samples and the centroids,
      * ``square_val`` : mean squared distance of points to their centroid,
      * ``sig_val``    : mean positive gap loss(point) - loss(centroid); the
        amount by which clustering under-counts the loss, used to inflate the
        clustered objective for the worst-case satisfaction check.
    """
    square_val = 0.0
    sig_val = 0.0
    cur_K = int(np.minimum(K, num_dat))
    for kk in range(cur_K):
        centroid = k_dict['d'][kk]
        cx = centroid[:m]
        cy = centroid[m]
        loss_c = (cy - cx @ beta) ** 2
        for dat in k_dict['data'][kk]:
            square_val += np.linalg.norm(dat - centroid, 2) ** 2
            loss_i = (dat[m] - dat[:m] @ beta) ** 2
            sig_val += max(0.0, loss_i - loss_c)
    cost_matrix = ot.dist(running_samples, k_dict['d'][:cur_K], metric='euclidean')
    w_distance = ot.emd2(np.ones(num_dat) / num_dat, k_dict['w'][:cur_K], cost_matrix)
    return w_distance, square_val / num_dat, sig_val / num_dat


def cluster_k_online(K, q_dict, k_dict, init=False):
    start_time = time.time()
    cur_K = np.minimum(K, q_dict['cur_Q'])
    cur_Q = q_dict['cur_Q']
    k_dict['K'] = cur_K
    if init or (cur_Q <= K):
        kmeans = KMeans(n_clusters=cur_K, init='k-means++', n_init=1).fit(q_dict['d'][:cur_Q, :])
    else:
        kmeans = KMeans(n_clusters=cur_K, init=k_dict['a'], n_init=1).fit(q_dict['d'][:cur_Q, :])
    k_dict['a'] = kmeans.cluster_centers_
    for k in range(cur_K):
        k_dict[k] = np.where(kmeans.labels_ == k)[0]
        d_cur = q_dict['d'][:cur_Q, :][kmeans.labels_ == k]
        w_cur = q_dict['w'][:cur_Q][kmeans.labels_ == k]
        k_dict['w'][k] = np.sum(w_cur)
        w_cur_norm = w_cur / (k_dict['w'][k])
        k_dict['d'][k] = np.sum(d_cur * w_cur_norm[:, np.newaxis], axis=0)
    total_time = time.time() - start_time
    for k in range(cur_K):
        k_dict['data'][k] = np.vstack([q_dict['data'][q] for q in k_dict[k]])
    return k_dict, total_time


def online_cluster_init_online(K, Q, data, m):
    start_time = time.time()
    k_dict = {}
    q_dict = {}
    init_num = data.shape[0]
    cur_Q = np.minimum(Q, init_num)
    q_dict['cur_Q'] = cur_Q
    qmeans = KMeans(n_clusters=q_dict['cur_Q']).fit(data)
    q_dict['a'] = np.zeros((Q + 1, m))
    q_dict['d'] = np.zeros((Q + 1, m))
    q_dict['w'] = np.zeros(Q + 1)
    q_dict['rmse'] = np.zeros(Q + 1)
    q_dict['a'][:cur_Q, :] = qmeans.cluster_centers_
    q_dict['d'][:cur_Q, :] = qmeans.cluster_centers_
    q_dict['w'][:cur_Q] = np.bincount(qmeans.labels_) / init_num
    q_dict['rmse'][:cur_Q] = np.zeros(q_dict['cur_Q'])
    # Data-adaptive floor for singleton micro-cluster radii.  The absorption
    # rule (online_cluster_update) admits a new point only when its distance to
    # the nearest micro-center is <= 2*rmse, so the floor must track the data
    # scale: the old portfolio-era constant 0.02 is ~75x smaller than the
    # nearest-neighbour distance of this regression data, which would disable
    # absorption entirely.  We set it to 0.3 * median nearest-neighbour distance
    # of the init centroids (~0.9 here), so it auto-rescales with noise_std/dim.
    if cur_Q > 1:
        _dd = cdist(qmeans.cluster_centers_, qmeans.cluster_centers_)
        np.fill_diagonal(_dd, np.inf)
        rmse_floor = 0.3 * np.median(_dd.min(axis=1))
    else:
        rmse_floor = 0.02
    if not np.isfinite(rmse_floor) or rmse_floor <= 1e-6:
        rmse_floor = 0.02
    total_time = time.time() - start_time
    q_dict['data'] = {}
    for q in range(q_dict['cur_Q']):
        cluster_data = data[qmeans.labels_ == q]
        q_dict['data'][q] = cluster_data
        rmse = np.sqrt(calc_rmse(cluster_data, np.reshape(q_dict['d'][q], (1, m))))
        if rmse <= 1e-6:
            rmse = rmse_floor
        q_dict['rmse'][q] = rmse
    k_dict = {}
    k_dict['a'] = np.zeros((K, m))
    k_dict['w'] = np.zeros(K)
    k_dict['d'] = np.zeros((K, m))
    k_dict['data'] = {}
    k_dict['K'] = np.minimum(K, init_num)
    k_dict, t_time = cluster_k_online(K, q_dict, k_dict, init=True)
    return q_dict, k_dict, total_time + t_time


def online_cluster_update_online(K, new_dat, q_dict, k_dict, num_dat, t, fix_time, m, Q, rmse_mult=2):
    cur_K = k_dict['K']
    new_dat = np.reshape(new_dat, (1, m))
    if t >= fix_time:
        k_dict, total_time = fixed_cluster(k_dict, new_dat, num_dat, m)
        return q_dict, k_dict, total_time
    cur_Q = q_dict['cur_Q']
    start_time = time.time()
    dists = cdist(new_dat, q_dict['d'][:cur_Q, :])
    min_dist = np.min(dists)
    min_ind = np.argmin(dists)
    if min_dist <= rmse_mult * q_dict['rmse'][min_ind] and cur_K == K:
        q_dict['d'][min_ind] = (q_dict['d'][min_ind] * q_dict['w'][min_ind] * num_dat + new_dat) / (q_dict['w'][min_ind] * num_dat + 1)
        q_dict['rmse'][min_ind] = np.sqrt((q_dict['rmse'][min_ind] ** 2 * q_dict['w'][min_ind] * num_dat + np.linalg.norm(new_dat - q_dict['d'][min_ind], 2) ** 2) / (q_dict['w'][min_ind] * num_dat + 1))
        w_q_temp = q_dict['w'][:cur_Q] * num_dat / (num_dat + 1)
        increased_w = (q_dict['w'][min_ind] * num_dat + 1) / (num_dat + 1)
        q_dict['w'][:cur_Q] = w_q_temp
        q_dict['w'][min_ind] = increased_w
        for k in range(cur_K):
            if min_ind in k_dict[k]:
                k_dict['d'][k] = (k_dict['d'][k] * k_dict['w'][k] * num_dat + new_dat) / (k_dict['w'][k] * num_dat + 1)
                k_dict['w'][k] = (k_dict['w'][k] * num_dat + 1) / (num_dat + 1)
            else:
                k_dict['w'][k] = (k_dict['w'][k] * num_dat) / (num_dat + 1)
        total_time = time.time() - start_time
        q_dict['data'][min_ind] = np.vstack([q_dict['data'][min_ind], new_dat])
        for k in range(cur_K):
            if min_ind in k_dict[k]:
                k_dict['data'][k] = np.vstack([k_dict['data'][k], new_dat])
    else:
        start_time = time.time()
        cur_Q = q_dict['cur_Q'] + 1
        q_dict['cur_Q'] = cur_Q
        q_dict['a'][cur_Q - 1] = new_dat
        q_dict['d'][cur_Q - 1] = new_dat
        q_dict['rmse'][cur_Q - 1] = min_dist
        q_dict['w'][:cur_Q - 1] = (q_dict['w'][:cur_Q - 1] * num_dat) / (num_dat + 1)
        q_dict['w'][cur_Q - 1] = 1 / (num_dat + 1)
        total_time = time.time() - start_time
        q_dict['data'][cur_Q - 1] = new_dat
        if cur_Q > Q:
            start_time = time.time()
            q_dict['cur_Q'] = Q
            min_pair = find_min_pairwise_distance(q_dict['a'])
            merged_weight = np.sum(q_dict['w'][min_pair[0]] + q_dict['w'][min_pair[1]])
            merged_center = (q_dict['a'][min_pair[0]] * q_dict['w'][min_pair[0]] + q_dict['a'][min_pair[1]] * q_dict['w'][min_pair[1]]) / merged_weight
            merged_centroid = (q_dict['d'][min_pair[0]] * q_dict['w'][min_pair[0]] + q_dict['d'][min_pair[1]] * q_dict['w'][min_pair[1]]) / merged_weight
            merged_rmse = np.sqrt((q_dict['rmse'][min_pair[0]] ** 2 * q_dict['w'][min_pair[0]] + q_dict['rmse'][min_pair[1]] ** 2 * q_dict['w'][min_pair[1]]) / merged_weight + (q_dict['w'][min_pair[0]] * np.linalg.norm(q_dict['d'][min_pair[0]] - merged_centroid) ** 2 + q_dict['w'][min_pair[1]] * np.linalg.norm(q_dict['d'][min_pair[1]] - merged_centroid) ** 2) / (merged_weight))
            q_dict['a'][min_pair[0]] = merged_center
            q_dict['d'][min_pair[0]] = merged_centroid
            q_dict['w'][min_pair[0]] = merged_weight
            q_dict['rmse'][min_pair[0]] = merged_rmse
            q_dict['a'][min_pair[1]] = q_dict['a'][Q]
            q_dict['d'][min_pair[1]] = q_dict['d'][Q]
            q_dict['w'][min_pair[1]] = q_dict['w'][Q]
            q_dict['rmse'][min_pair[1]] = q_dict['rmse'][Q]
            total_time += time.time() - start_time
            merged_data = np.vstack([q_dict['data'][q] for q in min_pair])
            q_dict['data'][min_pair[0]] = merged_data
            q_dict['data'][min_pair[1]] = q_dict['data'][Q]
        k_dict, time_temp = cluster_k_online(K, q_dict, k_dict)
        total_time += time_temp
    return q_dict, k_dict, total_time


# --------------------------------------------------------------------------- #
# Evaluation / regret
# --------------------------------------------------------------------------- #
def compute_cumulative_regret_online_reg(history, dateval, m):
    """Out-of-sample MSE and satisfaction for the online vs. batch-MRO policies.

    Mirrors ``compute_cumulative_regret_online``: the per-sample portfolio CVaR
    cost is replaced by the squared regression loss, and ``history['x']`` /
    ``history['MRO_x']`` store the regression coefficient vectors beta.

    Also evaluates ``cluster_SAA`` -- the non-robust BSS solved on the same
    kmeans-weighted centroids the batch-MRO branch builds.  Returns
    ``(MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s)``.
    """
    MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = [], [], [], [], [], []
    CSA_e, CSA_s = [], []
    T = len(history['t'])
    for j in range(2):
        eval_values = np.zeros(T)
        MRO_eval_values = np.zeros(T)
        CSA_eval_values = np.zeros(T)
        eval_samples = dateval[(j * 200):(j + 1) * 200, :m + 1]
        for t in range(T):
            eval_values[t] = evaluate_expected_cost_reg(eval_samples, m, history['x'][t])
            MRO_eval_values[t] = evaluate_expected_cost_reg(eval_samples, m, history['MRO_x'][t])
            CSA_eval_values[t] = evaluate_expected_cost_reg(eval_samples, m, history['cluster_SAA_x'][t])

        MRO_satisfy = np.array(history['MRO_obj_values'] >= MRO_eval_values).astype(float)
        satisfy = np.array(history['obj_values'] >= eval_values).astype(float)
        worst_satisfy = np.array(np.array(history['obj_values']) + np.array(history["sig_val"]) >= eval_values).astype(float)
        MRO_worst_satisfy = np.array(np.array(history['MRO_obj_values']) + np.array(history["sig_val_MRO"]) >= MRO_eval_values).astype(float)
        CSA_satisfy = np.array(history['cluster_SAA_obj_values'] >= CSA_eval_values).astype(float)

        MRO_e.append(MRO_eval_values)
        MRO_s.append(MRO_satisfy)
        online_e.append(eval_values)
        online_s.append(satisfy)
        online_ws.append(worst_satisfy)
        MRO_ws.append(MRO_worst_satisfy)
        CSA_e.append(CSA_eval_values)
        CSA_s.append(CSA_satisfy)
    return MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s


def compute_cumulative_regret_dro_reg(history, dateval, m):
    """Out-of-sample MSE and satisfaction for the full-DRO vs. SAA policies."""
    DRO_e, DRO_s, SA_e, SA_s = [], [], [], []
    T = len(history['t'])
    for j in range(2):
        DRO_eval_values = np.zeros(T)
        SA_eval_values = np.zeros(T)
        eval_samples = dateval[(j * 200):(j + 1) * 200, :m + 1]
        for t in range(T):
            DRO_eval_values[t] = evaluate_expected_cost_reg(eval_samples, m, history['DRO_x'][t])
            SA_eval_values[t] = evaluate_expected_cost_reg(eval_samples, m, history['SA_x'][t])

        DRO_satisfy = np.array(history['DRO_obj_values'] >= DRO_eval_values).astype(float)
        SA_satisfy = np.array(history['SA_obj_values'] >= SA_eval_values).astype(float)

        DRO_e.append(DRO_eval_values)
        DRO_s.append(DRO_satisfy)
        SA_e.append(SA_eval_values)
        SA_s.append(SA_satisfy)
    return DRO_e, DRO_s, SA_e, SA_s
