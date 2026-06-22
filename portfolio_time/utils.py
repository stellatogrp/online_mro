"""Shared utilities for the portfolio best-subset (cardinality-constrained) experiments.

Portfolio-MIP sibling of ``regression/utils.py`` / ``port_new/utils.py``.  It
reuses that family's (problem-agnostic) online-clustering and bookkeeping
machinery verbatim, swapping the problem-specific pieces for the W1-DRO CVaR
portfolio model with a cardinality (best-subset) constraint.

Problem (W1-DRO CVaR, cardinality-constrained)
----------------------------------------------
With loss ``a*tau + a*<d, x>`` (``a = -5``) the mean-robust DRO problem over the
K weighted cluster centroids ``(d_k, w_k)`` is the mixed-integer program

    min_{x, z, tau, lam, s}  tau + eps*lam + sum_k w_k s_k
    s.t.  a*tau + a*<d_k, x> <= s_k,  s_k >= 0,
          || a x ||_2 <= lam,
          sum_j x_j = 1,  0 <= x <= 1,
          x_j <= z_j,  sum_j z_j <= card,  z in {0,1}^m.

Setting eps = 0 recovers the (non-robust) sample-average / scenario problem.

Data layout
-----------
Every sample / cluster centroid is a single row of length ``m`` (asset returns);
the generic clustering helpers operate in dimension ``m``.
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


# --------------------------------------------------------------------------- #
# Generic bookkeeping / solver helpers (identical to regression/utils.py)
# --------------------------------------------------------------------------- #
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
MOSEK_TIME_LIMIT = 3000.0
MOSEK_PARAMS = {
    'MSK_DPAR_OPTIMIZER_MAX_TIME': MOSEK_TIME_LIMIT,
    'MSK_DPAR_MIO_MAX_TIME': MOSEK_TIME_LIMIT,
}


# --------------------------------------------------------------------------- #
# Portfolio problem builders
# --------------------------------------------------------------------------- #
def createproblem_portMIP(N, m, card, a=-5):
    """W1-DRO CVaR portfolio with a cardinality (best-subset) constraint.

    Parameters
    ----------
    N : int    number of (weighted) samples / cluster centroids.
    m : int    number of assets.
    card : int cardinality budget ``sum_j z_j <= card``.

    Returns ``problem, x, s, tau, lam, dat, eps, w`` where ``dat`` (N, m),
    ``eps`` (scalar Wasserstein radius) and ``w`` (N,) are the data parameters.
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)

    # VARIABLES #
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable(m, boolean=True)
    tau = cp.Variable()

    # OBJECTIVE #
    objective = tau + eps * lam + w @ s

    # CONSTRAINTS #
    constraints = []
    constraints += [a * tau + a * dat @ x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a * x, 2) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    constraints += [x - z <= 0, cp.sum(z) <= card]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w


def create_scenario(dat, m, num_dat, card, weights=None):
    """Sample-average (non-robust) cardinality-constrained CVaR portfolio.

    The eps -> 0 limit of ``createproblem_portMIP``; analogue of
    ``create_scenario_reg``.  With ``weights=None`` the objective is the uniform
    empirical CVaR; pass a weight vector (summing to 1) to fit the weighted
    centroid problem used by the clustered-SAA variant.  Returns
    ``(problem, x, tau)``.
    """
    tau = cp.Variable()
    x = cp.Variable(m)
    z = cp.Variable(m, boolean=True)
    shortfall = cp.maximum(-dat @ x - tau, 0)
    if weights is None:
        objective = cp.sum(tau + 5 * shortfall) / num_dat
    else:
        objective = tau + 5 * (weights @ shortfall)
    constraints = []
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [x - z <= 0, cp.sum(z) <= card]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau


def worst_case(N, m, dat, a=-5):
    """Dual program for the worst-case CVaR of a fixed ``(x, tau)``.

    ``x`` and ``tau`` enter as parameters so the same instance can be re-solved
    for several frozen iterates.  Returns ``problem, s, lam, x, tau, eps, w``.
    """
    # PARAMETERS #
    eps = cp.Parameter()
    w = cp.Parameter(N)
    tau = cp.Parameter()
    x = cp.Parameter(m)

    # VARIABLES #
    s = cp.Variable(N)
    lam = cp.Variable()

    # OBJECTIVE #
    objective = tau + eps * lam + w @ s

    # CONSTRAINTS #
    constraints = []
    constraints += [a * tau + a * dat @ x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a * x, 2) <= lam]
    constraints += [lam >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, s, lam, x, tau, eps, w


# --------------------------------------------------------------------------- #
# Generic clustering / distance machinery (identical to regression/utils.py,
# operating in the asset-return space of dimension m)
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


def calc_cluster_val(K, k_dict, num_dat, x, running_samples):
    """Clustering diagnostics for a fixed allocation ``x``.

    Returns ``(w_distance, square_val, sig_val)`` where
      * ``w_distance`` : Wasserstein-1 between the full samples and the centroids,
      * ``square_val`` : mean squared distance of points to their centroid,
      * ``sig_val``    : mean positive gap loss(point) - loss(centroid); the
        amount by which clustering under-counts the CVaR loss, used to inflate
        the clustered objective for the worst-case satisfaction check.
    """
    square_val = 0.0
    sig_val = 0.0
    cur_K = int(np.minimum(K, num_dat))
    for kk in range(cur_K):
        centroid = k_dict['d'][kk]
        for dat in k_dict['data'][kk]:
            square_val += np.linalg.norm(dat - centroid, 2) ** 2
            sig_val += max(0.0, (dat - centroid) @ x)
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
    # scale rather than a fixed constant.  We set it to 0.3 * median
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


def online_cluster_update_online(K, new_dat, q_dict, k_dict, num_dat, t, fix_time, m, Q):
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
    if min_dist <= 2 * q_dict['rmse'][min_ind] and cur_K == K:
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
def _evaluate_expected_cost(d_eval, x, tau):
    """Out-of-sample CVaR cost of allocation ``(x, tau)`` on ``d_eval``."""
    return np.mean(np.maximum(-5 * d_eval @ x - 4 * tau, tau))


def compute_cumulative_regret_online(history, dateval, m):
    """Out-of-sample cost and satisfaction for the online vs. batch-MRO policies.

    Also evaluates ``cluster_SAA`` -- the non-robust scenario problem solved on
    the same kmeans-weighted centroids the batch-MRO branch builds.  Returns
    ``(MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s)``.
    """
    MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = [], [], [], [], [], []
    CSA_e, CSA_s = [], []
    T = len(history['t'])
    for j in range(2):
        eval_values = np.zeros(T)
        MRO_eval_values = np.zeros(T)
        CSA_eval_values = np.zeros(T)
        eval_samples = dateval[(j * 200):(j + 1) * 200, :m]
        for t in range(T):
            eval_values[t] = _evaluate_expected_cost(eval_samples, history['x'][t], history['tau'][t])
            MRO_eval_values[t] = _evaluate_expected_cost(eval_samples, history['MRO_x'][t], history['MRO_tau'][t])
            CSA_eval_values[t] = _evaluate_expected_cost(eval_samples, history['cluster_SAA_x'][t], history['cluster_SAA_tau'][t])

        MRO_satisfy = np.array(history['MRO_obj_values'] >= MRO_eval_values).astype(float)
        satisfy = np.array(history['obj_values'] >= eval_values).astype(float)
        worst_satisfy = np.array(np.array(history['obj_values']) + 5 * np.array(history["sig_val"]) >= eval_values).astype(float)
        MRO_worst_satisfy = np.array(np.array(history['MRO_obj_values']) + 5 * np.array(history["sig_val_MRO"]) >= MRO_eval_values).astype(float)
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


def compute_cumulative_regret_dro(history, dateval, m):
    """Out-of-sample cost and satisfaction for the full-DRO vs. SAA policies."""
    DRO_e, DRO_s, SA_e, SA_s = [], [], [], []
    T = len(history['t'])
    for j in range(2):
        DRO_eval_values = np.zeros(T)
        SA_eval_values = np.zeros(T)
        eval_samples = dateval[(j * 200):(j + 1) * 200, :m]
        for t in range(T):
            DRO_eval_values[t] = _evaluate_expected_cost(eval_samples, history['DRO_x'][t], history['DRO_tau'][t])
            SA_eval_values[t] = _evaluate_expected_cost(eval_samples, history['SA_x'][t], history['SA_tau'][t])

        DRO_satisfy = np.array(history['DRO_obj_values'] >= DRO_eval_values).astype(float)
        SA_satisfy = np.array(history['SA_obj_values'] >= SA_eval_values).astype(float)

        DRO_e.append(DRO_eval_values)
        DRO_s.append(DRO_satisfy)
        SA_e.append(SA_eval_values)
        SA_s.append(SA_satisfy)
    return DRO_e, DRO_s, SA_e, SA_s
