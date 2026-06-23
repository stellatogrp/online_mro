"""Shared utilities for the p=1 hinge-loss (DRO sparse-SVM) experiments.

This is the *classification* sibling of ``utils.py``.  It reuses that file's
(problem-agnostic) online-clustering and bookkeeping machinery verbatim and
swaps the *problem-specific* pieces from the p=2 squared-loss regression model
to the **p = 1 hinge-loss** distributionally-robust best-subset SVM.

Problem (perturbed covariates, p = 1, hinge loss)
-------------------------------------------------
Labelled data ``(x_i, y_i)`` with ``x_i in R^d`` and ``y_i in {-1, +1}`` and
empirical measure  Phat_n = (1/n) sum_i delta_(x_i, y_i).  Only covariates may
be perturbed, with an **order-1** Wasserstein transport cost built from the
ell_q covariate norm (labels cannot move):

    c((x,y),(x',y')) = ||x - x'||_q  +  inf * 1[y != y'].

For ambiguity radius delta the DRO best-subset SVM is

    min_{beta : ||beta||_0 <= k}  sup_{P in B_delta(Phat_n)}  E_P[ (1 - y beta^T x)_+ ].

For a Lipschitz loss and order-1 Wasserstein the inner sup has the closed form
(Shafieezadeh-Abadeh-Kuhn-Esfahani, Gao-Kleywegt, Blanchet et al.):

    sup_P E_P[(1 - y beta^T x)_+] = (1/n) sum_i (1 - y_i beta^T x_i)_+ + delta ||beta||_p,
    1/p + 1/q = 1,

because the hinge loss x |-> (1 - y beta^T x)_+ is Lipschitz w.r.t. ||.||_q with
modulus ||y beta||_{q*} = ||beta||_p (|y| = 1).  With **p = 1** (hence q = inf)
the dual norm is the sparsity-promoting ell_1 penalty.  Substituting back, the
DRO sparse-SVM becomes the mixed-integer LP

    min_{beta, z}  (1/n) sum_i (1 - y_i beta^T x_i)_+ + delta ||beta||_1
    s.t.  -M z_j <= beta_j <= M z_j,  j = 1..d,
          sum_j z_j <= k,             z in {0,1}^d.

The objective value is exactly the worst-case expected hinge loss.  Unlike the
p=2 regression objective (RMSE + sqrt(delta)||beta||_2)^2 it is *linear* in the
loss, so ``eps`` carries the radius delta directly (not sqrt(delta)) and the
problem is a MILP rather than a MI-SOCP.

For the mean-robust (clustered) variants the empirical average is replaced by
the cluster-weighted average sum_k w_k (1 - y_k beta^T x_k)_+, with the K
weighted centroids (x_k, y_k) playing the role of the data (the centroid label
y_k is the weighted average label and may be fractional).

Data layout
-----------
Every sample / cluster centroid is stored as a single row of length ``m + 1``:
the first ``m`` entries are the covariates x, the last entry is the label y.
The generic clustering helpers therefore operate in dimension ``m + 1``, while
the SVM problem builders take the covariate dimension ``m`` and split each row
internally.  Clustering is done in the joint (x, y) space exactly as in the
regression code, so the generic machinery is reused unchanged.
"""
import numpy as np
import cvxpy as cp
import ot

# Reuse the problem-agnostic machinery from the regression utilities verbatim:
# metadata / process helpers, MOSEK options, and the entire online-clustering
# stack (which operates in the joint (x, y) space and is loss-independent).
from utils import (  # noqa: F401  (re-exported for the drivers)
    save_run_metadata,
    get_n_processes,
    MOSEK_PARAMS,
    MOSEK_TIME_LIMIT,
    find_min_pairwise_distance,
    fixed_cluster,
    calc_rmse,
    project_simplex,
    w2_dist,
    wasserstein,
    cluster_k_online,
    online_cluster_init_online,
    online_cluster_update_online,
)


# --------------------------------------------------------------------------- #
# Hinge-loss problem builders (p = 1)
# --------------------------------------------------------------------------- #
def createproblem_hingeMIO(N, m, k, M_big=10.0):
    """DRO sparse-SVM best-subset selection as a mixed-integer LP (p = 1).

    Parameters
    ----------
    N : int   number of (weighted) samples / cluster centroids.
    m : int   covariate dimension.  Each data row has length ``m + 1`` (x then y).
    k : int   cardinality budget ||beta||_0 <= k.
    M_big : float   big-M bound on |beta_j|.

    The empirical / clustered hinge loss enters as a *linear* weighted sum, so
    pass the sample weights directly through ``w`` (no square root, unlike the
    p=2 SOCP builder).  For full-data DRO use ``w.value = (1/N) ones``; for the
    mean-robust variant use ``w.value = w_k``.

    Returns
    -------
    problem, beta, z, dat, eps, w
        ``dat`` : Parameter (N, m+1)   the data (covariates | label)
        ``eps`` : Parameter (scalar)   set to delta (the order-1 radius / penalty coeff)
        ``w``   : Parameter (N,)       set to sample/cluster weights (sum to 1)
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m + 1))
    eps = cp.Parameter(nonneg=True)          # = delta (order-1 radius)
    w = cp.Parameter(N, nonneg=True)         # = weights, sum to 1

    # VARIABLES #
    beta = cp.Variable(m)
    z = cp.Variable(m, boolean=True)

    X = dat[:, :m]
    y = dat[:, m]
    margins = cp.multiply(y, X @ beta)
    hinge = cp.pos(1 - margins)

    # OBJECTIVE #  worst-case expected hinge = weighted hinge + delta ||beta||_1
    objective = w @ hinge + eps * cp.norm(beta, 1)

    # CONSTRAINTS #
    constraints = [
        beta <= M_big * z,
        beta >= -M_big * z,
        cp.sum(z) <= k,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, beta, z, dat, eps, w


def create_scenario_hinge(dat, m, num_dat, k, M_big=10.0, weights=None):
    """Sample-average (non-robust) best-subset SVM: cardinality-constrained hinge.

    The delta -> 0 limit of the DRO sparse-SVM; analogue of ``create_scenario_reg``.
    With ``weights=None`` the objective is the uniform empirical hinge loss; pass
    a weight vector (summing to 1) to fit the weighted centroid hinge used by the
    clustered-SAA variant.  Returns ``(problem, beta, z)``.
    """
    X = dat[:, :m]
    y = dat[:, m]
    beta = cp.Variable(m)
    z = cp.Variable(m, boolean=True)
    hinge = cp.pos(1 - cp.multiply(y, X @ beta))
    if weights is None:
        objective = cp.sum(hinge) / num_dat
    else:
        objective = weights @ hinge
    constraints = [
        beta <= M_big * z,
        beta >= -M_big * z,
        cp.sum(z) <= k,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, beta, z


def worst_case_hinge(dat, m, beta, delta, weights=None):
    """Closed-form worst-case expected hinge loss of a fixed ``beta``.

    worst hinge = sum_i w_i (1 - y_i beta^T x_i)_+  +  delta ||beta||_1.

    For p = 1 (Lipschitz hinge, order-1 Wasserstein) the inner sup is available
    in closed form, so no solver call is needed.
    """
    X = dat[:, :m]
    y = dat[:, m]
    n = dat.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    margins = y * (X @ beta)
    emp_hinge = np.sum(weights * np.maximum(0.0, 1.0 - margins))
    return float(emp_hinge + delta * np.linalg.norm(beta, 1))


def evaluate_expected_cost_hinge(d_eval, m, beta):
    """Out-of-sample mean hinge loss of ``beta`` on ``d_eval`` (rows x|y)."""
    X = d_eval[:, :m]
    y = d_eval[:, m]
    return float(np.mean(np.maximum(0.0, 1.0 - y * (X @ beta))))


def evaluate_misclassification_hinge(d_eval, m, beta):
    """Out-of-sample 0-1 misclassification rate of ``sign(beta^T x)``.

    Not used by the satisfaction check (which compares the worst-case *hinge*
    objective against the out-of-sample *hinge* loss); provided as a convenience
    metric for downstream analysis.
    """
    X = d_eval[:, :m]
    y = d_eval[:, m]
    pred = np.sign(X @ beta)
    pred[pred == 0] = 1.0
    return float(np.mean(pred != y))


# --------------------------------------------------------------------------- #
# Synthetic classification data generation (analogue of generate_regression_data)
# --------------------------------------------------------------------------- #
def generate_classification_data(n_total, m, k_true, noise_std=3.0, rho=0.5,
                                 beta_scale=2.0, seed=0):
    """Generate a sparse linear *classification* dataset.

    X ~ N(0, Sigma) with Toeplitz correlation Sigma_ij = rho^|i-j|; the true
    beta has ``k_true`` nonzero entries (random signs, magnitude ~beta_scale);
    the label is the noisy linear rule  y = sign(beta_true^T x + eps),
    eps ~ N(0, noise_std^2), so ``noise_std`` controls the label-flip rate
    (lower noise = wider margin = easier separation).

    Because X is zero-mean and the noise is symmetric the Bayes boundary passes
    through the origin, so an intercept-free classifier x |-> sign(beta^T x) is
    appropriate (matching the intercept-free regression model).

    Returns ``(data, beta_true)`` where ``data`` is ``(n_total, m + 1)`` with the
    label in the last column -- the same layout the experiments consume.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(m)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n_total, m)) @ L.T
    beta_true = np.zeros(m)
    support = rng.choice(m, size=k_true, replace=False)
    beta_true[support] = beta_scale * rng.choice([-1.0, 1.0], size=k_true)
    logits = X @ beta_true + noise_std * rng.standard_normal(n_total)
    y = np.sign(logits)
    y[y == 0] = 1.0
    data = np.column_stack([X, y])
    return data, beta_true


# --------------------------------------------------------------------------- #
# Clustering diagnostics (hinge loss)
# --------------------------------------------------------------------------- #
def calc_cluster_val_hinge(K, k_dict, num_dat, beta, running_samples, m):
    """Clustering diagnostics for a fixed ``beta`` (hinge loss).

    Returns ``(w_distance, square_val, sig_val)`` where
      * ``w_distance`` : Wasserstein-1 between the full samples and the centroids,
      * ``square_val`` : mean squared distance of points to their centroid
        (purely geometric, identical to the regression diagnostic),
      * ``sig_val``    : mean positive gap hinge(point) - hinge(centroid); the
        amount by which clustering under-counts the hinge loss, used to inflate
        the clustered objective for the worst-case satisfaction check.
    """
    square_val = 0.0
    sig_val = 0.0
    cur_K = int(np.minimum(K, num_dat))
    for kk in range(cur_K):
        centroid = k_dict['d'][kk]
        cx = centroid[:m]
        cy = centroid[m]
        loss_c = max(0.0, 1.0 - cy * (cx @ beta))
        for dat in k_dict['data'][kk]:
            square_val += np.linalg.norm(dat - centroid, 2) ** 2
            loss_i = max(0.0, 1.0 - dat[m] * (dat[:m] @ beta))
            sig_val += max(0.0, loss_i - loss_c)
    cost_matrix = ot.dist(running_samples, k_dict['d'][:cur_K], metric='euclidean')
    w_distance = ot.emd2(np.ones(num_dat) / num_dat, k_dict['w'][:cur_K], cost_matrix)
    return w_distance, square_val / num_dat, sig_val / num_dat


# --------------------------------------------------------------------------- #
# Evaluation / regret
# --------------------------------------------------------------------------- #
def compute_cumulative_regret_online_hinge(history, dateval, m):
    """Out-of-sample hinge loss and satisfaction for the online vs. batch-MRO SVM.

    Mirrors ``compute_cumulative_regret_online_reg``: the squared regression
    loss is replaced by the hinge loss, and ``history['x']`` / ``history['MRO_x']``
    store the SVM coefficient vectors beta.

    Also evaluates ``cluster_SAA`` -- the non-robust BSS-SVM solved on the same
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
            eval_values[t] = evaluate_expected_cost_hinge(eval_samples, m, history['x'][t])
            MRO_eval_values[t] = evaluate_expected_cost_hinge(eval_samples, m, history['MRO_x'][t])
            CSA_eval_values[t] = evaluate_expected_cost_hinge(eval_samples, m, history['cluster_SAA_x'][t])

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


def compute_cumulative_regret_dro_hinge(history, dateval, m):
    """Out-of-sample hinge loss and satisfaction for the full-DRO vs. SAA SVM."""
    DRO_e, DRO_s, SA_e, SA_s = [], [], [], []
    T = len(history['t'])
    for j in range(2):
        DRO_eval_values = np.zeros(T)
        SA_eval_values = np.zeros(T)
        eval_samples = dateval[(j * 200):(j + 1) * 200, :m + 1]
        for t in range(T):
            DRO_eval_values[t] = evaluate_expected_cost_hinge(eval_samples, m, history['DRO_x'][t])
            SA_eval_values[t] = evaluate_expected_cost_hinge(eval_samples, m, history['SA_x'][t])

        DRO_satisfy = np.array(history['DRO_obj_values'] >= DRO_eval_values).astype(float)
        SA_satisfy = np.array(history['SA_obj_values'] >= SA_eval_values).astype(float)

        DRO_e.append(DRO_eval_values)
        DRO_s.append(DRO_satisfy)
        SA_e.append(SA_eval_values)
        SA_s.append(SA_satisfy)
    return DRO_e, DRO_s, SA_e, SA_s
