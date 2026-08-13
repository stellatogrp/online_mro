"""Shared utilities for the p=1 hinge-loss (DRO sparse-SVM) experiments.

Provenance: verbatim copy of legacy ``svm/utils_svm.py`` with the
``svm1/utils_svm.py`` shim folded in.  The shim's only override was
``load_svm_dataset`` (default dataset ``ijcnn1``, default
``standardize=True``, data directory next to this module); those defaults
are folded into the single ``load_svm_dataset`` below.  The problem-agnostic
helpers previously imported from ``svm/utils.py`` now come from the sibling
modules ``utils.py`` and ``solver_params.py`` of this package.

This is the ``regression/utils_p1.py`` sibling adapted to a **real** LIBSVM
binary-classification dataset (see ``load_svm_dataset`` below) in place of
``generate_classification_data``'s synthetic sparse-linear model.  It reuses
that file's (problem-agnostic) online-clustering and bookkeeping machinery
verbatim and keeps the same **p = 1 hinge-loss** distributionally-robust
best-subset SVM problem.

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
The clustering helpers therefore operate in dimension ``m + 1``, while the SVM
problem builders take the covariate dimension ``m`` and split each row
internally.

Label-pure clustering
---------------------
Unlike the regression code (which clusters the joint (x, y) freely, producing
fractional centroid labels), this module enforces **label-pure** clusters: every
micro-cluster and macro-cluster contains points of a single label, so each
centroid label y_k^c is exactly +-1.  The two labels *share* the total budgets:
the number of macro-clusters never exceeds ``K`` and the number of
micro-clusters never exceeds ``Q``, with each budget split across the two labels
in proportion to their counts (``_split_budget``).  Concretely this is done by
clustering each label group separately into its allotted share and concatenating
the results; the online absorption / spawn / merge rules are restricted to
same-label clusters.  This replaces the loss-agnostic clustering stack of
``utils.py`` (the rest of which -- metadata, MOSEK options, distances, regret
bookkeeping -- is still reused unchanged).
"""
import os
import time

import numpy as np
import cvxpy as cp
import ot
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file

# Reuse the genuinely problem- and label-agnostic helpers from utils.py /
# solver_params.py (package-relative; never the legacy top-level svm/ folder).
from .solver_params import (  # noqa: F401  (re-exported for the drivers)
    MOSEK_PARAMS,
    MOSEK_TIME_LIMIT,
)
from .utils import (  # noqa: F401  (re-exported for the drivers)
    save_run_metadata,
    remove_files,
    get_n_processes,
    calc_rmse,
    project_simplex,
    w2_dist,
    wasserstein,
)


# --------------------------------------------------------------------------- #
# Hinge-loss problem builders (p = 1)
# --------------------------------------------------------------------------- #
def createproblem_hingeMIO(N, m, k, M_big=5.0, p=1):
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
    objective = w @ hinge + eps * cp.norm(beta, p)

    # CONSTRAINTS #
    constraints = [
        beta <= M_big * z,
        beta >= -M_big * z,
        cp.sum(z) <= k,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, beta, z, dat, eps, w


def create_scenario_hinge(dat, m, num_dat, k, M_big=5.0, weights=None):
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


def worst_case_hinge(dat, m, beta, delta, weights=None, p=1):
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
    return float(emp_hinge + delta * np.linalg.norm(beta, p))


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
# Finite label-transport-cost variants (kappa < infinity)
# --------------------------------------------------------------------------- #
def createproblem_hingeMIO_kappa(N, m, k, M_big=10.0, p=2):
    """DRO sparse-SVM with finite label transport cost kappa (MI-SOCP/MILP).

    Transport cost: c((x,y),(x',y')) = ||x - x'||_q + kappa * |y - y'|,
    with 1/p + 1/q = 1.

    With binary labels y in {-1,+1}, |y - y'| is 0 (same label) or 2 (flip).
    Taking the Wasserstein dual and splitting by y' = y_i vs y' = -y_i gives:

        min_{beta, z, lambda, s}  lambda * rho + w^T s
        s.t.  s_i >= 1 - y_i beta^T x_i               (same-label case)
              s_i >= 1 + y_i beta^T x_i - 2*kappa*lambda  (label-flip case)
              s_i >= 0
              ||beta||_p <= lambda,  lambda >= 0
              -M z_j <= beta_j <= M z_j,  sum z_j <= k,  z in {0,1}^m.

    p=2 (default): SOC constraint ||beta||_2 <= lambda  ->  MI-SOCP.
    p=1: LP constraint ||beta||_1 <= lambda  ->  MILP.

    When kappa -> inf the flip constraints are always slack and we recover the
    closed form: lambda = ||beta||_p, V = rho*||beta||_p + hinge.

    Parameters
    ----------
    N : int   number of (weighted) samples / cluster centroids.
    m : int   covariate dimension.  Each data row has length m+1 (x then y).
    k : int   cardinality budget ||beta||_0 <= k.
    M_big : float   big-M bound on |beta_j|.
    p : int   dual norm exponent (1 or 2).

    Returns
    -------
    problem, beta, z, lmbda, s, dat, eps, kappa_param, w
        ``dat``         : Parameter (N, m+1)
        ``eps``         : Parameter (scalar >= 0)  Wasserstein radius rho
        ``kappa_param`` : Parameter (scalar >= 0)  label transport cost kappa
        ``w``           : Parameter (N,)            sample/cluster weights
    """
    dat = cp.Parameter((N, m + 1))
    eps = cp.Parameter(nonneg=True)
    kappa_param = cp.Parameter(nonneg=True)
    w = cp.Parameter(N, nonneg=True)

    beta = cp.Variable(m)
    lmbda = cp.Variable(nonneg=True)
    s = cp.Variable(N)
    z = cp.Variable(m, boolean=True)

    X = dat[:, :m]
    y = dat[:, m]
    margins = cp.multiply(y, X @ beta)

    objective = lmbda * eps + w @ s

    constraints = [
        s >= 1 - margins,
        s >= 1 + margins - 2 * kappa_param * lmbda,
        s >= 0,
        cp.norm(beta, p) <= lmbda,
        beta <= M_big * z,
        beta >= -M_big * z,
        cp.sum(z) <= k,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, beta, z, lmbda, s, dat, eps, kappa_param, w


def worst_case_hinge_kappa(dat, m, beta, delta, kappa, weights=None, p=2):
    """Worst-case expected hinge loss for fixed beta with finite label cost kappa.

    Solves the scalar convex minimization over lambda:

        V = min_{lambda >= ||beta||_p} lambda*delta
              + sum_i w_i * max((1-m_i)_+, (1+m_i-2*kappa*lambda)_+)

    where m_i = y_i * beta^T * x_i.  The objective is piecewise linear and
    convex in lambda; scipy minimize_scalar finds the exact minimum.
    When kappa is very large the flip terms vanish and V reduces to the
    closed form delta*||beta||_p + empirical_hinge.
    """
    from scipy.optimize import minimize_scalar

    X = dat[:, :m]
    y = dat[:, m]
    n = dat.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    margins = y * (X @ beta)
    norm_beta = float(np.linalg.norm(beta, p))

    def f(lam):
        same = np.maximum(0.0, 1.0 - margins)
        flip = np.maximum(0.0, 1.0 + margins - 2.0 * kappa * lam)
        return lam * delta + float(np.dot(weights, np.maximum(same, flip)))

    # Upper bound: all flip terms vanish once lam >= (1 + max margins) / (2 kappa).
    if kappa > 0:
        lam_upper = max(norm_beta, (1.0 + float(np.max(margins))) / (2.0 * kappa)) + 1.0
    else:
        lam_upper = norm_beta + 1.0
    result = minimize_scalar(f, bounds=(norm_beta, lam_upper), method='bounded')
    return float(result.fun)


# --------------------------------------------------------------------------- #
# Order-2 Wasserstein / hinge-loss variant (W2 hinge, MI-SOCP)
# --------------------------------------------------------------------------- #
def createproblem_hingeMIO_w2(N, m, k, M_big=10.0, p=2):
    """DRO sparse-SVM with order-2 Wasserstein ball and hinge loss (W2 hinge).

    Transport cost on covariates: c(x, x') = ||x - x'||_q^2, 1/p + 1/q = 1.
    Labels are held fixed (infinite label transport cost).

    By the Wasserstein dual (see WRITEUP_w2_hinge.md), the inner supremum for
    each sample i is:

        sup_{x'} [(1 - y_i beta^T x')_+ - lambda ||x_i - x'||_q^2]
            = max(0, 1 - m_i + ||beta||_p^2 / (4 lambda)),

    where m_i = y_i beta^T x_i.  Substituting back and introducing a shared
    scalar s = ||beta||_p^2 / (4 lambda) gives the MI-SOCP:

        min_{beta, z, lambda, s, xi}  lambda * delta + w^T xi
        s.t.  xi_i >= 1 - m_i + s,  xi_i >= 0,  i = 1..N,
              ||beta||_p^2 <= 4 * s * lambda     (rotated SOC for p=2)
              lambda >= 0,  s >= 0,
              -M z_j <= beta_j <= M z_j,  j = 1..m,
              sum_j z_j <= k,  z in {0,1}^m.

    p=2 (default, Euclidean transport): the rotated SOC is a single constraint
        quad_over_lin(beta, lambda) <= 4 * s.
    p=1 (L_inf transport):  ||beta||_inf^2 <= 4*s*lambda is encoded via an
        auxiliary variable mu = ||beta||_inf with mu^2/lambda <= 4*s.

    Parameters
    ----------
    N     : int     number of (weighted) samples / cluster centroids
    m     : int     covariate dimension (rows have length m+1: x | y)
    k     : int     cardinality budget ||beta||_0 <= k
    M_big : float   big-M bound on |beta_j|
    p     : int     dual norm exponent (2 for Euclidean transport; 1 for L_inf)

    Returns
    -------
    problem, beta, z, lmbda, s_pen, dat, eps, w
        dat   : Parameter (N, m+1)       data (covariates | label)
        eps   : Parameter (scalar >= 0)  Wasserstein radius delta
        w     : Parameter (N,)           sample / cluster weights
    """
    dat = cp.Parameter((N, m + 1))
    eps = cp.Parameter(nonneg=True)
    w = cp.Parameter(N, nonneg=True)

    beta = cp.Variable(m)
    lmbda = cp.Variable(nonneg=True)
    s_pen = cp.Variable(nonneg=True)   # = ||beta||_p^2 / (4*lambda) at opt.
    xi = cp.Variable(N, nonneg=True)   # per-sample hinge epigraph
    z = cp.Variable(m, boolean=True)

    X = dat[:, :m]
    y = dat[:, m]
    margins = cp.multiply(y, X @ beta)

    objective = lmbda * eps + w @ xi

    constraints = [
        xi >= 1 - margins + s_pen,
        xi >= 0,
        lmbda >= 0,
        s_pen >= 0,
    ]

    # ||beta||_p^2 / (4*lambda) <= s_pen  <=>  ||beta||_p^2 <= 4 * s_pen * lambda
    if p == 2:
        # quad_over_lin(beta, lmbda) = ||beta||_2^2 / lmbda (DCP, RSOC internally)
        constraints.append(cp.quad_over_lin(beta, lmbda) <= 4 * s_pen)
    elif p == 1:
        # ||beta||_inf^2 <= 4 * s_pen * lambda; lift with mu = ||beta||_inf
        mu = cp.Variable(nonneg=True)
        constraints += [beta <= mu, beta >= -mu,
                        cp.quad_over_lin(mu, lmbda) <= 4 * s_pen]
    else:
        raise ValueError(f"p must be 1 or 2, got {p}")

    constraints += [
        beta <= M_big * z,
        beta >= -M_big * z,
        cp.sum(z) <= k,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, beta, z, lmbda, s_pen, dat, eps, w


def worst_case_hinge_w2(dat, m, beta, delta, weights=None, p=2):
    """Worst-case expected hinge loss for fixed beta under the W2 ball.

    Minimises over the dual variable lambda the scalar convex function

        f(lambda) = lambda * delta
                    + sum_i w_i * max(0, 1 - m_i + ||beta||_p^2 / (4*lambda))

    where m_i = y_i * beta^T * x_i.  f is convex and coercive in lambda > 0,
    so scipy.minimize_scalar finds the global minimum.

    When delta -> 0: lambda^* -> +inf and V -> empirical hinge (SAA).
    When beta = 0: V = empirical hinge (no transport benefit).
    """
    from scipy.optimize import minimize_scalar

    X = dat[:, :m]
    y = dat[:, m]
    n = dat.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    margins = y * (X @ beta)
    norm_p_sq = float(np.linalg.norm(beta, p) ** 2)

    emp_hinge = float(np.dot(weights, np.maximum(0.0, 1.0 - margins)))
    if norm_p_sq < 1e-14 or delta <= 1e-14:
        return emp_hinge

    def f(lam):
        return lam * delta + float(
            np.dot(weights, np.maximum(0.0, 1.0 - margins + norm_p_sq / (4.0 * lam))))

    # Rough optimum: d/dlam f = 0 with all hinge terms active =>
    #   lambda^* ~ sqrt(norm_p_sq / (4 * delta))
    lam_opt = np.sqrt(norm_p_sq / (4.0 * delta))
    lam_lo = max(lam_opt * 0.01, 1e-10)
    lam_hi = max(lam_opt * 100.0, 1.0)
    result = minimize_scalar(f, bounds=(lam_lo, lam_hi), method='bounded')
    return float(result.fun)


# --------------------------------------------------------------------------- #
# Gurobi problem builders / solvers (p = 1)  -- raw gurobipy, no cvxpy
# --------------------------------------------------------------------------- #
# These solve the same DRO sparse-SVM best-subset MILP as ``createproblem_hingeMIO``
#
#   min_{beta, z}  sum_i w_i (1 - y_i beta^T x_i)_+ + delta ||beta||_1
#   s.t.  -M z_j <= beta_j <= M z_j,  sum_j z_j <= k,  z in {0,1}^m,
#
# but build the model directly in gurobipy (epigraph form: hinge slacks s_i >= 0
# with s_i >= 1 - y_i beta^T x_i, and ell_1 slacks u_j >= |beta_j|).  delta = 0
# with uniform weights recovers the non-robust (SAA) best-subset SVM.
#
# gurobipy is imported lazily inside each entry point so that this module stays
# importable for the MOSEK/cvxpy drivers on machines without a Gurobi install.
GUROBI_TIME_LIMIT = 1500.0


def solve_hinge_gurobi(dat, m, k, delta, weights=None, M_big=10.0,
                       time_limit=GUROBI_TIME_LIMIT, threads=1, env=None):
    """Solve the p = 1 DRO sparse-SVM best-subset MILP directly in Gurobi.

    Parameters
    ----------
    dat : array (N, m+1)   rows are (covariates x | label y in {-1,+1}).
    m, k : int             covariate dimension and cardinality budget.
    delta : float          order-1 Wasserstein radius / ell_1 penalty coeff
                           (pass 0.0 for the non-robust SAA problem).
    weights : array (N,) or None   sample/cluster weights (sum to 1); ``None``
                           uses the uniform 1/N weighting.
    M_big : float          big-M bound on |beta_j|.
    threads : int          Gurobi thread count (default 1 -- the drivers fan the
                           epsilon/seed sweep out over processes with joblib, so
                           one thread per solve avoids CPU oversubscription).

    Returns ``(beta_opt, obj_val, solve_time)`` -- a drop-in for the
    ``(x.value, problem.objective.value, solver_stats.solve_time)`` triple the
    cvxpy/MOSEK drivers read off ``createproblem_hingeMIO``.
    """
    import gurobipy as gp
    from gurobipy import GRB

    dat = np.asarray(dat, dtype=float)
    N = dat.shape[0]
    X = dat[:, :m]
    y = dat[:, m]
    if weights is None:
        weights = np.ones(N) / N
    weights = np.asarray(weights, dtype=float)

    model = gp.Model("hinge_dro", env=env) if env is not None else gp.Model("hinge_dro")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    beta = model.addMVar(m, lb=-M_big, ub=M_big, name="beta")
    z = model.addMVar(m, vtype=GRB.BINARY, name="z")
    s = model.addMVar(N, lb=0.0, name="s")          # hinge epigraph
    u = model.addMVar(m, lb=0.0, name="u")          # |beta_j| epigraph

    A = y[:, None] * X                              # (N, m): margins = A @ beta
    model.addConstr(s + A @ beta >= np.ones(N), name="hinge")
    model.addConstr(u >= beta, name="l1p")
    model.addConstr(u >= -beta, name="l1n")
    model.addConstr(beta <= M_big * z, name="bigMp")
    model.addConstr(beta >= -M_big * z, name="bigMn")
    model.addConstr(z.sum() <= k, name="card")

    model.setObjective(weights @ s + delta * u.sum(), GRB.MINIMIZE)
    model.optimize()

    return np.asarray(beta.X, dtype=float), float(model.ObjVal), float(model.Runtime)


class HingeGurobiCG:
    """Persistent Gurobi model for the p = 1 DRO sparse-SVM, grown by constraint
    generation over the streamed samples.

    The best-subset structure on ``(beta, z, u)`` -- big-M coupling, cardinality
    budget, and the ell_1 epigraph -- is built **once**.  Each ingested data point
    then contributes a single hinge epigraph constraint

        s_i >= 1 - y_i beta^T x_i      (one new variable s_i + one new row),

    appended incrementally rather than rebuilding the whole MILP from scratch at
    every online iteration.  Because ``running_samples`` is a cumulative prefix of
    the stream, row ``i``'s hinge constraint never changes; only the objective
    coefficients move between solves (the sample weights w_i on the hinge slacks
    and the radius delta on the ell_1 slacks).  Re-solving rewrites just those
    coefficients and Gurobi warm-starts from the retained incumbent.

    Usage
    -----
        cg = HingeGurobiCG(m, k)
        for t in ...:
            cg.ensure_rows(running_samples)            # add only the new tail rows
            beta, obj, solve_time = cg.solve(weights, delta)
    """

    def __init__(self, m, k, M_big=10.0, time_limit=GUROBI_TIME_LIMIT,
                 threads=1, env=None):
        import gurobipy as gp
        from gurobipy import GRB

        self._gp = gp
        self._GRB = GRB
        self.m = m
        self.M_big = M_big

        model = gp.Model("hinge_dro_cg", env=env) if env is not None else gp.Model("hinge_dro_cg")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        model.Params.Threads = threads
        self.model = model

        # Fixed-size best-subset structure (built once).
        self.beta = model.addVars(m, lb=-M_big, ub=M_big, name="beta")
        self.z = model.addVars(m, vtype=GRB.BINARY, name="z")
        self.u = model.addVars(m, lb=0.0, name="u")        # |beta_j|
        for j in range(m):
            model.addConstr(self.u[j] >= self.beta[j])
            model.addConstr(self.u[j] >= -self.beta[j])
            model.addConstr(self.beta[j] <= M_big * self.z[j])
            model.addConstr(self.beta[j] >= -M_big * self.z[j])
        model.addConstr(gp.quicksum(self.z[j] for j in range(m)) <= k)
        model.ModelSense = GRB.MINIMIZE

        self.s = []        # hinge epigraph vars, one per ingested row
        self.N = 0         # number of rows currently in the model

    def add_samples(self, dat):
        """Append hinge constraints/variables for new rows ``dat`` (n_new, m+1)."""
        dat = np.asarray(dat, dtype=float)
        gp = self._gp
        m = self.m
        for r in range(dat.shape[0]):
            x = dat[r, :m]
            yv = float(dat[r, m])
            sv = self.model.addVar(lb=0.0, name=f"s{self.N}")
            # s_i >= 1 - y_i beta^T x_i   <=>   s_i + y_i beta^T x_i >= 1
            self.model.addConstr(
                sv + yv * gp.quicksum(x[j] * self.beta[j] for j in range(m)) >= 1.0
            )
            self.s.append(sv)
            self.N += 1

    def ensure_rows(self, running_samples):
        """Ensure the model has a hinge row for every sample in ``running_samples``
        (a cumulative prefix); add only the newly arrived tail rows."""
        running_samples = np.asarray(running_samples, dtype=float)
        n_target = running_samples.shape[0]
        if n_target > self.N:
            self.add_samples(running_samples[self.N:n_target])

    def solve(self, weights, delta):
        """Set the objective (weights w_i on the hinge slacks, delta on the ell_1
        slacks) and re-optimize the warm-started model.

        Returns ``(beta_opt, obj_val, solve_time)``.
        """
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != self.N:
            raise ValueError(
                f"weights length {weights.shape[0]} != #rows in model {self.N}; "
                "call ensure_rows(running_samples) first.")
        self.model.setAttr("Obj", self.s, [float(wi) for wi in weights])
        self.model.setAttr("Obj", [self.u[j] for j in range(self.m)],
                           [float(delta)] * self.m)
        self.model.optimize()
        beta_opt = np.array([self.beta[j].X for j in range(self.m)], dtype=float)
        return beta_opt, float(self.model.ObjVal), float(self.model.Runtime)


# --------------------------------------------------------------------------- #
# Real LIBSVM classification data loading (replaces generate_classification_data)
# --------------------------------------------------------------------------- #
def load_svm_dataset(name='ijcnn1', data_dir=None, standardize=True):
    """Load a LIBSVM binary-classification dataset from this package's ``data/``.

    Folded-in ``svm1/utils_svm.py`` shim override of the legacy loader; the
    only changes vs. ``svm/utils_svm.py`` are the defaults:
      * name defaults to 'ijcnn1'  (not 'a9a')
      * data_dir defaults to the ``data/`` directory next to this module
        (resolved relative to ``__file__``, not the CWD)
      * standardize defaults to True (ijcnn1's 22 kinematic features are
        continuous and span different physical scales, so z-scoring before
        the clustering and IHT is essential -- unlike a9a's binary
        indicators)

    Expects the two files LIBSVM ships for ``name`` -- the train split
    ``<name>`` and the test split ``<name>.t`` -- already downloaded (see
    ``data/``, sourced from
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).
    They are parsed and concatenated into one pool; each experiment draws its
    own random train/test split at run time exactly as with the synthetic
    data, so LIBSVM's own split is not load-bearing here.

    Labels are mapped to {-1, +1} (LIBSVM's {0, 1}-labelled sets would
    otherwise leave the hinge loss ill-posed). ``standardize`` optionally
    z-scores each feature column; a9a's features are already binary
    (one-hot categorical indicators) so this defaults to off.

    Returns ``(data, m)`` where ``data`` is ``(n_total, m + 1)`` with the
    label in the last column and ``m`` is the (dense) covariate dimension --
    the same layout ``generate_classification_data`` returned, so every
    downstream driver and utility (clustering, createproblem_hingeMIO,
    worst_case_hinge, ...) is unchanged.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    X_tr, y_tr = load_svmlight_file(os.path.join(data_dir, name))
    m = X_tr.shape[1]
    X_te, y_te = load_svmlight_file(os.path.join(data_dir, name + '.t'), n_features=m)
    X = np.vstack([X_tr.toarray(), X_te.toarray()])
    y = np.concatenate([y_tr, y_te])
    uniq = np.unique(y)
    if not np.array_equal(np.sort(uniq), np.array([-1.0, 1.0])):
        y = np.where(y == uniq[0], -1.0, 1.0)
    if standardize:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        X = (X - mu) / sigma
    data = np.column_stack([X, y])
    return data, m


# --------------------------------------------------------------------------- #
# Label-pure clustering machinery (shared K / Q budget across the two labels)
# --------------------------------------------------------------------------- #
def _split_budget(total, counts):
    """Split an integer budget ``total`` across groups with sizes ``counts``.

    Returns an integer allocation ``alloc`` with ``sum(alloc) = min(total,
    sum(counts))``, ``alloc[g] <= counts[g]``, and ``alloc[g] >= 1`` for every
    non-empty group whenever the budget allows.  Roughly proportional to
    ``counts`` (largest-remainder), so the two labels share the total cluster /
    micro-cluster budget in proportion to how many points each has.
    """
    counts = np.asarray(counts, dtype=int)
    G = len(counts)
    avail = int(min(int(total), int(counts.sum())))
    if avail <= 0:
        return np.zeros(G, dtype=int)
    # Largest-remainder apportionment: floor of the proportional ideal, then
    # hand out the leftover units to the largest fractional parts (capacity-aware).
    ideal = avail * counts / counts.sum()
    base = np.minimum(np.floor(ideal).astype(int), counts)
    remaining = avail - int(base.sum())
    frac = ideal - np.floor(ideal)
    pref = np.argsort(-frac)
    i = 0
    while remaining > 0 and i < 100000:
        g = pref[i % G]
        if base[g] < counts[g]:
            base[g] += 1
            remaining -= 1
        i += 1
    # Guarantee every non-empty group at least one cluster (so both labels are
    # represented) whenever the budget can cover one per non-empty group.
    active = np.where(counts > 0)[0]
    if avail >= active.size:
        for g in active:
            if base[g] == 0:
                donor = int(np.argmax(base))
                if base[donor] > 1:
                    base[donor] -= 1
                    base[g] += 1
    return base.astype(int)


def _row_labels(rows, m):
    """Integer labels (+-1) read from the last coordinate of joint (x, y) rows."""
    return np.round(np.asarray(rows)[..., m - 1]).astype(int)


def _min_pair_same_label(centers, m):
    """Closest pair of rows sharing a label (indices into ``centers``).

    Used by the micro-cluster merge step so that merging never mixes labels.
    """
    lab = _row_labels(centers, m)
    D = distance.cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    D[lab[:, None] != lab[None, :]] = np.inf
    return np.unravel_index(np.argmin(D), D.shape)


def cluster_k_online(K, q_dict, k_dict, init=False):
    """Re-cluster the micro-centers into <= K label-pure macro-clusters.

    Splits the macro budget ``cur_K = min(K, cur_Q)`` across the labels present
    among the micro-clusters (``_split_budget``) and runs k-means within each
    label group, so every macro-cluster is label-pure and the total count is
    ``cur_K``.  Macro slots are laid out label-group by label-group.
    """
    start_time = time.time()
    cur_Q = q_dict['cur_Q']
    m = q_dict['d'].shape[1]
    cur_K = int(np.minimum(K, cur_Q))

    # Clear any previous macro -> micro index assignments (integer keys only).
    for key in [kk for kk in k_dict if isinstance(kk, (int, np.integer))]:
        del k_dict[key]

    if cur_Q <= K:
        # Each micro-cluster is its own macro-cluster; no KMeans needed.
        for k in range(cur_K):
            k_dict[k] = np.array([k])
            k_dict['a'][k] = q_dict['a'][k]
            k_dict['d'][k] = q_dict['d'][k]
            k_dict['w'][k] = q_dict['w'][k]
        k_dict['K'] = cur_K
        total_time = time.time() - start_time
        k_dict['idx'] = {k: list(q_dict['idx'][k]) for k in range(cur_K)}
        return k_dict, total_time

    micro_d = q_dict['d'][:cur_Q]
    micro_lab = _row_labels(micro_d, m)
    uniq = sorted(set(micro_lab.tolist()))
    counts = [int(np.sum(micro_lab == lab)) for lab in uniq]
    alloc = _split_budget(cur_K, counts)

    k_dict['idx'] = {}
    a_list = []
    gk = 0
    for gi, lab in enumerate(uniq):
        kg = int(alloc[gi])
        if kg <= 0:
            continue
        grp = np.where(micro_lab == lab)[0]           # global micro indices
        grp_d = q_dict['d'][grp]
        if init or (grp.size <= kg):
            kmeans = KMeans(n_clusters=kg, init='k-means++', n_init=1).fit(grp_d)
        else:
            kmeans = KMeans(n_clusters=kg, init='k-means++', n_init=1).fit(grp_d)
        for j in range(kg):
            sel = grp[kmeans.labels_ == j]            # global micro indices
            k_dict[gk] = sel
            w_cur = q_dict['w'][sel]
            total_w = np.sum(w_cur)
            k_dict['w'][gk] = total_w
            if total_w > 0:
                w_norm = w_cur / total_w
            else:
                w_norm = np.ones_like(w_cur) / max(len(w_cur), 1)
            k_dict['d'][gk] = np.sum(q_dict['d'][sel] * w_norm[:, np.newaxis], axis=0)
            a_list.append(kmeans.cluster_centers_[j])
            gk += 1
    k_dict['K'] = gk
    k_dict['a'] = np.array(a_list)
    for k in range(gk):
        idx_k = []
        for q in k_dict[k]:
            idx_k.extend(q_dict['idx'][int(q)])
        k_dict['idx'][k] = idx_k
    total_time = time.time() - start_time
    return k_dict, total_time


def online_cluster_init_online(K, Q, data, m):
    """Initialize label-pure micro- (<= Q) and macro- (<= K) clusters.

    Each label group is k-means clustered into its share of the Q micro-cluster
    budget (``_split_budget``); the macro layer is then built by
    ``cluster_k_online``.  Mirrors the regression initializer but per label.
    """
    start_time = time.time()
    init_num = data.shape[0]
    lab_all = _row_labels(data, m)
    uniq = sorted(set(lab_all.tolist()))
    counts = [int(np.sum(lab_all == lab)) for lab in uniq]
    total_micro = int(np.minimum(Q, init_num))
    qalloc = _split_budget(total_micro, counts)

    q_dict = {}
    q_dict['a'] = np.zeros((Q + 1, m))
    q_dict['d'] = np.zeros((Q + 1, m))
    q_dict['w'] = np.zeros(Q + 1)
    q_dict['rmse'] = np.zeros(Q + 1)
    q_dict['idx'] = {}

    slot = 0
    centers_for_floor = []
    for gi, lab in enumerate(uniq):
        qg = int(qalloc[gi])
        if qg <= 0:
            continue
        idx = np.where(lab_all == lab)[0]
        gdata = data[idx]
        qmeans = KMeans(n_clusters=qg, n_init=1).fit(gdata)
        for j in range(qg):
            cluster_data = gdata[qmeans.labels_ == j]
            q_dict['a'][slot] = qmeans.cluster_centers_[j]
            q_dict['d'][slot] = qmeans.cluster_centers_[j]
            q_dict['w'][slot] = cluster_data.shape[0] / init_num
            q_dict['idx'][slot] = list(idx[qmeans.labels_ == j])
            centers_for_floor.append(qmeans.cluster_centers_[j])
            slot += 1
    cur_Q = slot
    q_dict['cur_Q'] = cur_Q

    # Data-adaptive singleton-radius floor (see utils.online_cluster_init_online):
    # 0.3 * median nearest-neighbour distance of the init centroids (any label).
    C = np.array(centers_for_floor)
    if C.shape[0] > 1:
        _dd = cdist(C, C)
        np.fill_diagonal(_dd, np.inf)
        rmse_floor = 0.3 * np.median(_dd.min(axis=1))
    else:
        rmse_floor = 0.02
    if not np.isfinite(rmse_floor) or rmse_floor <= 1e-6:
        rmse_floor = 0.02

    for q in range(cur_Q):
        rmse = np.sqrt(calc_rmse(data[np.asarray(q_dict['idx'][q], dtype=int)], np.reshape(q_dict['d'][q], (1, m))))
        if rmse <= 1e-6:
            rmse = rmse_floor
        q_dict['rmse'][q] = rmse
    total_time = time.time() - start_time

    cq = q_dict['cur_Q']
    q_dict['D'] = np.full((Q + 1, Q + 1), np.inf)
    if cq > 1:
        sub = cdist(q_dict['a'][:cq], q_dict['a'][:cq])
        np.fill_diagonal(sub, np.inf)
        q_dict['D'][:cq, :cq] = sub
    k_dict = {}
    k_dict['a'] = np.zeros((K, m))
    k_dict['w'] = np.zeros(K)
    k_dict['d'] = np.zeros((K, m))
    k_dict['idx'] = {}
    k_dict['K'] = int(np.minimum(K, cur_Q))
    k_dict, t_time = cluster_k_online(K, q_dict, k_dict, init=True)
    return q_dict, k_dict, total_time + t_time


def assign_k_online(K, q_dict, k_dict):
    """Cheap label-pure macro-cluster update used between full KMeans re-clusters
    when ``cluster_interval > 1``: keep the existing macro centers ``k_dict['a']``
    fixed and assign each micro-cluster centroid to its nearest *same-label* macro
    center, recomputing weights / centroids / data (one assignment step)."""
    start_time = time.time()
    cur_Q = q_dict['cur_Q']
    m = q_dict['d'].shape[1]
    cur_K = k_dict['K']
    centers = k_dict['a']
    micro_d = q_dict['d'][:cur_Q]
    micro_lab = _row_labels(micro_d, m)
    macro_lab = _row_labels(centers, m)
    D = cdist(micro_d, centers)
    D_masked = np.where(micro_lab[:, None] != macro_lab[None, :], np.inf, D)
    labels = np.argmin(D_masked, axis=1)
    no_macro = np.isinf(D_masked.min(axis=1))     # micro with no same-label macro
    if no_macro.any():
        labels[no_macro] = np.argmin(D[no_macro], axis=1)
    for key in [kk for kk in k_dict if isinstance(kk, (int, np.integer))]:
        del k_dict[key]
    k_dict['idx'] = {}
    for k in range(cur_K):
        sel = np.where(labels == k)[0]
        k_dict[k] = sel
        w_cur = q_dict['w'][sel]
        total_w = np.sum(w_cur)
        k_dict['w'][k] = total_w
        if total_w > 0:
            w_norm = w_cur / total_w
            k_dict['d'][k] = np.sum(q_dict['d'][sel] * w_norm[:, np.newaxis], axis=0)
        idx_k = []
        for q in sel:
            idx_k.extend(q_dict['idx'][int(q)])
        k_dict['idx'][k] = idx_k
    k_dict['K'] = cur_K
    total_time = time.time() - start_time
    return k_dict, total_time


def online_cluster_update_online(K, new_dat, q_dict, k_dict, num_dat, t, fix_time, m, Q, rmse_mult=2, cluster_interval=1):
    """Ingest one new (x, y) point into the label-pure online clustering.

    Absorption and spawning are restricted to the new point's own label: the
    nearest *same-label* micro-cluster is considered for absorption, a spawn
    creates a new micro-cluster of that label, and an over-budget merge combines
    the closest *same-label* micro-pair (so the shared Q budget is respected
    while every micro-cluster stays label-pure).
    """
    cur_K = k_dict['K']
    new_dat = np.reshape(new_dat, (1, m))
    lab = int(_row_labels(new_dat, m)[0])
    if t >= fix_time:
        k_dict, total_time = fixed_cluster(k_dict, new_dat, num_dat, m)
        return q_dict, k_dict, total_time
    cur_Q = q_dict['cur_Q']
    start_time = time.time()

    # nearest micro-cluster of the *same* label (for the absorption decision)
    micro_lab = _row_labels(q_dict['d'][:cur_Q], m)
    same = np.where(micro_lab == lab)[0]
    if same.size > 0:
        dsame = cdist(new_dat, q_dict['d'][same])
        jloc = int(np.argmin(dsame))
        min_dist = float(dsame[0, jloc])
        min_ind = int(same[jloc])
    else:
        min_dist = np.inf
        min_ind = -1
    # nearest micro-cluster of *any* label (only used to seed a new singleton's
    # radius, mirroring the regression code's spawn rmse)
    global_min = float(np.min(cdist(new_dat, q_dict['d'][:cur_Q])))

    if min_ind >= 0 and min_dist <= rmse_mult * q_dict['rmse'][min_ind] and cur_K == K:
        # ---- absorb into same-label micro-cluster min_ind ----
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
                k_dict['idx'][k].append(num_dat)
            else:
                k_dict['w'][k] = (k_dict['w'][k] * num_dat) / (num_dat + 1)
        total_time = time.time() - start_time
        q_dict['idx'][min_ind].append(num_dat)
    else:
        # ---- spawn a new same-label micro-cluster ----
        start_time = time.time()
        cur_Q = q_dict['cur_Q'] + 1
        q_dict['cur_Q'] = cur_Q
        q_dict['a'][cur_Q - 1] = new_dat
        q_dict['d'][cur_Q - 1] = new_dat
        q_dict['rmse'][cur_Q - 1] = global_min
        q_dict['w'][:cur_Q - 1] = (q_dict['w'][:cur_Q - 1] * num_dat) / (num_dat + 1)
        q_dict['w'][cur_Q - 1] = 1 / (num_dat + 1)
        ni = cur_Q - 1
        if ni > 0:
            row = cdist(q_dict['a'][ni:ni + 1], q_dict['a'][:ni]).ravel()
            q_dict['D'][ni, :ni] = row
            q_dict['D'][:ni, ni] = row
        q_dict['D'][ni, ni] = np.inf
        total_time = time.time() - start_time
        q_dict['idx'][cur_Q - 1] = [num_dat]
        if cur_Q > Q:
            # over budget: merge the closest *same-label* micro-pair, then
            # compact the freed slot with the last active micro-cluster.
            start_time = time.time()
            q_dict['cur_Q'] = Q
            # closest same-label micro-pair from the maintained distance matrix
            # (label-masked argmin; replaces the full O(Q^2 m) cdist recompute).
            labs = np.round(q_dict['a'][:cur_Q, m - 1]).astype(int)
            Dsub = q_dict['D'][:cur_Q, :cur_Q].copy()
            Dsub[labs[:, None] != labs[None, :]] = np.inf
            i, j = np.unravel_index(np.argmin(Dsub), Dsub.shape)
            i, j = (int(i), int(j)) if i < j else (int(j), int(i))
            merged_weight = q_dict['w'][i] + q_dict['w'][j]
            merged_center = (q_dict['a'][i] * q_dict['w'][i] + q_dict['a'][j] * q_dict['w'][j]) / merged_weight
            merged_centroid = (q_dict['d'][i] * q_dict['w'][i] + q_dict['d'][j] * q_dict['w'][j]) / merged_weight
            merged_rmse = np.sqrt((q_dict['rmse'][i] ** 2 * q_dict['w'][i] + q_dict['rmse'][j] ** 2 * q_dict['w'][j]) / merged_weight + (q_dict['w'][i] * np.linalg.norm(q_dict['d'][i] - merged_centroid) ** 2 + q_dict['w'][j] * np.linalg.norm(q_dict['d'][j] - merged_centroid) ** 2) / merged_weight)
            merged_idx = list(q_dict['idx'][i]) + list(q_dict['idx'][j])
            q_dict['a'][i] = merged_center
            q_dict['d'][i] = merged_centroid
            q_dict['w'][i] = merged_weight
            q_dict['rmse'][i] = merged_rmse
            q_dict['idx'][i] = merged_idx
            last = cur_Q - 1            # last active slot before compaction
            if j != last:
                q_dict['a'][j] = q_dict['a'][last]
                q_dict['d'][j] = q_dict['d'][last]
                q_dict['w'][j] = q_dict['w'][last]
                q_dict['rmse'][j] = q_dict['rmse'][last]
                q_dict['idx'][j] = q_dict['idx'][last]
            # patch D: rows i (merged) and j (moved 'last') over active 0..Q-1;
            # blank the freed slot Q.
            for sk in (i, j):
                drow = cdist(q_dict['a'][sk:sk + 1], q_dict['a'][:Q]).ravel()
                q_dict['D'][sk, :Q] = drow
                q_dict['D'][:Q, sk] = drow
                q_dict['D'][sk, sk] = np.inf
            q_dict['D'][Q, :] = np.inf
            q_dict['D'][:, Q] = np.inf
            total_time += time.time() - start_time
        # Macro re-cluster: full KMeans every cluster_interval steps, otherwise a
        # cheap nearest-(same-label-)center assignment of the micro-clusters
        # (cluster_interval=1, the default, reproduces every-spawn full KMeans).
        if q_dict['cur_Q'] <= K or (t % cluster_interval == 0):
            k_dict, time_temp = cluster_k_online(K, q_dict, k_dict)
        else:
            k_dict, time_temp = assign_k_online(K, q_dict, k_dict)
        total_time += time_temp
    return q_dict, k_dict, total_time


def fixed_cluster(k_dict, new_dat, num_dat, m):
    """Assign a new point to its nearest *same-label* (frozen) macro-cluster."""
    new_dat = np.reshape(new_dat, (1, m))
    lab = int(_row_labels(new_dat, m)[0])
    start_time = time.time()
    macro_lab = _row_labels(k_dict['a'], m)
    same = np.where(macro_lab == lab)[0]
    if same.size > 0:
        dists = cdist(new_dat, k_dict['a'][same])
        min_ind = int(same[int(np.argmin(dists))])
    else:
        # No macro-cluster of this label exists (should not happen once both
        # labels are present); fall back to the globally nearest centroid.
        dists = cdist(new_dat, k_dict['a'])
        min_ind = int(np.argmin(dists))
    k_dict['d'][min_ind] = (k_dict['d'][min_ind] * k_dict['w'][min_ind] * num_dat + new_dat) / (k_dict['w'][min_ind] * num_dat + 1)
    w_k_temp = k_dict['w'] * num_dat / (num_dat + 1)
    increased_w = (k_dict['w'][min_ind] * num_dat + 1) / (num_dat + 1)
    k_dict['w'] = w_k_temp
    k_dict['w'][min_ind] = increased_w
    total_time = time.time() - start_time
    if 'idx' in k_dict:
        k_dict['idx'][min_ind].append(num_dat)
    else:
        k_dict['data'][min_ind] = np.vstack([k_dict['data'][min_ind], new_dat])
    return k_dict, total_time


def label_aware_kmeans(samples, K, m):
    """Batch label-pure k-means into <= K clusters sharing the budget by label.

    Used by the batch-MRO branch of ``reg_orig_p1.py`` in place of a single
    joint k-means.  Returns ``(centers, labels, weights)`` where ``centers`` is
    ``(cur_K, m)`` (cur_K = min(K, n)), ``labels`` are global cluster ids in
    ``0..cur_K-1`` (label-group by label-group), and ``weights`` sum to 1.
    """
    n = samples.shape[0]
    lab_all = _row_labels(samples, m)
    uniq = sorted(set(lab_all.tolist()))
    counts = [int(np.sum(lab_all == lab)) for lab in uniq]
    cur_K = int(np.minimum(K, n))
    alloc = _split_budget(cur_K, counts)

    centers = []
    global_labels = np.empty(n, dtype=int)
    gk = 0
    for gi, lab in enumerate(uniq):
        kg = int(alloc[gi])
        if kg <= 0:
            continue
        idx = np.where(lab_all == lab)[0]
        kmeans = KMeans(n_clusters=kg, init='k-means++', n_init=1).fit(samples[idx])
        centers.append(kmeans.cluster_centers_)
        for j in range(kg):
            global_labels[idx[kmeans.labels_ == j]] = gk + j
        gk += kg
    centers = np.vstack(centers)
    weights = np.bincount(global_labels, minlength=centers.shape[0]) / n
    return centers, global_labels, weights


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
    use_idx = 'idx' in k_dict
    for kk in range(cur_K):
        centroid = k_dict['d'][kk]
        cx = centroid[:m]
        cy = centroid[m]
        loss_c = max(0.0, 1.0 - cy * (cx @ beta))
        if use_idx:
            members = np.asarray(k_dict['idx'][kk], dtype=int)
            if members.size == 0:
                continue
            pts = running_samples[members]
        else:
            pts = np.asarray(k_dict['data'][kk])
            if pts.shape[0] == 0:
                continue
        diff = pts - centroid
        square_val += float(np.einsum('ij,ij->i', diff, diff).sum())
        loss_i = np.maximum(0.0, 1.0 - pts[:, m] * (pts[:, :m] @ beta))
        sig_val += float(np.maximum(0.0, loss_i - loss_c).sum())
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
