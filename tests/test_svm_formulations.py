"""Unit tests for the SPARSE-SVM problem formulations (svm/utils_svm.py, svm/utils.py).

Verified against independent CVXPY models written here (never the code under
test's own builders):

  * ``createproblem_hingeMIO`` (MOSEK MILP / MI-SOCP) vs. brute-force support
    enumeration: all C(5,2)=10 supports, each restricted problem solved as a
    plain convex program with Clarabel.
  * ``worst_case_hinge`` closed form vs. (a) a CVXPY solve of the fixed-beta
    Wasserstein dual and (b) an explicit primal transport-map certificate.
  * ``create_scenario_hinge`` (delta=0 cluster-SAA) vs. the same enumeration.
  * ``wasserstein`` vs. a brute-force optimal-transport LP in CVXPY and the
    1-D sorted-matching closed form; ``w2_dist`` vs. hand-rolled arithmetic.

Self-sufficient: no conftest fixtures are used.
"""
import itertools
import pathlib
import sys

import numpy as np
import pytest
import cvxpy as cp
from scipy.spatial.distance import cdist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from svm.utils import w2_dist, wasserstein  # noqa: E402
from svm.utils_svm import (  # noqa: E402
    MOSEK_PARAMS,
    create_scenario_hinge,
    createproblem_hingeMIO,
    worst_case_hinge,
)

# Tiny deterministic instance: m=5 features, k=2 sparsity, N=8 points.
M_FEAT = 5
K_SPARSE = 2
N_PTS = 8
M_BIG = 5.0


def tiny_instance(seed=7, flip=()):
    """(N, m+1) rows (x | y) from a seeded 2-sparse linear model.

    ``flip`` lists sample indices whose label is negated, making the data
    non-separable so the delta=0 optimum is strictly positive.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_PTS, M_FEAT))
    beta_true = np.zeros(M_FEAT)
    beta_true[[1, 3]] = [1.5, -2.0]
    y = np.sign(X @ beta_true + 0.3 * rng.standard_normal(N_PTS))
    y[y == 0] = 1.0
    y[list(flip)] *= -1.0
    return np.column_stack([X, y])


def random_weights(seed=3):
    return np.random.default_rng(seed).dirichlet(np.ones(N_PTS))


def brute_force_best_subset(dat, k, delta, weights, p, M_big=M_BIG):
    """Global optimum by enumerating every size-k support and solving the
    support-restricted convex problem with an independent CVXPY model.

    Supports of size < k are dominated by their size-k supersets, so
    enumerating exactly the C(m, k) size-k supports suffices.
    """
    X, y = dat[:, :M_FEAT], dat[:, M_FEAT]
    best = np.inf
    best_beta = None
    for support in itertools.combinations(range(M_FEAT), k):
        b = cp.Variable(M_FEAT)
        cons = [b[j] == 0 for j in range(M_FEAT) if j not in support]
        cons += [b <= M_big, b >= -M_big]
        obj = weights @ cp.pos(1 - cp.multiply(y, X @ b))
        if delta > 0:
            obj = obj + delta * cp.norm(b, p)
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        if prob.value < best:
            best, best_beta = prob.value, b.value
    return best, best_beta


# --------------------------------------------------------------------------- #
# createproblem_hingeMIO vs. brute force
# --------------------------------------------------------------------------- #
@pytest.mark.mosek
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("wname", ["uniform", "random"])
def test_hinge_mio_matches_brute_force(p, wname):
    dat = tiny_instance()
    weights = np.ones(N_PTS) / N_PTS if wname == "uniform" else random_weights()
    delta = 0.1

    prob, beta, z, dpar, epar, wpar = createproblem_hingeMIO(
        N_PTS, M_FEAT, K_SPARSE, M_big=M_BIG, p=p)
    dpar.value = dat
    epar.value = delta
    wpar.value = weights
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
               mosek_params=MOSEK_PARAMS)
    assert prob.status == cp.OPTIMAL

    best, _ = brute_force_best_subset(dat, K_SPARSE, delta, weights, p)
    # Measured agreement ~5e-9 (p=1) / ~5e-9 (p=2); 1e-6 leaves headroom.
    np.testing.assert_allclose(prob.value, best, atol=1e-6, rtol=0)

    beta_opt = beta.value
    # k-sparse (big-M forces |beta_j| <= M * z_j).
    assert np.sum(np.abs(beta_opt) > 1e-6) <= K_SPARSE
    # Big-M must NOT be active at the optimum, otherwise the MILP would be a
    # relaxation of the true best-subset problem (measured max|beta| ~ 1.9).
    assert np.max(np.abs(beta_opt)) < 0.6 * M_BIG
    # MILP objective is exactly the closed-form worst case of its own beta.
    np.testing.assert_allclose(
        prob.value,
        worst_case_hinge(dat, M_FEAT, beta_opt, delta, weights=weights, p=p),
        atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# worst_case_hinge (closed form) vs. CVXPY dual and primal certificate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [1, 2])
def test_worst_case_hinge_matches_cvxpy_dual(p):
    """The closed form emp_hinge + delta*||beta||_p equals the CVXPY solve of
    the fixed-beta W1 dual:

        min_{lambda, s}  lambda*delta + w's
        s.t.  s_i >= 1 - y_i beta'x_i,  s_i >= 0,  lambda >= ||beta||_p,

    where lambda >= ||beta||_p is the finiteness condition of the inner
    sup_x' [hinge(x') - lambda||x - x'||_q] (hinge is ||beta||_p-Lipschitz
    w.r.t. ||.||_q).
    """
    dat = tiny_instance()
    X, y = dat[:, :M_FEAT], dat[:, M_FEAT]
    weights = random_weights()
    beta = np.array([0.8, -0.3, 0.0, 1.1, 0.0])
    delta = 0.15

    closed = worst_case_hinge(dat, M_FEAT, beta, delta, weights=weights, p=p)

    lam = cp.Variable(nonneg=True)
    s = cp.Variable(N_PTS)
    cons = [s >= 1 - cp.multiply(y, X @ beta), s >= 0,
            lam >= np.linalg.norm(beta, p)]
    dual = cp.Problem(cp.Minimize(lam * delta + weights @ s), cons)
    dual.solve(solver=cp.CLARABEL)
    assert dual.status == cp.OPTIMAL
    np.testing.assert_allclose(closed, dual.value, atol=1e-7, rtol=0)


@pytest.mark.parametrize("p,q", [(1, np.inf), (2, 2)])
def test_worst_case_hinge_primal_transport_certificate(p, q):
    """Attaining primal perturbation: move ONE active point's mass a q-distance
    N*delta along the steepest hinge-increase direction.  Its per-sample cost
    averages to exactly delta (W1 feasible) and, because the hinge stays active
    along the ray, the perturbed empirical hinge equals the closed form."""
    dat = tiny_instance()
    X, y = dat[:, :M_FEAT], dat[:, M_FEAT]
    beta = np.array([0.8, -0.3, 0.0, 1.1, 0.0])
    delta = 0.15

    margins = y * (X @ beta)
    i = int(np.argmin(margins))          # most active sample (margin < 1)
    assert margins[i] < 1.0

    if p == 1:
        step = np.sign(beta) * (N_PTS * delta)     # ||step||_inf = N*delta
    else:
        step = beta / np.linalg.norm(beta, 2) * (N_PTS * delta)
    X_pert = X.copy()
    X_pert[i] -= y[i] * step
    # Transport cost of the map: (1/N) * ||x_i - x'_i||_q == delta.
    np.testing.assert_allclose(
        np.linalg.norm(X_pert[i] - X[i], ord=q) / N_PTS, delta, rtol=1e-12)

    perturbed_hinge = np.mean(np.maximum(0.0, 1.0 - y * (X_pert @ beta)))
    closed = worst_case_hinge(dat, M_FEAT, beta, delta, weights=None, p=p)
    np.testing.assert_allclose(perturbed_hinge, closed, atol=1e-12, rtol=0)


def test_worst_case_hinge_zero_delta_is_empirical_hinge():
    dat = tiny_instance()
    X, y = dat[:, :M_FEAT], dat[:, M_FEAT]
    beta = np.array([0.5, 0.0, -1.0, 0.2, 0.0])
    emp = np.mean(np.maximum(0.0, 1.0 - y * (X @ beta)))
    np.testing.assert_allclose(
        worst_case_hinge(dat, M_FEAT, beta, 0.0), emp, atol=1e-14)


# --------------------------------------------------------------------------- #
# create_scenario_hinge (cluster-SAA, delta = 0) vs. brute force
# --------------------------------------------------------------------------- #
@pytest.mark.mosek
def test_scenario_hinge_matches_brute_force_uniform():
    # Two flipped labels make the instance non-separable at k=2, so the SAA
    # optimum is strictly positive (a separable instance would trivially give
    # 0 == 0 and test nothing).
    dat = tiny_instance(flip=(0, 4))
    prob, beta, z = create_scenario_hinge(dat, M_FEAT, N_PTS, K_SPARSE,
                                          M_big=M_BIG)
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
               mosek_params=MOSEK_PARAMS)
    assert prob.status == cp.OPTIMAL

    weights = np.ones(N_PTS) / N_PTS
    best, _ = brute_force_best_subset(dat, K_SPARSE, 0.0, weights, p=1)
    assert best > 0.1
    np.testing.assert_allclose(prob.value, best, atol=1e-6, rtol=0)
    assert np.sum(np.abs(beta.value) > 1e-6) <= K_SPARSE
    assert np.max(np.abs(beta.value)) < 0.6 * M_BIG


@pytest.mark.mosek
def test_scenario_hinge_matches_brute_force_weighted():
    dat = tiny_instance(flip=(0, 4))
    weights = random_weights()
    prob, beta, z = create_scenario_hinge(dat, M_FEAT, N_PTS, K_SPARSE,
                                          M_big=M_BIG, weights=weights)
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
               mosek_params=MOSEK_PARAMS)
    assert prob.status == cp.OPTIMAL
    best, _ = brute_force_best_subset(dat, K_SPARSE, 0.0, weights, p=1)
    np.testing.assert_allclose(prob.value, best, atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# wasserstein / w2_dist
# --------------------------------------------------------------------------- #
def test_wasserstein_matches_cvxpy_transport_lp():
    """ot.emd2 result vs. a brute-force optimal-transport LP in CVXPY."""
    rng = np.random.default_rng(11)
    P = rng.standard_normal((5, 3))
    Q = rng.standard_normal((4, 3)) + 0.5

    w1 = wasserstein(P, Q)

    C = cdist(P, Q, metric="euclidean")
    plan = cp.Variable((5, 4), nonneg=True)
    cons = [cp.sum(plan, axis=1) == np.ones(5) / 5,
            cp.sum(plan, axis=0) == np.ones(4) / 4]
    lp = cp.Problem(cp.Minimize(cp.sum(cp.multiply(plan, C))), cons)
    lp.solve(solver=cp.CLARABEL)
    assert lp.status == cp.OPTIMAL
    np.testing.assert_allclose(w1, lp.value, atol=1e-7, rtol=0)


def test_wasserstein_1d_sorted_matching_closed_form():
    """For two equal-size 1-D empirical distributions W1 is the mean absolute
    difference of the sorted samples (monotone rearrangement)."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal(7)
    b = rng.standard_normal(7) + 1.0
    expected = np.mean(np.abs(np.sort(a) - np.sort(b)))
    np.testing.assert_allclose(wasserstein(a, b), expected, atol=1e-10)


def test_wasserstein_identical_samples_is_zero():
    rng = np.random.default_rng(2)
    P = rng.standard_normal((6, 2))
    np.testing.assert_allclose(wasserstein(P, P.copy()), 0.0, atol=1e-10)


def test_w2_dist_equal_K_hand_computed():
    m = 3
    rng = np.random.default_rng(21)
    d1 = rng.standard_normal((3, m))
    d2 = rng.standard_normal((3, m))
    w1 = np.array([0.5, 0.3, 0.2])
    w2 = np.array([0.4, 0.4, 0.2])
    k1 = {"K": 3, "w": w1, "d": d1}
    k2 = {"K": 3, "w": w2, "d": d2}

    expected = 0.0
    for k in range(3):
        expected += abs(w1[k] - w2[k]) * float(np.sqrt(np.sum((d1[k] - d2[k]) ** 2)))
    np.testing.assert_allclose(w2_dist(k1, k2, m), expected, atol=1e-12)
    # Identical dicts -> zero.
    np.testing.assert_allclose(w2_dist(k1, {"K": 3, "w": w1, "d": d1}, m),
                               0.0, atol=1e-14)


def test_w2_dist_extra_cluster_hand_computed():
    """k1 has one more cluster than k2: the extra term couples centroid K of k1
    with all K centroids of k2 through the weight differences."""
    m = 3
    rng = np.random.default_rng(22)
    d1 = rng.standard_normal((4, m))
    d2 = rng.standard_normal((3, m))
    w1 = np.array([0.4, 0.3, 0.2, 0.1])
    w2 = np.array([0.5, 0.3, 0.2])
    k1 = {"K": 4, "w": w1, "d": d1}
    k2 = {"K": 3, "w": w2, "d": d2}

    expected = 0.0
    for k in range(3):
        expected += abs(w1[k] - w2[k]) * float(np.sqrt(np.sum((d1[k] - d2[k]) ** 2)))
    for k in range(3):
        dist_k = float(np.sqrt(np.sum((d1[3] - d2[k]) ** 2)))
        expected += dist_k * abs(w2[k] - w1[k])
    np.testing.assert_allclose(w2_dist(k1, k2, m), expected, atol=1e-12)
