"""Unit tests for the subgradient-IHT path of the SPARSE-SVM experiment
(svm/methods.py: _iht, _q_step, _iht_grad_step, _milp_warmstart).

What ``_q_step`` solves (read off the code): for every ACTIVE sample
(y_i beta'x_i < 1) it replaces x_i by the maximizer of the linearized hinge
1 - y_i beta'x' over the l2-ball ||x' - x_i||_2 <= delta, whose closed form is
x_i - delta * y_i * beta / ||beta||_2; inactive samples are left unperturbed
(a heuristic worst-case map for the p=2 W1 ball, per-point radius delta).
The CVXPY cross-check below solves exactly that per-sample ball maximization.

Self-sufficient: no conftest fixtures are used.
"""
import itertools
import pathlib
import sys

import numpy as np
import pytest
import cvxpy as cp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from svm.methods import (  # noqa: E402
    _iht,
    _iht_grad_step,
    _milp_warmstart,
    _q_step,
)
from svm.utils_svm import (  # noqa: E402
    MOSEK_PARAMS,
    createproblem_hingeMIO,
    worst_case_hinge,
)

M_FEAT = 5
K_SPARSE = 2
N_PTS = 8


def tiny_instance(seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_PTS, M_FEAT))
    beta_true = np.zeros(M_FEAT)
    beta_true[[1, 3]] = [1.5, -2.0]
    y = np.sign(X @ beta_true + 0.3 * rng.standard_normal(N_PTS))
    y[y == 0] = 1.0
    return np.column_stack([X, y])


# --------------------------------------------------------------------------- #
# _q_step
# --------------------------------------------------------------------------- #
def test_q_step_matches_cvxpy_ball_maximizer():
    dat = tiny_instance()
    X, y = dat[:, :M_FEAT], dat[:, M_FEAT]
    beta = np.array([1.0, -0.5, 0.0, 0.7, 0.2])
    delta = 0.3

    dat_tilde = _q_step(dat, M_FEAT, beta, delta)
    active = y * (X @ beta) < 1
    assert active.any() and not active.all()   # both branches exercised

    for i in range(N_PTS):
        if not active[i]:
            # inactive samples are untouched
            np.testing.assert_array_equal(dat_tilde[i], dat[i])
            continue
        # independent CVXPY solve of max 1 - y_i beta'x'  s.t. ||x'-x_i|| <= delta
        xp = cp.Variable(M_FEAT)
        prob = cp.Problem(cp.Maximize(1 - y[i] * (beta @ xp)),
                          [cp.norm(xp - X[i], 2) <= delta])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(dat_tilde[i, :M_FEAT], xp.value,
                                   atol=1e-7, rtol=0)
        # perturbed hinge argument equals the maximized value
        np.testing.assert_allclose(
            1 - y[i] * (beta @ dat_tilde[i, :M_FEAT]), prob.value,
            atol=1e-7, rtol=0)

    # labels never move
    np.testing.assert_array_equal(dat_tilde[:, M_FEAT], y)
    # margins of active samples drop by exactly delta * ||beta||_2
    shift = y * (X @ beta) - dat_tilde[:, M_FEAT] * (dat_tilde[:, :M_FEAT] @ beta)
    np.testing.assert_allclose(shift[active],
                               delta * np.linalg.norm(beta, 2), atol=1e-12)


def test_q_step_zero_beta_returns_data_unchanged():
    dat = tiny_instance()
    out = _q_step(dat, M_FEAT, np.zeros(M_FEAT), 0.3)
    np.testing.assert_array_equal(out, dat)
    assert out is not dat  # a copy, not an alias


# --------------------------------------------------------------------------- #
# _iht (hard-threshold projection)
# --------------------------------------------------------------------------- #
def test_iht_output_is_k_sparse_topk_projection():
    v = np.array([0.3, -2.0, 0.05, 1.1, -0.4])
    out = _iht(v, K_SPARSE)
    assert np.sum(out != 0) <= K_SPARSE
    expected = np.zeros_like(v)
    top = np.argsort(np.abs(v))[-K_SPARSE:]
    expected[top] = v[top]
    np.testing.assert_array_equal(out, expected)
    # input not mutated
    np.testing.assert_array_equal(v, [0.3, -2.0, 0.05, 1.1, -0.4])


def test_iht_is_euclidean_projection_onto_sparsity_set():
    """H_k(v) minimizes ||v - u|| over all k-sparse u: check by enumerating
    every support (the best u on support S zeroes the complement of S)."""
    rng = np.random.default_rng(9)
    for _ in range(5):
        v = rng.standard_normal(M_FEAT)
        out = _iht(v, K_SPARSE)
        d_iht = np.linalg.norm(v - out)
        d_best = min(
            np.linalg.norm(v[[j for j in range(M_FEAT) if j not in S]])
            for S in itertools.combinations(range(M_FEAT), K_SPARSE))
        np.testing.assert_allclose(d_iht, d_best, atol=1e-12)


def test_iht_idempotent_on_k_sparse_input():
    v = np.array([0.0, -2.0, 0.0, 1.1, 0.0])
    np.testing.assert_array_equal(_iht(v, K_SPARSE), v)


def test_iht_k_geq_dim_is_identity_copy():
    v = np.array([0.3, -2.0, 0.05, 1.1, -0.4])
    out = _iht(v, M_FEAT)
    np.testing.assert_array_equal(out, v)
    assert out is not v


# --------------------------------------------------------------------------- #
# _milp_warmstart
# --------------------------------------------------------------------------- #
@pytest.mark.mosek
def test_milp_warmstart_improves_mid_iht_objective():
    dat = tiny_instance()
    delta = 0.05
    weights = np.ones(N_PTS) / N_PTS

    # a few IHT steps to get a genuine mid-run iterate
    beta = np.zeros(M_FEAT)
    for t in range(5):
        dat_tilde = _q_step(dat, M_FEAT, beta, delta)
        beta = _iht_grad_step(dat_tilde, M_FEAT, beta, 0.5 / np.sqrt(t + 1),
                              K_SPARSE)
    assert np.sum(beta != 0) <= K_SPARSE

    obj_before = worst_case_hinge(dat, M_FEAT, beta, delta,
                                  weights=weights, p=2)
    res = _milp_warmstart(dat, M_FEAT, K_SPARSE, 2, delta, weights, beta)
    assert res is not None
    beta_after, obj_after, solve_time = res

    # the MILP globally minimizes over k-sparse beta, and the warm-start hint
    # is feasible, so the objective can only improve
    assert obj_after <= obj_before + 1e-8
    assert np.sum(np.abs(beta_after) > 1e-6) <= K_SPARSE
    assert solve_time >= 0.0
    # returned objective is consistent with the closed-form worst case
    np.testing.assert_allclose(
        obj_after,
        worst_case_hinge(dat, M_FEAT, beta_after, delta, weights=weights, p=2),
        atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# IHT + Q-step loop vs. exact MILP optimum
# --------------------------------------------------------------------------- #
@pytest.mark.mosek
def test_iht_qstep_loop_decreases_and_approaches_milp_optimum():
    """Short alternating Q-step / IHT-gradient loop (the task_dro_subgrad inner
    loop, p=2). Subgradient IHT is NON-monotone, so only the overall decrease
    and the final gap to the exact MI-SOCP optimum are asserted.  Measured on
    this instance: final/opt ~ 1.44."""
    dat = tiny_instance()
    delta = 0.05
    weights = np.ones(N_PTS) / N_PTS

    beta = np.zeros(M_FEAT)
    objs = [worst_case_hinge(dat, M_FEAT, beta, delta, weights=weights, p=2)]
    for t in range(200):
        dat_tilde = _q_step(dat, M_FEAT, beta, delta)
        beta = _iht_grad_step(dat_tilde, M_FEAT, beta, 0.5 / np.sqrt(t + 1),
                              K_SPARSE, weights=weights)
        assert np.sum(beta != 0) <= K_SPARSE      # every iterate k-sparse
        objs.append(worst_case_hinge(dat, M_FEAT, beta, delta,
                                     weights=weights, p=2))

    # overall decrease from the beta=0 start (hinge = 1.0)
    np.testing.assert_allclose(objs[0], 1.0, atol=1e-12)
    assert objs[-1] < 0.5 * objs[0]

    # exact optimum from the MI-SOCP (p=2)
    prob, bvar, zvar, dpar, epar, wpar = createproblem_hingeMIO(
        N_PTS, M_FEAT, K_SPARSE, p=2)
    dpar.value = dat
    epar.value = delta
    wpar.value = weights
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
               mosek_params=MOSEK_PARAMS)
    assert prob.status == cp.OPTIMAL
    opt = prob.value

    assert objs[-1] >= opt - 1e-8       # never below the global optimum
    ratio = objs[-1] / opt
    assert ratio <= 2.0, (
        f"IHT final objective {objs[-1]:.6f} vs MILP optimum {opt:.6f} "
        f"(ratio {ratio:.3f} > 2.0)")
