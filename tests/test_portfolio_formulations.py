"""Unit tests for ``portfolio.utils.createproblem_box_DRO``.

The SOCP built by ``createproblem_box_DRO`` is checked against an
INDEPENDENT CVXPY formulation written from scratch below, re-derived from
the Esfahani-Kuhn strong-duality reformulation for polyhedral support
(port_box/METHOD.md, Sections 1.1-1.2 for the problem and Section 2.1
"The Wasserstein dual (Section 9.1)" for the dual; the finite reformulation
is Mohajerin Esfahani & Kuhn 2018, Thm 4.2, specialized to the box written
as C xi <= d with C = [I; -I], d = [ub; -lb]).

Also checks consistency of the SOCP optimum with the independent nested
bisection oracle ``worst_case_value_box`` evaluated at the SOCP minimizer.
"""
import cvxpy as cp
import numpy as np
import pytest

from portfolio.utils import (
    createproblem_box_DRO,
    generate_returns,
    worst_case_value_box,
)

A_CONST = -5.0


# --------------------------------------------------------------------------- #
# Independent from-scratch formulation (do NOT reuse utils.py code)
# --------------------------------------------------------------------------- #
def _solve_box_dro_independent(dat, w, eps, lb, ub, a=A_CONST):
    """W1-DRO CVaR with box support, derived independently.

    Loss g(xi; x, tau) = tau + max_k (b_k^T xi + c_k) with pieces
    k=1: (b, c) = (0, 0) and k=2: (b, c) = (a x, a tau)  [METHOD.md 1.1].
    Support Xi = {xi : C xi <= d}, C = [I; -I], d = [ub; -lb].
    Esfahani-Kuhn (Thm 4.2, l2 ground norm so the dual norm is l2):

        min  tau + eps*lam + sum_i w_i s_i
        s.t. c_k + b_k^T xi_i + gam_ik^T (d - C xi_i) <= s_i     for all i, k
             || b_k - C^T gam_ik ||_2 <= lam                     for all i, k
             gam_ik >= 0,  lam >= 0,  x in simplex.

    Both pieces keep their own multiplier gam_ik (no shortcut for piece 1),
    so this is structurally different from createproblem_box_DRO, which
    eliminates piece 1 down to s >= 0 and splits gam into (gpos, gneg).
    """
    dat = np.asarray(dat, dtype=float)
    N, m = dat.shape
    C = np.vstack([np.eye(m), -np.eye(m)])          # (2m, m)
    d = np.concatenate([ub, -lb])                    # (2m,)

    x = cp.Variable(m)
    tau = cp.Variable()
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(N)
    cons = [cp.sum(x) == 1, x >= 0, x <= 1]
    for i in range(N):
        slack_i = d - C @ dat[i]                     # (2m,) >= 0 since xi_i in box
        # piece 1: b = 0, c = 0
        g1 = cp.Variable(2 * m, nonneg=True)
        cons.append(g1 @ slack_i <= s[i])
        cons.append(cp.norm(-C.T @ g1, 2) <= lam)
        # piece 2: b = a x, c = a tau
        g2 = cp.Variable(2 * m, nonneg=True)
        cons.append(a * tau + a * (dat[i] @ x) + g2 @ slack_i <= s[i])
        cons.append(cp.norm(a * x - C.T @ g2, 2) <= lam)
    prob = cp.Problem(cp.Minimize(tau + eps * lam + w @ s), cons)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal", prob.status
    return float(prob.value), x.value, float(tau.value)


def _solve_utils_socp(dat, w, eps, lb, ub):
    N, m = dat.shape
    prob, x, tau, dat_p, eps_p, w_p = createproblem_box_DRO(N, m, lb, ub, a=A_CONST)
    dat_p.value = dat
    eps_p.value = eps
    w_p.value = w
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal", prob.status
    return float(prob.value), x.value, float(tau.value)


def _instance(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(5, 9))
    N = int(rng.integers(8, 21))
    dat, lb, ub = generate_returns(N, m, seed=500 + seed)
    if seed % 2 == 0:
        w = np.ones(N) / N
    else:
        w = rng.dirichlet(np.ones(N))
    eps = [1e-4, 1e-3, 5e-3, 2e-2][seed % 4]
    return dat, w, eps, lb, ub


@pytest.mark.parametrize("seed", range(5))
def test_socp_matches_independent_formulation(seed):
    dat, w, eps, lb, ub = _instance(seed)
    v_utils, x_utils, _ = _solve_utils_socp(dat, w, eps, lb, ub)
    v_indep, x_indep, _ = _solve_box_dro_independent(dat, w, eps, lb, ub)
    assert abs(v_utils - v_indep) / max(1.0, abs(v_indep)) < 1e-6, (
        f"utils SOCP {v_utils} vs independent {v_indep}")
    # minimizers agree loosely (the optimum may be nearly flat in x)
    np.testing.assert_allclose(np.sum(x_utils), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(x_indep), 1.0, atol=1e-6)


def test_socp_eps_zero_matches_saa():
    """At eps = 0 the DRO SOCP collapses to the weighted SAA CVaR problem."""
    dat, w, _, lb, ub = _instance(0)
    v_utils, _, _ = _solve_utils_socp(dat, w, 0.0, lb, ub)
    m = dat.shape[1]
    x = cp.Variable(m)
    tau = cp.Variable()
    obj = tau + w @ cp.maximum(A_CONST * tau + A_CONST * (dat @ x), 0.0)
    saa = cp.Problem(cp.Minimize(obj), [cp.sum(x) == 1, x >= 0])
    saa.solve(solver=cp.CLARABEL)
    assert abs(v_utils - saa.value) / max(1.0, abs(saa.value)) < 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_bisection_consistent_at_socp_optimum(seed):
    """worst_case_value_box evaluated at the SOCP minimizer (x*, tau*) must
    reproduce the SOCP optimal value: the outer min is attained there."""
    dat, w, eps, lb, ub = _instance(seed)
    N = dat.shape[0]
    v_socp, x_star, tau_star = _solve_utils_socp(dat, w, eps, lb, ub)
    x_star = np.maximum(x_star, 0.0)         # clip solver-level negatives
    x_star /= x_star.sum()
    f_bis = worst_case_value_box(x_star, tau_star, dat, w, eps, lb, ub)
    assert abs(f_bis - v_socp) / max(1.0, abs(v_socp)) < 1e-6

    # and the SOCP value is a global lower bound on the worst-case value at
    # any other feasible point (here: uniform portfolio, tau = tau*)
    x_unif = np.ones(dat.shape[1]) / dat.shape[1]
    f_other = worst_case_value_box(x_unif, tau_star, dat, w, eps, lb, ub)
    assert f_other >= v_socp - 1e-8
