"""Validation of ``portfolio.direct_socp`` against the CVXPY oracles.

``BoxDROSocp`` must reproduce ``createproblem_box_DRO`` (CVXPY + Clarabel,
ignore_dpp) to solver accuracy, agree with the independent
``worst_case_value_box`` evaluation at its own optimizer, and be deterministic.
``SaaCvarLp`` must reproduce ``create_scenario`` / ``create_scenario_cluster``
and coincide with ``BoxDROSocp`` at eps=0 (the SAA limit).  Production-size
timing comparisons (m=300; N=2000 for the SOCP, N in {2000, 5000} for the SAA
LP) are marked ``slow`` and print (do not assert) CVXPY totals vs direct
assembly+solve totals.
"""
import time

import cvxpy as cp
import numpy as np
import pytest

from portfolio.direct_socp import BoxDROSocp, SaaCvarLp
from portfolio.utils import (
    create_scenario,
    create_scenario_cluster,
    createproblem_box_DRO,
    worst_case_value_box,
)


A_CONST = -5.0


def _random_instance(N, m, seed):
    """Random clipped return panel + random box + random weights."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.01, 0.03, (N, m))
    lb = np.quantile(raw, 0.35, axis=0)
    ub = np.quantile(raw, 0.65, axis=0)
    dat = np.clip(raw, lb, ub)
    w = rng.random(N)
    w /= w.sum()
    return dat, lb, ub, w


def _cvxpy_solve(dat, lb, ub, w, eps, a=A_CONST):
    N, m = dat.shape
    prob, x, tau, p_dat, p_eps, p_w = createproblem_box_DRO(N, m, lb, ub, a=a)
    p_dat.value = dat
    p_eps.value = eps
    p_w.value = w
    prob.solve(solver=cp.CLARABEL, ignore_dpp=True, verbose=False)
    assert prob.status in ('optimal', 'optimal_inaccurate')
    return float(prob.value), np.asarray(x.value), float(tau.value)


CASES = [(N, m, eps)
         for N in (5, 50, 250)
         for m in (10, 50)
         for eps in (0.0, 1e-3, 1e-2)]


@pytest.mark.parametrize("N,m,eps", CASES)
def test_matches_cvxpy_oracle(N, m, eps):
    dat, lb, ub, w = _random_instance(N, m, seed=1000 * N + 10 * m + int(1e4 * eps))
    obj_c, x_c, tau_c = _cvxpy_solve(dat, lb, ub, w, eps)

    direct = BoxDROSocp(m, lb, ub, a=A_CONST)
    obj_d, x_d, tau_d, solve_t, status = direct.solve(dat, w, eps)

    assert status in ('optimal', 'optimal_inaccurate')
    assert np.isfinite(solve_t) and solve_t >= 0.0
    # rel 1e-6 with a small absolute floor: Clarabel's tol_gap_abs=1e-8 holds
    # on the *equilibrated* problem, and the CVXPY canonicalization scales rows
    # differently from the direct assembly, so realized gaps differ by ~1e-7
    # when the optimal value is near zero.
    assert obj_d == pytest.approx(obj_c, rel=1e-6, abs=2e-7)

    if np.max(np.abs(x_d - x_c)) < 1e-4 and abs(tau_d - tau_c) < 1e-4:
        return
    # Degenerate optimizer (non-unique argmin): fall back to an objective-gap
    # check -- the direct solution must be optimal for the same DRO objective.
    F_d = worst_case_value_box(x_d, tau_d, dat, w, eps, lb, ub, a=A_CONST)
    assert F_d == pytest.approx(obj_c, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("N,m,eps", [(30, 10, 1e-2), (60, 20, 1e-3)])
def test_consistent_with_worst_case_value_box(N, m, eps):
    """F(x*, tau*) evaluated independently must equal the direct objective."""
    dat, lb, ub, w = _random_instance(N, m, seed=7 * N + m)
    direct = BoxDROSocp(m, lb, ub, a=A_CONST)
    obj_d, x_d, tau_d, _, status = direct.solve(dat, w, eps)
    assert status in ('optimal', 'optimal_inaccurate')
    F = worst_case_value_box(x_d, tau_d, dat, w, eps, lb, ub, a=A_CONST)
    assert F == pytest.approx(obj_d, rel=1e-5, abs=1e-7)


def test_deterministic():
    dat, lb, ub, w = _random_instance(80, 15, seed=3)
    direct = BoxDROSocp(15, lb, ub, a=A_CONST)
    o1, x1, t1, _, s1 = direct.solve(dat, w, 5e-3)
    o2, x2, t2, _, s2 = direct.solve(dat, w, 5e-3)
    assert o1 == o2
    assert t1 == t2
    assert s1 == s2
    assert np.array_equal(x1, x2)


def test_failure_returns_nan_sentinels():
    """An infeasible-shaped failure must produce the safe_solve NaN convention."""
    direct = BoxDROSocp(4, np.zeros(4), np.ones(4), a=A_CONST)
    # wrong data shape -> exception path
    obj, x, tau, solve_t, status = direct.solve(np.zeros((5, 3)), np.ones(5) / 5, 1e-2)
    assert np.isnan(obj) and np.isnan(tau) and np.isnan(solve_t)
    assert x is None
    assert status not in ('optimal', 'optimal_inaccurate')


@pytest.mark.slow
def test_timing_production_size():
    """N=2000, m=300: print CVXPY total vs direct assembly+solve. No asserts
    on the timings themselves (informational)."""
    N, m, eps = 2000, 300, 1e-2
    dat, lb, ub, w = _random_instance(N, m, seed=42)

    direct = BoxDROSocp(m, lb, ub, a=A_CONST)
    t0 = time.perf_counter()
    P, q, A, b, cones = direct._assemble(dat, w, eps)
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    obj_d, x_d, tau_d, clarabel_t, status = direct.solve(dat, w, eps)
    t_direct_total = time.perf_counter() - t0
    assert status in ('optimal', 'optimal_inaccurate')

    t0 = time.perf_counter()
    obj_c, x_c, tau_c = _cvxpy_solve(dat, lb, ub, w, eps)
    t_cvxpy_total = time.perf_counter() - t0

    assert obj_d == pytest.approx(obj_c, rel=1e-5, abs=1e-7)
    print(f"\n[N={N}, m={m}] assembly {t_assemble:.2f}s | "
          f"direct total {t_direct_total:.2f}s (clarabel {clarabel_t:.2f}s) | "
          f"cvxpy total {t_cvxpy_total:.2f}s | "
          f"speedup {t_cvxpy_total / t_direct_total:.2f}x")


# --------------------------------------------------------------------------- #
# SaaCvarLp (SAA CVaR LP, eps -> 0 limit of the DRO problem)
# --------------------------------------------------------------------------- #
def _saa_value(x, tau, dat, w, a=A_CONST):
    """Weighted SAA objective  sum(w)*tau + sum_i w_i max(0, a*tau + a<xi_i,x>)."""
    x = np.asarray(x, dtype=float)
    return float(w.sum() * tau + w @ np.maximum(a * tau + a * (dat @ x), 0.0))


def _cvxpy_saa_solve(dat, w, uniform, a=A_CONST):
    """Solve the SAA problem with the CVXPY builders used in production
    (``create_scenario`` for uniform weights, ``create_scenario_cluster``
    for general weights)."""
    N, m = dat.shape
    if uniform:
        prob, x, tau = create_scenario(dat, m, N, a=a)
    else:
        prob, x, tau = create_scenario_cluster(dat, m, N, w, a=a)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    assert prob.status in ('optimal', 'optimal_inaccurate')
    return float(prob.value), np.asarray(x.value), float(tau.value)


# 10 random small instances; ``uniform`` picks the create_scenario oracle
# (the one used in task_dro_exact) vs the weighted create_scenario_cluster.
SAA_CASES = [
    (5, 10, True), (5, 10, False), (5, 50, True),
    (50, 10, True), (50, 10, False), (50, 50, False),
    (250, 10, True), (250, 10, False), (250, 50, True), (250, 50, False),
]


@pytest.mark.parametrize("N,m,uniform", SAA_CASES)
def test_saa_matches_cvxpy_oracle(N, m, uniform):
    dat, _, _, w = _random_instance(N, m, seed=17 * N + 3 * m + int(uniform))
    if uniform:
        w = np.ones(N) / N
    obj_c, x_c, tau_c = _cvxpy_saa_solve(dat, w, uniform)

    direct = SaaCvarLp(m, a=A_CONST)
    obj_d, x_d, tau_d, solve_t, status = direct.solve(dat, w)

    assert status in ('optimal', 'optimal_inaccurate')
    assert np.isfinite(solve_t) and solve_t >= 0.0
    assert obj_d == pytest.approx(obj_c, rel=1e-6, abs=2e-7)

    if np.max(np.abs(x_d - x_c)) < 1e-4 and abs(tau_d - tau_c) < 1e-4:
        return
    # LP optimizers are often non-unique: fall back to an objective-gap check.
    assert _saa_value(x_d, tau_d, dat, w) == pytest.approx(
        obj_c, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("N,m", [(5, 10), (50, 10), (120, 30)])
def test_saa_equals_dro_at_eps_zero(N, m):
    """At eps=0 the box-DRO SOCP collapses to SAA (lam free, gpos=gneg=0
    optimal), so the two direct solvers must agree to solver tolerance."""
    dat, lb, ub, w = _random_instance(N, m, seed=100 + N + m)
    obj_saa, x_s, tau_s, _, st_s = SaaCvarLp(m, a=A_CONST).solve(dat, w)
    obj_dro, x_d, tau_d, _, st_d = BoxDROSocp(m, lb, ub, a=A_CONST).solve(
        dat, w, eps=0.0)
    assert st_s in ('optimal', 'optimal_inaccurate')
    assert st_d in ('optimal', 'optimal_inaccurate')
    # equality holds exactly in the mathematical problems; numerically the two
    # cone programs are conditioned differently, so assert to solver tolerance.
    assert obj_saa == pytest.approx(obj_dro, rel=1e-6, abs=5e-7)


def test_saa_deterministic():
    dat, _, _, w = _random_instance(80, 15, seed=11)
    direct = SaaCvarLp(15, a=A_CONST)
    o1, x1, t1, _, s1 = direct.solve(dat, w)
    o2, x2, t2, _, s2 = direct.solve(dat, w)
    assert o1 == o2
    assert t1 == t2
    assert s1 == s2
    assert np.array_equal(x1, x2)


def test_saa_failure_returns_nan_sentinels():
    direct = SaaCvarLp(4, a=A_CONST)
    # wrong data shape -> exception path
    obj, x, tau, solve_t, status = direct.solve(np.zeros((5, 3)), np.ones(5) / 5)
    assert np.isnan(obj) and np.isnan(tau) and np.isnan(solve_t)
    assert x is None
    assert status not in ('optimal', 'optimal_inaccurate')


@pytest.mark.slow
@pytest.mark.parametrize("N", [2000, 5000])
def test_saa_timing_production_size(N):
    """m=300: print CVXPY total (create_scenario build+solve) vs direct
    assembly+solve for the SAA LP.  No asserts on the timings (informational).
    Unlike the SOCP (whose CVXPY model loops over N norm constraints), the SAA
    LP canonicalizes vectorized and cheaply (~0.2 s at N=5000), so the solve
    dominates and the direct path gives only a modest speedup."""
    m = 300
    dat, _, _, _ = _random_instance(N, m, seed=42)
    w = np.ones(N) / N

    direct = SaaCvarLp(m, a=A_CONST)
    t0 = time.perf_counter()
    P, q, A, b, cones = direct._assemble(dat, w)
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    obj_d, x_d, tau_d, clarabel_t, status = direct.solve(dat, w)
    t_direct_total = time.perf_counter() - t0
    assert status in ('optimal', 'optimal_inaccurate')

    t0 = time.perf_counter()
    obj_c, x_c, tau_c = _cvxpy_saa_solve(dat, w, uniform=True)
    t_cvxpy_total = time.perf_counter() - t0

    assert obj_d == pytest.approx(obj_c, rel=1e-5, abs=1e-7)
    print(f"\n[SAA N={N}, m={m}] assembly {t_assemble:.2f}s | "
          f"direct total {t_direct_total:.2f}s (clarabel {clarabel_t:.2f}s) | "
          f"cvxpy total {t_cvxpy_total:.2f}s | "
          f"speedup {t_cvxpy_total / t_direct_total:.2f}x")
