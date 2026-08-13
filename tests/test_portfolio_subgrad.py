"""Unit tests for the projected-subgradient machinery in ``portfolio.utils``:
``box_dro_subgrad_step`` (METHOD.md Sections 3.1-3.3) and ``project_simplex``.

Convergence methodology
-----------------------
The driver schedule eta_t = eta_0 / sqrt(t+1) (port_box/port_box.py) is very
slow on these CVaR instances: the tau-subgradient (1 + a * active mass, up to
|a|-1 = 4 in magnitude) dominates the x-subgradient (~5e-2), so after 1500
steps the gap to the SOCP optimum is still ~6e-3.  To actually verify that
the step routine drives the iterate to the optimum within the test budget we
use Polyak step sizes eta_t = (F_t - F*) / ||g_t||^2, which are valid exactly
when g_t is a true subgradient of the worst-case value -- so this test would
fail if the Danskin subgradient (or its regime split) were wrong.  F* comes
from the independent Esfahani-Kuhn SOCP.  The subgradient itself is
recomputed in-test from the ``return_state`` output of
``worst_case_value_box`` following METHOD.md Sections 3.1-3.2, and the
actual update is performed by ``box_dro_subgrad_step``.
"""
import cvxpy as cp
import numpy as np

from portfolio.utils import (
    box_dro_subgrad_step,
    createproblem_box_DRO,
    generate_returns,
    project_simplex,
    worst_case_value_box,
    worst_case_value_unbounded,
)

A_CONST = -5.0
M, N = 5, 15
EPS = 2e-3


def _instance():
    dat, lb, ub = generate_returns(N, M, seed=3)
    w = np.ones(N) / N
    return dat, w, lb, ub


def _socp_optimum(dat, w, lb, ub):
    prob, x, tau, dat_p, eps_p, w_p = createproblem_box_DRO(N, M, lb, ub, a=A_CONST)
    dat_p.value = dat
    eps_p.value = EPS
    w_p.value = w
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal"
    return float(prob.value)


def _subgradient(x, tau, dat, w, lb, ub):
    """Danskin subgradient of the worst-case value (METHOD.md 3.1-3.2),
    recomputed here from the oracle state as an independent mirror of the
    formula inside box_dro_subgrad_step."""
    F, _, zeta, active = worst_case_value_box(
        x, tau, dat, w, EPS, lb, ub, a=A_CONST, return_state=True)
    F_unb = worst_case_value_unbounded(x, tau, dat, w, EPS, a=A_CONST)
    if F >= F_unb - 1e-9 * (1.0 + abs(F_unb)):     # box has slack: keep regularizer
        scores = A_CONST * tau + A_CONST * (dat @ x)
        act = scores > 0.0
        gx = A_CONST * ((w * act) @ dat) + EPS * abs(A_CONST) * x / np.linalg.norm(x)
        gt = 1.0 + A_CONST * float(w @ act)
    else:                                          # box binds: moved atoms
        gx = A_CONST * ((w * active) @ zeta)
        gt = 1.0 + A_CONST * float(w @ active)
    return F, gx, gt


def _assert_simplex(x):
    assert x.min() >= -1e-12
    np.testing.assert_allclose(x.sum(), 1.0, atol=1e-9)


def test_polyak_subgradient_converges_to_socp_optimum():
    """~600 Polyak-stepped iterations of box_dro_subgrad_step reach the SOCP
    optimum to ~1e-4 (observed ~2e-5; asserted 1e-3 relative with 10x+
    margin).  Every iterate stays on the simplex."""
    dat, w, lb, ub = _instance()
    f_star = _socp_optimum(dat, w, lb, ub)

    x = np.ones(M) / M
    tau = 0.0
    best = np.inf
    for _ in range(600):
        F, gx, gt = _subgradient(x, tau, dat, w, lb, ub)
        assert F >= f_star - 1e-7          # SOCP optimum is a global lower bound
        best = min(best, F)
        g_sq = float(gx @ gx + gt * gt)
        eta = max(F - f_star, 1e-12) / max(g_sq, 1e-12)
        x, tau, F_ret = box_dro_subgrad_step(
            x, tau, dat, w, EPS, lb, ub, eta, a=A_CONST, line_search=False)
        _assert_simplex(x)
        # contract: returned F is the worst-case value at the PRE-step iterate
        np.testing.assert_allclose(F_ret, F, rtol=0, atol=1e-12)

    f_final = worst_case_value_box(x, tau, dat, w, EPS, lb, ub, a=A_CONST)
    rel_gap = (f_final - f_star) / max(1.0, abs(f_star))
    assert rel_gap < 1e-3, f"final {f_final} vs SOCP {f_star} (rel gap {rel_gap})"
    assert (best - f_star) / max(1.0, abs(f_star)) < 1e-3


def test_driver_schedule_descends_and_stays_feasible():
    """The driver's eta_t = eta_0/sqrt(t+1) schedule (port_box.py): 300 steps
    must strictly improve the objective, keep every iterate on the simplex,
    and never cross below the SOCP lower bound.  (This schedule converges too
    slowly to hit a tight gap in-budget; see module docstring.)"""
    dat, w, lb, ub = _instance()
    f_star = _socp_optimum(dat, w, lb, ub)

    x = np.ones(M) / M
    tau = 0.0
    f_init = worst_case_value_box(x, tau, dat, w, EPS, lb, ub, a=A_CONST)
    eta_0 = 0.2
    F = np.inf
    for t in range(300):
        eta = eta_0 / np.sqrt(t + 1)
        x, tau, F = box_dro_subgrad_step(
            x, tau, dat, w, EPS, lb, ub, eta, a=A_CONST, line_search=False)
        _assert_simplex(x)
        assert np.isfinite(tau)
        assert F >= f_star - 1e-7
    f_final = worst_case_value_box(x, tau, dat, w, EPS, lb, ub, a=A_CONST)
    # substantial progress toward the optimum (observed: ~39% of the gap
    # closed in 300 steps; assert at least 20%)
    assert f_final - f_star < 0.8 * (f_init - f_star)


# --------------------------------------------------------------------------- #
# project_simplex vs QP reference
# --------------------------------------------------------------------------- #
def _qp_project(v):
    n = v.size
    z = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(z - v)),
                      [cp.sum(z) == 1, z >= 0])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal"
    return z.value


def test_project_simplex_matches_qp():
    rng = np.random.default_rng(0)
    for scale in [0.1, 1.0, 10.0, 1000.0]:      # incl. far-outside points
        for n in [2, 5, 9]:
            v = rng.normal(size=n) * scale
            p = project_simplex(v)
            _assert_simplex(p)
            z = _qp_project(v)
            # interior-point accuracy degrades with the data scale (at
            # scale 1e3 Clarabel returns the exact vertex only to ~1e-5),
            # so compare iterates at a scale-aware tolerance ...
            np.testing.assert_allclose(p, z, atol=1e-6 * max(1.0, scale))
            # ... and check exact optimality via the objective: the
            # closed-form projection must be at least as close to v
            assert np.sum((p - v) ** 2) <= np.sum((z - v) ** 2) + 1e-9


def test_project_simplex_identity_on_feasible_points():
    rng = np.random.default_rng(1)
    for n in [2, 6, 11]:
        v = rng.dirichlet(np.ones(n))
        np.testing.assert_allclose(project_simplex(v), v, atol=1e-12)
    # vertices are fixed points
    e = np.zeros(7)
    e[3] = 1.0
    np.testing.assert_allclose(project_simplex(e), e, atol=1e-12)


def test_project_simplex_extremes():
    # all-equal vector projects to uniform
    np.testing.assert_allclose(project_simplex(np.full(4, 7.3)),
                               np.full(4, 0.25), atol=1e-12)
    # single coordinate
    np.testing.assert_allclose(project_simplex(np.array([-5.0])),
                               np.array([1.0]), atol=1e-12)
    # dominated by one huge coordinate -> that vertex
    v = np.array([1e6, 0.0, -3.0])
    np.testing.assert_allclose(project_simplex(v),
                               np.array([1.0, 0.0, 0.0]), atol=1e-9)
