"""Unit tests for the box-support W1-DRO worst-case value oracle.

``portfolio.utils.worst_case_value_box`` (nested 60x80 bisection, Sections
9.5 + 9.9 of the METHOD.md writeup in port_box/) is validated against the
independent Esfahani-Kuhn SOCP oracle ``worst_case_value_box_socp`` solved
with Clarabel.

Empirically the bisection agrees with the interior-point oracle to ~1e-7
relative (worst case over the randomized instances below), so the assertions
use 1e-6 relative with denominator max(1, |value|): a 10x margin.
"""
import numpy as np
import pytest

from portfolio.utils import (
    generate_returns,
    worst_case_value_box,
    worst_case_value_box_socp,
    worst_case_value_unbounded,
)

A_CONST = -5.0  # a = -1/(1-alpha), alpha = 0.8


def _rel(v, ref):
    return abs(v - ref) / max(1.0, abs(ref))


def _random_instance(seed):
    """Small random instance: (x, tau, dat, w, lb, ub)."""
    rng = np.random.default_rng(seed)
    m = int(rng.integers(5, 11))
    N = int(rng.integers(8, 31))
    dat, lb, ub = generate_returns(N, m, seed=1000 + seed)
    x = rng.dirichlet(np.ones(m))
    tau = float(rng.uniform(-0.05, 0.05))
    if seed % 3 == 0:
        w = np.ones(N) / N
    else:
        w = rng.dirichlet(np.ones(N))
    return x, tau, dat, w, lb, ub


# --------------------------------------------------------------------------- #
# Bisection vs SOCP oracle on randomized instances
# --------------------------------------------------------------------------- #
# 18 randomized (x, tau, data, weights) draws crossed with eps regimes:
# eps=0 (empirical branch), tiny, moderate, and large relative to the data
# scale (returns are ~1e-2, box width ~1e-2).
EPS_GRID = [0.0, 1e-4, 1e-3, 5e-3, 2e-2, 1e-1]


@pytest.mark.parametrize("seed", range(18))
def test_box_value_matches_socp(seed):
    x, tau, dat, w, lb, ub = _random_instance(seed)
    eps = EPS_GRID[seed % len(EPS_GRID)]
    f_bis = worst_case_value_box(x, tau, dat, w, eps, lb, ub, a=A_CONST)
    f_socp = worst_case_value_box_socp(x, tau, dat, w, eps, lb, ub, a=A_CONST)
    assert _rel(f_bis, f_socp) < 1e-6, (
        f"bisection {f_bis} vs SOCP {f_socp} (eps={eps})")


def test_box_value_matches_socp_single_sample():
    """N = 1 edge case."""
    rng = np.random.default_rng(42)
    dat, lb, ub = generate_returns(1, 5, seed=3)
    x = rng.dirichlet(np.ones(5))
    w = np.ones(1)
    for tau, eps in [(0.0, 1e-3), (0.02, 5e-3), (-0.01, 0.0)]:
        f_bis = worst_case_value_box(x, tau, dat, w, eps, lb, ub)
        f_socp = worst_case_value_box_socp(x, tau, dat, w, eps, lb, ub)
        assert _rel(f_bis, f_socp) < 1e-6


def test_eps_zero_equals_weighted_empirical_loss():
    """eps = 0 short-circuits to the exact weighted empirical CVaR loss."""
    for seed in range(5):
        x, tau, dat, w, lb, ub = _random_instance(seed)
        scores = A_CONST * tau + A_CONST * (dat @ x)
        emp = tau + float(w @ np.maximum(scores, 0.0))
        f0 = worst_case_value_box(x, tau, dat, w, 0.0, lb, ub)
        np.testing.assert_allclose(f0, emp, rtol=0, atol=1e-12)


def test_value_lower_bounded_by_empirical():
    """For eps > 0 the worst case dominates the empirical value (the empirical
    distribution is inside the ball); allow only bisection-level slack."""
    for seed in range(6):
        x, tau, dat, w, lb, ub = _random_instance(seed)
        emp = worst_case_value_box(x, tau, dat, w, 0.0, lb, ub)
        for eps in [1e-4, 1e-3, 1e-2]:
            f = worst_case_value_box(x, tau, dat, w, eps, lb, ub)
            assert f >= emp - 1e-9


def test_monotone_nondecreasing_in_eps():
    """A larger ball can only help the adversary: F nondecreasing in eps."""
    eps_seq = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 2e-1]
    for seed in [1, 4, 7]:
        x, tau, dat, w, lb, ub = _random_instance(seed)
        vals = [worst_case_value_box(x, tau, dat, w, e, lb, ub) for e in eps_seq]
        diffs = np.diff(vals)
        assert np.all(diffs >= -1e-9), f"non-monotone in eps: {vals}"


def test_huge_box_approaches_unbounded_closed_form():
    """With a very large box the support constraint never binds, so the exact
    box value must match the unbounded closed form (METHOD.md Section 1.3:
    the closed form eps*|a|*||x||_2 is exact when the box has slack).

    The residual scales with eps (finite lambda-bisection resolution);
    empirically ~5e-4 * eps, asserted with 10x margin."""
    rng = np.random.default_rng(0)
    dat, lb, ub = generate_returns(15, 6, seed=7)
    x = rng.dirichlet(np.ones(6))
    tau = 0.01
    w = np.ones(15) / 15
    big = 100.0
    for eps in [1e-4, 1e-3, 1e-2]:
        f_box = worst_case_value_box(x, tau, dat, w, eps, lb - big, ub + big)
        f_unb = worst_case_value_unbounded(x, tau, dat, w, eps)
        assert abs(f_box - f_unb) <= 5e-3 * eps + 1e-9
        # and the closed form is an upper bound in general (tight here)
        assert f_box <= f_unb + 1e-9


def test_unbounded_closed_form_upper_bounds_box_value():
    """On the true (binding) box the closed form is a strict upper bound for
    moderate eps (METHOD.md Section 1.3)."""
    x, tau, dat, w, lb, ub = _random_instance(2)
    for eps in [1e-3, 1e-2, 1e-1]:
        f_box = worst_case_value_box(x, tau, dat, w, eps, lb, ub)
        f_unb = worst_case_value_unbounded(x, tau, dat, w, eps)
        assert f_box <= f_unb + 1e-9
    # at eps large vs the box diameter the gap must be strictly positive
    assert (worst_case_value_unbounded(x, tau, dat, w, 1e-1)
            - worst_case_value_box(x, tau, dat, w, 1e-1, lb, ub)) > 1e-4
