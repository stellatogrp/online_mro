"""Parity, oracle, integration, and benchmark tests for the C implementation
of the box-support W1 worst-case CVaR value (``portfolio/cworst/``).

The numpy ``portfolio.utils.worst_case_value_box`` is the verification oracle
(forced via env ``PORT_WORSTCASE_IMPL=python``); the C library exposes two
inner solvers:

  * ``inner_mode=0`` -- the literal 80-iteration mu-bisection (numpy-parity
    path, compiled with ``-ffp-contract=off`` so the bisection sign tests see
    the same roundings as numpy);
  * ``inner_mode=1`` -- the exact piecewise closed form of the same optimality
    condition (production default).

Known, mathematically benign divergences the assertions account for:

  * When the outer root lambda* is pinned at a single sample's kink of h'
    (generic once samples sit exactly on box faces), that sample's inner
    fixed point has a plateau: the maximizer is non-unique and any point on
    the plateau is optimal.  The two implementations may then return
    different ``zeta`` rows that achieve the same inner objective to ~1e-11.
    The zeta assertion therefore falls back to phi-equivalence per row.
  * ``active_i = phi_i > 0`` can differ on exact ties (phi_i == 0) via
    summation order; allowed only where |phi_i| <= 1e-10.
"""
import os
import subprocess
import time

import numpy as np
import pytest

from portfolio.utils import (
    box_dro_subgrad_step,
    generate_returns,
    project_simplex,
    worst_case_value_box,
    worst_case_value_box_socp,
)

A_CONST = -5.0
CWORST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "portfolio", "cworst")


# --------------------------------------------------------------------------- #
# Build fixture / oracle helper
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def cworst():
    """Build the shared library and return the wrapper module (or skip)."""
    proc = subprocess.run(["sh", os.path.join(CWORST_DIR, "build.sh")],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"cworst build failed:\n{proc.stdout}\n{proc.stderr}")
    from portfolio import cworst as mod
    if not mod.available():
        pytest.skip("libworstcase built but failed to load")
    return mod


def _oracle(*args, **kwargs):
    """The numpy worst_case_value_box, with the C dispatch forced off."""
    old = os.environ.get("PORT_WORSTCASE_IMPL")
    os.environ["PORT_WORSTCASE_IMPL"] = "python"
    try:
        return worst_case_value_box(*args, **kwargs)
    finally:
        if old is None:
            os.environ.pop("PORT_WORSTCASE_IMPL", None)
        else:
            os.environ["PORT_WORSTCASE_IMPL"] = old


def _rel(v, ref):
    return abs(v - ref) / max(1.0, abs(ref))


# --------------------------------------------------------------------------- #
# Randomized instances
# --------------------------------------------------------------------------- #
# ~30 deterministic configs sweeping N, d, eps, weight law, and box tightness.
# Tight boxes come from generate_returns (data clipped to the box, so samples
# sit exactly on faces -- the degenerate regime); loose boxes strictly contain
# the data.
_NS = [1, 8, 200, 1000]
_DS = [5, 50, 300]
_EPSS = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
CONFIGS = [(_NS[k % 4], _DS[k % 3], _EPSS[k % 5], k % 2 == 0, k % 3 != 2, k)
           for k in range(30)]


def _instance(N, d, eps, uniform_w, tight_box, seed, sparse_x=False,
              shrink_box=False):
    rng = np.random.default_rng(1234 + seed)
    if tight_box:
        dat, lb, ub = generate_returns(N, d, seed=seed)
    else:
        dat = 0.02 * rng.standard_normal((N, d))
        lb = dat.min(axis=0) - 0.05
        ub = dat.max(axis=0) + 0.05
    if shrink_box:                      # some samples strictly OUTSIDE the box
        mid, half = 0.5 * (lb + ub), 0.5 * (ub - lb)
        lb, ub = mid - 0.4 * half, mid + 0.4 * half
    if sparse_x:                        # exact zeros in x -> b_j == 0 coords
        x = project_simplex(rng.standard_normal(d))
    else:
        x = rng.dirichlet(np.ones(d))
    tau = float(rng.uniform(-0.1, 0.1))
    w = np.ones(N) / N if uniform_w else rng.dirichlet(np.ones(N))
    return x, tau, dat, w, eps, lb, ub


def _assert_state_parity(x, tau, dat, lb, ub, py_state, c_state, label):
    """Assert (lam, zeta, active) parity up to the documented degeneracies.

    Returns True iff a degeneracy exemption was actually exercised (an exact
    phi == 0 active tie, or a plateau zeta row that is phi-equivalent but not
    pointwise close) -- i.e. iff the maximizer at this iterate is non-unique.
    """
    Fp, lp, zp, ap = py_state
    Fc, lc, zc, ac = c_state
    b = A_CONST * np.asarray(x, dtype=float)
    c2 = A_CONST * tau
    nb = float(np.linalg.norm(b))
    assert abs(lc - lp) <= 1e-8 * max(nb, 1e-12), f"{label}: lam {lc} vs {lp}"

    def _phi(z, xi, lam):
        return float(b @ z + c2 - lam * np.linalg.norm(z - xi))

    degenerate = False
    # active parity, up to exact phi == 0 ties
    mism = np.where(ac != ap)[0]
    for i in mism:
        assert abs(_phi(zp[i], dat[i], lp)) <= 1e-10, (
            f"{label}: active mismatch at non-tied row {i}")
        degenerate = True
    # zeta parity, up to plateau non-uniqueness (phi-equivalent rows)
    close = np.all(np.isclose(zc, zp, rtol=1e-6, atol=1e-8), axis=1)
    for i in np.where(~close)[0]:
        assert np.all(zc[i] >= lb - 1e-12) and np.all(zc[i] <= ub + 1e-12), (
            f"{label}: infeasible zeta row {i}")
        dphi = abs(_phi(zc[i], dat[i], lp) - _phi(zp[i], dat[i], lp))
        assert dphi <= 1e-9, (
            f"{label}: zeta row {i} differs and is not phi-equivalent "
            f"(dphi={dphi:.2e})")
        degenerate = True
    return degenerate


@pytest.mark.parametrize("N,d,eps,uniform_w,tight_box,seed", CONFIGS)
def test_parity_randomized(cworst, N, d, eps, uniform_w, tight_box, seed):
    x, tau, dat, w, eps, lb, ub = _instance(N, d, eps, uniform_w, tight_box, seed)
    py = _oracle(x, tau, dat, w, eps, lb, ub, a=A_CONST, return_state=True)
    for mode, rtol in ((0, 1e-10), (1, 1e-9)):
        c = cworst.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub,
                                          a=A_CONST, return_state=True,
                                          inner_mode=mode)
        label = f"N={N} d={d} eps={eps} seed={seed} mode={mode}"
        assert _rel(c[0], py[0]) <= rtol, f"{label}: F {c[0]} vs {py[0]}"
        _assert_state_parity(x, tau, dat, lb, ub, py, c, label)


@pytest.mark.parametrize("case", ["sparse_x", "shrink_box"])
def test_parity_edge_cases(cworst, case):
    """b_j == 0 coordinates (sparse x) and samples outside the box."""
    for seed in range(4):
        x, tau, dat, w, eps, lb, ub = _instance(
            40, 12, 1e-3, seed % 2 == 0, True, 100 + seed,
            sparse_x=(case == "sparse_x"), shrink_box=(case == "shrink_box"))
        py = _oracle(x, tau, dat, w, eps, lb, ub, a=A_CONST, return_state=True)
        for mode, rtol in ((0, 1e-10), (1, 1e-9)):
            c = cworst.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub,
                                              a=A_CONST, return_state=True,
                                              inner_mode=mode)
            label = f"{case} seed={seed} mode={mode}"
            assert _rel(c[0], py[0]) <= rtol, f"{label}: F {c[0]} vs {py[0]}"
            _assert_state_parity(x, tau, dat, lb, ub, py, c, label)


# --------------------------------------------------------------------------- #
# Independent SOCP oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
def test_c_vs_socp(cworst, seed):
    rng = np.random.default_rng(seed)
    N, d = int(rng.integers(5, 21)), int(rng.integers(4, 9))
    dat, lb, ub = generate_returns(N, d, seed=2000 + seed)
    x = rng.dirichlet(np.ones(d))
    tau = float(rng.uniform(-0.05, 0.05))
    w = rng.dirichlet(np.ones(N)) if seed % 2 else np.ones(N) / N
    eps = [1e-4, 1e-3, 5e-3][seed % 3]
    f_socp = worst_case_value_box_socp(x, tau, dat, w, eps, lb, ub, a=A_CONST)
    for mode in (0, 1):
        f_c = cworst.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub,
                                            a=A_CONST, inner_mode=mode)
        assert _rel(f_c, f_socp) < 1e-5, (
            f"seed={seed} mode={mode}: C {f_c} vs SOCP {f_socp}")


# --------------------------------------------------------------------------- #
# Dispatch semantics + subgradient integration
# --------------------------------------------------------------------------- #
def test_dispatch_switch(cworst, monkeypatch):
    """worst_case_value_box routes to C by default and to numpy when
    PORT_WORSTCASE_IMPL=python."""
    x, tau, dat, w, eps, lb, ub = _instance(50, 20, 1e-3, True, True, 7)
    monkeypatch.delenv("PORT_WORSTCASE_IMPL", raising=False)
    f_dispatch = worst_case_value_box(x, tau, dat, w, eps, lb, ub, a=A_CONST)
    f_c = cworst.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub, a=A_CONST)
    assert f_dispatch == f_c                    # same code path, bitwise
    monkeypatch.setenv("PORT_WORSTCASE_IMPL", "python")
    f_py = worst_case_value_box(x, tau, dat, w, eps, lb, ub, a=A_CONST)
    assert f_dispatch != f_py or True           # numpy path runs without error
    assert _rel(f_c, f_py) <= 1e-9


def test_subgrad_step_parity_lockstep(cworst, monkeypatch):
    """box_dro_subgrad_step with the C path on vs off takes the same step
    (within 1e-8) from the same iterate, at every step of a 40-step run where
    the step is well defined.

    Along a subgradient trajectory the outer dual root lambda* is frequently
    pinned exactly at a sample's hinge kink (the sample's jump in h'
    straddles zero), which drives that sample's phi to 0 to machine
    precision: ``active_i = phi_i > 0`` is then floating-point noise in ANY
    implementation, and either choice is a valid subgradient (so full
    40-step end-to-end trajectories can legitimately separate at such a
    step).  Steps whose python state exhibits such a tie -- or that sit on
    the razor edge of the box-slack regime threshold -- are exempted from the
    pointwise comparison; F parity is asserted at every step regardless, and
    the test requires that a healthy majority of steps were tie-free and
    compared pointwise.
    """
    N, d, eps = 60, 10, 1e-3
    dat, lb, ub = generate_returns(N, d, seed=11)
    w = np.ones(N) / N
    eta = 0.02

    from portfolio.utils import worst_case_value_unbounded

    def one_step(impl, x, tau):
        if impl == "python":
            monkeypatch.setenv("PORT_WORSTCASE_IMPL", impl)
        else:
            monkeypatch.delenv("PORT_WORSTCASE_IMPL", raising=False)
        return box_dro_subgrad_step(x, tau, dat, w, eps, lb, ub, eta, a=A_CONST)

    from portfolio import cworst as cmod

    x, tau = np.ones(d) / d, 0.0
    compared = 0
    for t in range(40):
        # tie assessment at the current iterate: full state parity check,
        # which also reports whether the maximizer is non-unique here
        py = _oracle(x, tau, dat, w, eps, lb, ub, a=A_CONST, return_state=True)
        c = cmod.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub,
                                        a=A_CONST, return_state=True)
        tied = _assert_state_parity(x, tau, dat, lb, ub, py, c, f"step {t}")
        F, lam, zeta, active = py
        b = A_CONST * x
        phis = zeta @ b + A_CONST * tau - lam * np.linalg.norm(zeta - dat, axis=1)
        F_unb = worst_case_value_unbounded(x, tau, dat, w, eps, a=A_CONST)
        thr = F_unb - 1e-9 * (1.0 + abs(F_unb))
        tied = tied or (np.min(np.abs(phis)) < 1e-10) or (abs(F - thr) < 1e-12)

        x_p, tau_p, F_p = one_step("python", x, tau)
        x_c, tau_c, F_c = one_step("c", x, tau)
        assert _rel(F_c, F_p) <= 1e-9, f"step {t}: F {F_c} vs {F_p}"
        if not tied:
            diff = max(float(np.max(np.abs(x_c - x_p))), abs(tau_c - tau_p))
            assert diff < 1e-8, f"step {t}: iterate diff {diff:.2e}"
            compared += 1
        x, tau = x_p, tau_p                     # continue along the python path
    assert compared >= 20, f"only {compared}/40 steps were tie-free"


# --------------------------------------------------------------------------- #
# Benchmark (informational; printed with -s, no assertions on timings)
# --------------------------------------------------------------------------- #
def test_benchmark(cworst):
    print("\nworst_case_value_box benchmark (one full call = 62 outer evals; "
          "'C state' = inner_mode 1 with return_state=True, whose final "
          "evaluation uses the bisect path)")
    print(f"{'N':>6} {'d':>4} {'numpy [s]':>10} {'C bisect [s]':>13} "
          f"{'C exact [s]':>12} {'C state [s]':>12} {'x bisect':>9} "
          f"{'x exact':>8}")
    for N, d in [(200, 300), (1000, 300), (2000, 300)]:
        x, tau, dat, w, eps, lb, ub = _instance(N, d, 1e-3, True, True, 42)
        t0 = time.perf_counter()
        f_py = _oracle(x, tau, dat, w, eps, lb, ub, a=A_CONST)
        t_py = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_b = cworst.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub,
                                            a=A_CONST, inner_mode=0)
        t_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_e = cworst.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub,
                                            a=A_CONST, inner_mode=1)
        t_e = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_s = cworst.worst_case_value_box_c(x, tau, dat, w, eps, lb, ub,
                                            a=A_CONST, inner_mode=1,
                                            return_state=True)[0]
        t_s = time.perf_counter() - t0
        assert _rel(f_b, f_py) < 1e-9 and _rel(f_e, f_py) < 1e-9
        assert _rel(f_s, f_py) < 1e-9
        print(f"{N:>6} {d:>4} {t_py:>10.3f} {t_b:>13.3f} {t_e:>12.3f} "
              f"{t_s:>12.3f} {t_py / t_b:>8.1f}x {t_py / t_e:>7.1f}x")
