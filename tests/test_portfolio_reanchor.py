"""Tests for the periodic exact re-anchor solves in the portfolio subgradient
paths (``task_mro_subgrad`` / ``task_dro_subgrad``), mirroring the svm
driver's ``_milp_warmstart`` gate.

Instrumentation: ``box_dro_subgrad_step`` is wrapped with a spy that records
each call's input iterate, cluster data snapshot, and output iterate.  Because
the drivers chain the iterate across timesteps, ``x_in`` of the call at t+1
must equal ``x_out`` of the call at t EXACTLY unless a re-anchor solve
replaced the iterate at t -- which localizes jumps to precisely the re-anchor
steps.  ``safe_solve`` (mro) and ``BoxDROSocp.solve`` (dro) are wrapped to
record the re-anchor solver times for the timing-fairness checks.

``solve_interval=0`` must reproduce the legacy pure-subgradient trajectory
(no jumps anywhere, no re-anchor solves issued).
"""
import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

import portfolio.direct_socp as direct_socp_mod
import portfolio.methods as methods
from portfolio.utils import createproblem_box_DRO, generate_returns

from .conftest import PAPER_DIR, assert_ran, run_cli

# Tiny but non-degenerate instance shared by the direct-call tests
# (box_q=0.05 like test_equivalence_legacy: light clipping keeps the tiny
# initial sample free of duplicate points, which KMeans needs).
M_ASSETS = 8
N_INIT = 5
K = 3
Q = 50
T = 8
POWER = 0.05
ETA_0 = 0.1
EPS_INIT = [0.005]
ALPHA = 0.8
TRAIN_SIZE, TEST_SIZE = 1000, 400

DATA, LB, UB = generate_returns(1600, M_ASSETS, seed=12345, box_q=0.05)


# --------------------------------------------------------------------------- #
# Spies
# --------------------------------------------------------------------------- #
def _spy_subgrad_step(monkeypatch):
    """Record (x_in, tau_in, dat, w, eps, x_out, tau_out) per call."""
    calls = []
    real_step = methods.box_dro_subgrad_step

    def spy(x, tau, dat, w, eps, lb, ub, eta, **kw):
        out = real_step(x, tau, dat, w, eps, lb, ub, eta, **kw)
        calls.append({
            'x_in': np.array(x, dtype=float), 'tau_in': float(tau),
            'dat': np.array(dat, dtype=float), 'w': np.array(w, dtype=float),
            'eps': float(eps),
            'x_out': np.array(out[0], dtype=float), 'tau_out': float(out[1]),
        })
        return out

    monkeypatch.setattr(methods, 'box_dro_subgrad_step', spy)
    return calls


def _spy_safe_solve(monkeypatch):
    """Record every re-anchor safe_solve (name, t, ok, solver time)."""
    solves = []
    real_safe = methods.safe_solve

    def spy(problem, name='?', t=None, **kw):
        ok = real_safe(problem, name=name, t=t, **kw)
        if 'reanchor' in name:
            solves.append({
                'name': name, 't': t, 'ok': ok,
                'solve_time': float(problem.solver_stats.solve_time) if ok else np.nan,
            })
        return ok

    monkeypatch.setattr(methods, 'safe_solve', spy)
    return solves


def _spy_direct_solve(monkeypatch):
    """Record every BoxDROSocp.solve (dro re-anchor) call and result."""
    solves = []
    real_solve = direct_socp_mod.BoxDROSocp.solve

    def spy(self, dat, w, eps, **kw):
        obj, x, tau, solve_time, status = real_solve(self, dat, w, eps, **kw)
        solves.append({
            'obj': obj, 'x': None if x is None else np.array(x, dtype=float),
            'tau': tau, 'solve_time': solve_time, 'status': status,
        })
        return obj, x, tau, solve_time, status

    monkeypatch.setattr(direct_socp_mod.BoxDROSocp, 'solve', spy)
    return solves


# --------------------------------------------------------------------------- #
# Direct task runs
# --------------------------------------------------------------------------- #
def _run_mro(tmp_path, solve_interval, t_solve_list):
    df = methods.task_mro_subgrad(
        0, K, T, N_INIT, DATA, LB, UB, ETA_0, 0, str(tmp_path) + '/', POWER,
        [(0, 0)], EPS_INIT, M_ASSETS, TRAIN_SIZE, TEST_SIZE, 0, Q,
        1, [], 10**6, 1.25, 1, False, 10**6,
        solve_interval=solve_interval, t_solve_list=t_solve_list)
    assert df is not None, "task_mro_subgrad swallowed an exception"
    return df


def _run_dro(tmp_path, solve_interval, t_solve_list):
    df = methods.task_dro_subgrad(
        0, T, N_INIT, DATA, LB, UB, ETA_0, 0, POWER, [(0, 0)], EPS_INIT,
        ALPHA, M_ASSETS, TRAIN_SIZE, TEST_SIZE, 0, 1, [], False,
        str(tmp_path) + '/', 10**6,
        solve_interval=solve_interval, t_solve_list=t_solve_list)
    assert df is not None, "task_dro_subgrad swallowed an exception"
    return df


def _exact_clustered_solution(dat, w, eps, a=-5.0):
    """Independent exact solve of the clustered box SOCP (same builder)."""
    prob, x, tau, p_dat, p_eps, p_w = createproblem_box_DRO(
        dat.shape[0], M_ASSETS, LB, UB, a=a)
    p_dat.value = dat
    p_eps.value = eps
    p_w.value = w
    prob.solve(solver=cp.CLARABEL, ignore_dpp=True, verbose=False)
    assert prob.status in ('optimal', 'optimal_inaccurate')
    return np.asarray(x.value), float(tau.value), float(prob.value)


def _jump(cur, nxt):
    """Iterate discontinuity between consecutive subgradient calls."""
    return (np.linalg.norm(nxt['x_in'] - cur['x_out'])
            + abs(nxt['tau_in'] - cur['tau_out']))


# --------------------------------------------------------------------------- #
# MRO path
# --------------------------------------------------------------------------- #
def test_mro_reanchor_replaces_iterate_and_charges_time(tmp_path, monkeypatch):
    """solve_interval=2: at t in {2, 4, 6} both iterates are replaced by the
    EXACT clustered solve (checked against an independent solve of the same
    problem, tol 1e-6); jumps happen at exactly those steps; the recorded
    per-iteration times at those steps include the re-anchor solve time."""
    calls = _spy_subgrad_step(monkeypatch)
    solves = _spy_safe_solve(monkeypatch)
    df = _run_mro(tmp_path, solve_interval=2, t_solve_list=())

    # online and MRO subgradient calls alternate, one pair per t = 1..T-1
    online, mro = calls[0::2], calls[1::2]
    assert len(online) == len(mro) == T - 1
    reanchor_ts = {2, 4, 6}

    # every re-anchor solve succeeded, at exactly the expected steps
    assert sorted(s['t'] for s in solves if s['name'] == 'online_reanchor') == sorted(reanchor_ts)
    assert sorted(s['t'] for s in solves if s['name'] == 'MRO_reanchor') == sorted(reanchor_ts)
    assert all(s['ok'] for s in solves)

    for stream in (online, mro):
        for j in range(len(stream) - 1):
            t = j + 1                      # timestep of stream[j]
            if t in reanchor_ts:
                # (a) the next call's input is the exact clustered solution
                x_star, tau_star, _ = _exact_clustered_solution(
                    stream[j]['dat'], stream[j]['w'], stream[j]['eps'])
                np.testing.assert_allclose(stream[j + 1]['x_in'], x_star, atol=1e-6)
                np.testing.assert_allclose(stream[j + 1]['tau_in'], tau_star, atol=1e-6)
                # (b) ... which is a genuine jump off the subgradient path
                assert _jump(stream[j], stream[j + 1]) > 1e-9
            else:
                # (b) no re-anchor: the iterate chains EXACTLY
                assert np.array_equal(stream[j + 1]['x_in'], stream[j]['x_out'])
                assert stream[j + 1]['tau_in'] == stream[j]['tau_out']

    # (c) timing fairness: recorded per-iteration time at re-anchor steps
    # includes the re-anchor solve (>= the solver's own solve time)
    for s in solves:
        col = 'online_time' if s['name'] == 'online_reanchor' else 'MRO_time'
        recorded = float(df.loc[df['t'] == s['t'], col].iloc[0])
        assert recorded >= s['solve_time'] - 1e-12, (
            f"{col} at t={s['t']}: recorded {recorded} < solve {s['solve_time']}")


def test_mro_solve_interval_zero_never_jumps(tmp_path, monkeypatch):
    """solve_interval=0 (legacy behavior): the (x, tau) trajectory is exactly
    the chained subgradient path -- no jumps, no re-anchor solves at all."""
    calls = _spy_subgrad_step(monkeypatch)
    solves = _spy_safe_solve(monkeypatch)
    _run_mro(tmp_path, solve_interval=0, t_solve_list=(5,))

    assert solves == []                    # no re-anchor solves issued
    online, mro = calls[0::2], calls[1::2]
    for stream in (online, mro):
        for j in range(len(stream) - 1):
            assert np.array_equal(stream[j + 1]['x_in'], stream[j]['x_out'])
            assert stream[j + 1]['tau_in'] == stream[j]['tau_out']


def test_mro_t_solve_list_gate(tmp_path, monkeypatch):
    """A large solve_interval with t_solve_list=(3,) re-anchors at exactly t=3."""
    calls = _spy_subgrad_step(monkeypatch)
    solves = _spy_safe_solve(monkeypatch)
    _run_mro(tmp_path, solve_interval=100, t_solve_list=(3,))

    assert sorted(s['t'] for s in solves if s['name'] == 'online_reanchor') == [3]
    assert sorted(s['t'] for s in solves if s['name'] == 'MRO_reanchor') == [3]
    online, mro = calls[0::2], calls[1::2]
    for stream in (online, mro):
        for j in range(len(stream) - 1):
            t = j + 1
            if t == 3:
                assert _jump(stream[j], stream[j + 1]) > 1e-9
            else:
                assert np.array_equal(stream[j + 1]['x_in'], stream[j]['x_out'])
                assert stream[j + 1]['tau_in'] == stream[j]['tau_out']


# --------------------------------------------------------------------------- #
# DRO path
# --------------------------------------------------------------------------- #
def test_dro_reanchor_replaces_iterate_and_charges_time(tmp_path, monkeypatch):
    """solve_interval=2: the DRO iterate is replaced by the exact full-sample
    solve (direct BoxDROSocp; value checked against the CVXPY builder) at
    t in {2, 4, 6}, chains exactly elsewhere, and the recorded DRO_time at
    those steps includes the re-anchor solve time."""
    calls = _spy_subgrad_step(monkeypatch)
    solves = _spy_direct_solve(monkeypatch)
    df = _run_dro(tmp_path, solve_interval=2, t_solve_list=())

    assert len(calls) == T                 # one DRO subgradient step per t
    reanchor_ts = [2, 4, 6]
    assert len(solves) == len(reanchor_ts)
    assert all(s['status'] in ('optimal', 'optimal_inaccurate') for s in solves)

    for i, t in enumerate(reanchor_ts):
        s = solves[i]
        # the next step's input iterate is exactly the re-anchor solution
        np.testing.assert_array_equal(calls[t + 1]['x_in'], s['x'])
        assert calls[t + 1]['tau_in'] == s['tau']
        assert _jump(calls[t], calls[t + 1]) > 1e-9
        # independent exact solve of the same full-sample problem (CVXPY
        # builder): optimal values agree to solver tolerance
        _, _, obj_star = _exact_clustered_solution(
            calls[t]['dat'], calls[t]['w'], calls[t]['eps'])
        assert s['obj'] == pytest.approx(obj_star, rel=1e-5, abs=1e-7)
        # timing fairness on the DRO_time column
        recorded = float(df.loc[df['t'] == t, 'DRO_time'].iloc[0])
        assert recorded >= s['solve_time'] - 1e-12
    # non-re-anchor steps chain exactly
    for t in range(T - 1):
        if t not in reanchor_ts:
            assert np.array_equal(calls[t + 1]['x_in'], calls[t]['x_out'])
            assert calls[t + 1]['tau_in'] == calls[t]['tau_out']


def test_dro_solve_interval_zero_never_jumps(tmp_path, monkeypatch):
    """solve_interval=0 (legacy behavior): pure chained subgradient path."""
    calls = _spy_subgrad_step(monkeypatch)
    solves = _spy_direct_solve(monkeypatch)
    _run_dro(tmp_path, solve_interval=0, t_solve_list=(5,))

    assert solves == []                    # direct solver never invoked
    for t in range(T - 1):
        assert np.array_equal(calls[t + 1]['x_in'], calls[t]['x_out'])
        assert calls[t + 1]['tau_in'] == calls[t]['tau_out']


# --------------------------------------------------------------------------- #
# Driver smoke: both subgrad paths, re-anchoring on (2) and off (0)
# --------------------------------------------------------------------------- #
SMOKE = [
    ("mro-subgrad", ["--method", "mro", "--solver", "subgrad", "--K", "3"],
     "T5/df_K3R0.csv"),
    ("dro-subgrad", ["--method", "dro", "--solver", "subgrad"],
     "T5/df_K0R0.csv"),
]


@pytest.mark.parametrize("solve_interval", ["2", "0"])
@pytest.mark.parametrize("pid,args,final", SMOKE, ids=[s[0] for s in SMOKE])
def test_smoke_driver(tmp_path, pid, args, final, solve_interval):
    results_dir = tmp_path / "results"
    proc = run_cli(
        ["-m", "portfolio.run", "--results_dir", str(results_dir), *args,
         "--T", "6", "--R", "1", "--m", "8", "--N_init", "3",
         "--interval", "1", "--eps_index", "0",
         "--solve_interval", solve_interval],
        cwd=PAPER_DIR)
    assert_ran(proc, label=f"{pid} solve_interval={solve_interval}")

    csv_path = results_dir / final
    assert csv_path.exists(), f"final CSV missing: {csv_path}"
    df = pd.read_csv(csv_path, index_col=0)
    assert len(df) > 0
    num = df.select_dtypes(include=[np.number])
    nan_cols = [c for c in num.columns if num[c].isna().any()]
    assert not nan_cols, f"NaNs in final CSV columns: {nan_cols}"
