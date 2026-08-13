"""Direct Clarabel assembly of the box-support W1-DRO CVaR SOCP and SAA LP.

This is a canonicalization-free replacement for the hot-loop use of
``utils.createproblem_box_DRO`` (CVXPY) in ``methods.task_dro_exact``.  At
production size (N=2000, m=300) CVXPY canonicalization takes ~100 s per solve
and grows ~O(N^2), while the Clarabel solve itself takes ~50 s.  Here the cone
program is assembled directly with scipy.sparse (a few hundred ms) and handed
to ``clarabel.DefaultSolver``.  CVXPY remains the verification oracle (see
``tests/test_direct_socp.py``).

``SaaCvarLp`` is the same treatment for the SAA branch of ``task_dro_exact``
(``utils.create_scenario``, the eps -> 0 limit of the DRO problem): a plain
piecewise-linear CVaR LP whose CVXPY canonicalization also grows ~O(N^2) with
the sample count.

Problem (Esfahani-Kuhn dual of the box-support W1 DRO CVaR, identical to
``createproblem_box_DRO``):

    min   tau + eps * lam + w^T s
    s.t.  sum(x) = 1,  0 <= x <= 1
          s >= 0
          s_i >= a*tau + a*<xi_i, x> + gpos_i^T (ub - xi_i) + gneg_i^T (xi_i - lb)
          gpos, gneg >= 0,  lam >= 0
          || a*x - (gpos_i - gneg_i) ||_2 <= lam        for i = 1..N

Clarabel native form:  min q^T z  s.t.  A z + s = b,  s in K, with the variable
stacking  z = [x (m), tau, lam, s (N), gpos (N*m, row-major), gneg (N*m)]
and the cone order  K = Zero(1) x Nonneg(2m + 1 + 2N + 2Nm) x SOC(m+1)^N:

    Zero(1):     sum(x) = 1
    Nonneg:      -x <= 0;  x <= 1;  -lam <= 0;  -s <= 0;
                 a*tau + a*(dat @ x) - s + box_term <= 0  (N epigraph rows);
                 -gpos <= 0;  -gneg <= 0
    SOC_i(m+1):  (lam, a*x - gpos_i + gneg_i) in K_soc
                 (rows: A z = -s, i.e. A[., lam] = -1 on the head row and
                 A[., x_j] = -a, A[., gpos_ij] = +1, A[., gneg_ij] = -1)

Settings replicate what CVXPY passes to Clarabel: ``DefaultSettings()`` with
only ``verbose`` overridden (CVXPY's Clarabel interface forwards no other
options from ``safe_solve``; in particular ``time_limit`` stays ``inf`` and
tolerances stay at the Clarabel defaults).  A ``time_limit`` override is
still accepted for callers that want one.
"""

import numpy as np
import scipy.sparse as sp

try:
    import clarabel
except ImportError:  # pragma: no cover - exercised only without clarabel
    clarabel = None

# Clarabel -> CVXPY-style status strings (same map cvxpy's interface uses).
_STATUS_MAP = {
    'Solved': 'optimal',
    'PrimalInfeasible': 'infeasible',
    'DualInfeasible': 'unbounded',
    'AlmostSolved': 'optimal_inaccurate',
    'AlmostPrimalInfeasible': 'infeasible_inaccurate',
    'AlmostDualInfeasible': 'unbounded_inaccurate',
    'MaxIterations': 'user_limit',
    'MaxTime': 'user_limit',
    'NumericalError': 'solver_error',
    'InsufficientProgress': 'solver_error',
}


class BoxDROSocp:
    """Box-support W1-DRO CVaR SOCP with direct (CVXPY-free) Clarabel calls.

    Parameters fixed at construction: ``m`` assets, box ``[lb, ub]`` and the
    CVaR slope ``a`` (baked in as constants, like ``createproblem_box_DRO``).
    ``solve(dat, w, eps)`` accepts a varying number of samples N call to call;
    the sparse matrices are assembled fresh each call (vectorized COO
    assembly, cheap relative to the conic solve).
    """

    def __init__(self, m, lb, ub, a=-5.0):
        self.m = int(m)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.a = float(a)
        if self.lb.shape != (self.m,) or self.ub.shape != (self.m,):
            raise ValueError('lb/ub must have shape (m,)')

    # ------------------------------------------------------------------ #
    def _assemble(self, dat, w, eps):
        """Build (P, q, A, b, cones) in Clarabel native form."""
        m, a = self.m, self.a
        lb, ub = self.lb, self.ub
        dat = np.ascontiguousarray(dat, dtype=float)
        N = dat.shape[0]
        if dat.shape != (N, m):
            raise ValueError('dat must have shape (N, m)')
        w = np.asarray(w, dtype=float)

        # variable stacking: z = [x, tau, lam, s, gpos (row-major), gneg]
        i_tau = m
        i_lam = m + 1
        i_s = m + 2                     # s occupies [i_s, i_s + N)
        i_gp = i_s + N                  # gpos occupies [i_gp, i_gp + N*m)
        i_gn = i_gp + N * m             # gneg occupies [i_gn, i_gn + N*m)
        n = i_gn + N * m

        q = np.zeros(n)
        q[i_tau] = 1.0
        q[i_lam] = float(eps)
        q[i_s:i_s + N] = w

        Nm = N * m
        ar_m = np.arange(m)
        ar_N = np.arange(N)
        ar_Nm = np.arange(Nm)

        # ---- row layout ------------------------------------------------ #
        # zero cone: 1 row.  nonneg cone rows (in order):
        r_xlo = 1                       # -x <= 0            (m rows)
        r_xhi = r_xlo + m               # x <= 1             (m rows)
        r_lam = r_xhi + m               # -lam <= 0          (1 row)
        r_s = r_lam + 1                 # -s <= 0            (N rows)
        r_epi = r_s + N                 # epigraph rows      (N rows)
        r_gp = r_epi + N                # -gpos <= 0         (N*m rows)
        r_gn = r_gp + Nm                # -gneg <= 0         (N*m rows)
        r_soc = r_gn + Nm               # SOC block          (N*(m+1) rows)
        n_rows = r_soc + N * (m + 1)
        n_nonneg = r_soc - 1

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.asarray(r, dtype=np.int64).ravel())
            cols.append(np.asarray(c, dtype=np.int64).ravel())
            vals.append(np.asarray(v, dtype=float).ravel())

        # zero cone: sum(x) = 1
        add(np.zeros(m), ar_m, np.ones(m))
        # -x <= 0 and x <= 1
        add(r_xlo + ar_m, ar_m, -np.ones(m))
        add(r_xhi + ar_m, ar_m, np.ones(m))
        # -lam <= 0
        add([r_lam], [i_lam], [-1.0])
        # -s <= 0
        add(r_s + ar_N, i_s + ar_N, -np.ones(N))
        # epigraph rows: a*tau + a*dat_i.x - s_i + (ub-dat_i).gpos_i
        #                + (dat_i-lb).gneg_i <= 0
        epi_rows = r_epi + ar_N
        add(epi_rows, np.full(N, i_tau), np.full(N, a))
        add(np.repeat(epi_rows, m), np.tile(ar_m, N), a * dat.ravel())
        add(epi_rows, i_s + ar_N, -np.ones(N))
        add(np.repeat(epi_rows, m), i_gp + ar_Nm, (ub[None, :] - dat).ravel())
        add(np.repeat(epi_rows, m), i_gn + ar_Nm, (dat - lb[None, :]).ravel())
        # -gpos <= 0, -gneg <= 0
        add(r_gp + ar_Nm, i_gp + ar_Nm, -np.ones(Nm))
        add(r_gn + ar_Nm, i_gn + ar_Nm, -np.ones(Nm))
        # SOC blocks: cone i at rows r_soc + i*(m+1) .. r_soc + (i+1)*(m+1)-1
        #   head: s0 = lam            -> A[., lam] = -1
        #   tail: s_j = a*x_j - gpos_ij + gneg_ij
        #         -> A[., x_j] = -a, A[., gpos_ij] = +1, A[., gneg_ij] = -1
        head_rows = r_soc + ar_N * (m + 1)
        add(head_rows, np.full(N, i_lam), -np.ones(N))
        tail_rows = (head_rows[:, None] + 1 + ar_m[None, :]).ravel()
        add(tail_rows, np.tile(ar_m, N), np.full(Nm, -a))
        add(tail_rows, i_gp + ar_Nm, np.ones(Nm))
        add(tail_rows, i_gn + ar_Nm, -np.ones(Nm))

        A = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_rows, n),
        )
        b = np.zeros(n_rows)
        b[0] = 1.0                      # sum(x) = 1
        b[r_xhi:r_xhi + m] = 1.0        # x <= 1

        cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n_nonneg)]
        cones += [clarabel.SecondOrderConeT(m + 1) for _ in range(N)]

        P = sp.csc_matrix((n, n))
        return P, q, A, b, cones

    # ------------------------------------------------------------------ #
    def solve(self, dat, w, eps, verbose=False, time_limit=None):
        """Solve for data ``dat`` (N, m), weights ``w`` (N,), radius ``eps``.

        Returns ``(obj_value, x_star, tau_star, solve_time, status)`` with
        ``status`` in CVXPY vocabulary.  On any failure (exception, or status
        not optimal/optimal_inaccurate) returns the NaN sentinels
        ``(nan, None, nan, nan, status)`` -- the same convention as
        ``safe_solve`` + the caller-side NaN branch in ``task_dro_exact``.
        """
        if clarabel is None:
            raise ImportError('the clarabel package is required for BoxDROSocp')
        try:
            P, q, A, b, cones = self._assemble(dat, w, eps)
            settings = clarabel.DefaultSettings()
            settings.verbose = bool(verbose)
            if time_limit is not None:
                settings.time_limit = float(time_limit)
            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
            sol = solver.solve()
        except Exception as e:  # mirror safe_solve's exception behavior
            print(f"[BoxDROSocp] raised: {type(e).__name__}: {e}", flush=True)
            return np.nan, None, np.nan, np.nan, 'solver_error'
        status = _STATUS_MAP.get(str(sol.status), 'unknown')
        if status not in ('optimal', 'optimal_inaccurate'):
            return np.nan, None, np.nan, np.nan, status
        z = np.asarray(sol.x)
        m = self.m
        return (float(sol.obj_val), z[:m].copy(), float(z[m]),
                float(sol.solve_time), status)


class SaaCvarLp:
    """Weighted SAA CVaR LP with direct (CVXPY-free) Clarabel calls.

    Same problem as ``utils.create_scenario`` (with ``w = 1/N``) and
    ``utils.create_scenario_cluster`` (general weights):

        min_{x, tau}  sum(w)*tau + sum_i w_i * max(0, a*tau + a*<xi_i, x>)
        s.t.          sum(x) = 1,  0 <= x <= 1

    i.e. the eps -> 0 limit of the box-DRO problem.  Epigraph form with
    u_i >= max(0, a*tau + a*<xi_i, x>) is a pure LP; the Clarabel cone
    structure is Zero(1) x Nonneg(2m + 2N) -- no SOC blocks.

    Variable stacking:  z = [x (m), tau, u (N)].  Rows:

        Zero(1):   sum(x) = 1
        Nonneg:    -x <= 0;  x <= 1;  -u <= 0;
                   a*tau + a*(dat @ x) - u <= 0   (N epigraph rows)

    Interface conventions match ``BoxDROSocp``: ``solve(dat, w)`` returns
    ``(obj_value, x_star, tau_star, solve_time, status)`` with NaN sentinels
    on failure, deterministic output, and Clarabel ``DefaultSettings`` with
    only ``verbose`` (and an optional ``time_limit``) overridden.
    """

    def __init__(self, m, a=-5.0):
        self.m = int(m)
        self.a = float(a)

    # ------------------------------------------------------------------ #
    def _assemble(self, dat, w):
        """Build (P, q, A, b, cones) in Clarabel native form."""
        m, a = self.m, self.a
        dat = np.ascontiguousarray(dat, dtype=float)
        N = dat.shape[0]
        if dat.shape != (N, m):
            raise ValueError('dat must have shape (N, m)')
        w = np.asarray(w, dtype=float)
        if w.shape != (N,):
            raise ValueError('w must have shape (N,)')

        # variable stacking: z = [x, tau, u]
        i_tau = m
        i_u = m + 1                     # u occupies [i_u, i_u + N)
        n = i_u + N

        q = np.zeros(n)
        q[i_tau] = w.sum()
        q[i_u:i_u + N] = w

        ar_m = np.arange(m)
        ar_N = np.arange(N)

        # ---- row layout ------------------------------------------------ #
        # zero cone: 1 row.  nonneg cone rows (in order):
        r_xlo = 1                       # -x <= 0            (m rows)
        r_xhi = r_xlo + m               # x <= 1             (m rows)
        r_u = r_xhi + m                 # -u <= 0            (N rows)
        r_epi = r_u + N                 # epigraph rows      (N rows)
        n_rows = r_epi + N

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.asarray(r, dtype=np.int64).ravel())
            cols.append(np.asarray(c, dtype=np.int64).ravel())
            vals.append(np.asarray(v, dtype=float).ravel())

        # zero cone: sum(x) = 1
        add(np.zeros(m), ar_m, np.ones(m))
        # -x <= 0 and x <= 1
        add(r_xlo + ar_m, ar_m, -np.ones(m))
        add(r_xhi + ar_m, ar_m, np.ones(m))
        # -u <= 0
        add(r_u + ar_N, i_u + ar_N, -np.ones(N))
        # epigraph rows: a*tau + a*dat_i.x - u_i <= 0
        epi_rows = r_epi + ar_N
        add(epi_rows, np.full(N, i_tau), np.full(N, a))
        add(np.repeat(epi_rows, m), np.tile(ar_m, N), a * dat.ravel())
        add(epi_rows, i_u + ar_N, -np.ones(N))

        A = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_rows, n),
        )
        b = np.zeros(n_rows)
        b[0] = 1.0                      # sum(x) = 1
        b[r_xhi:r_xhi + m] = 1.0        # x <= 1

        cones = [clarabel.ZeroConeT(1),
                 clarabel.NonnegativeConeT(n_rows - 1)]
        P = sp.csc_matrix((n, n))
        return P, q, A, b, cones

    # ------------------------------------------------------------------ #
    def solve(self, dat, w, verbose=False, time_limit=None):
        """Solve for data ``dat`` (N, m) and weights ``w`` (N,).

        Returns ``(obj_value, x_star, tau_star, solve_time, status)`` with
        ``status`` in CVXPY vocabulary and the same NaN-sentinel failure
        convention as ``BoxDROSocp.solve``.
        """
        if clarabel is None:
            raise ImportError('the clarabel package is required for SaaCvarLp')
        try:
            P, q, A, b, cones = self._assemble(dat, w)
            settings = clarabel.DefaultSettings()
            settings.verbose = bool(verbose)
            if time_limit is not None:
                settings.time_limit = float(time_limit)
            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
            sol = solver.solve()
        except Exception as e:  # mirror safe_solve's exception behavior
            print(f"[SaaCvarLp] raised: {type(e).__name__}: {e}", flush=True)
            return np.nan, None, np.nan, np.nan, 'solver_error'
        status = _STATUS_MAP.get(str(sol.status), 'unknown')
        if status not in ('optimal', 'optimal_inaccurate'):
            return np.nan, None, np.nan, np.nan, status
        z = np.asarray(sol.x)
        m = self.m
        return (float(sol.obj_val), z[:m].copy(), float(z[m]),
                float(sol.solve_time), status)
