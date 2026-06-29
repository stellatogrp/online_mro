#!/usr/bin/env python
"""Solve-cost & memory vs problem size N for the two full-DRO formulations.

The full-DRO per-task wall time is dominated by the solves at large N (the
`t_list` checkpoints push N up to ~2000). Rather than run the whole 2001-step
loop, we build and solve the *actual* problems at a sweep of N values, measuring
canonicalization (compile) vs solver time and peak RSS. We then extrapolate the
per-task budget by summing over the real solve schedule.

Variants:
  reg_dro  -> createproblem_hingeMIO_kappa(N, m, k) + create_scenario_hinge  (MOSEK MI-SOCP)
  port_dro -> createproblem_portLP_p2(N, m) + worst_case_p2(N, m, data)       (CLARABEL)
"""
import argparse
import os
import sys
import time
import json
import threading

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


class PeakRSS(threading.Thread):
    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._flag = threading.Event()
        self._p = psutil.Process(os.getpid())

    def run(self):
        while not self._flag.is_set():
            try:
                self.peak = max(self.peak, self._p.memory_info().rss)
            except psutil.Error:
                pass
            self._flag.wait(self.interval)

    def stop(self):
        self._flag.set()


def timed_solve(prob, **kw):
    import cvxpy as cp  # noqa
    rss = PeakRSS()
    base = psutil.Process(os.getpid()).memory_info().rss
    rss.start()
    t0 = time.perf_counter()
    try:
        prob.solve(**kw)
        status = prob.status
    except Exception as e:
        status = f"ERROR:{repr(e)[:80]}"
    wall = time.perf_counter() - t0
    rss.stop(); rss.join(timeout=1)
    st = getattr(prob, "solver_stats", None)
    return dict(
        wall=round(wall, 3),
        solve_time=round(getattr(st, "solve_time", 0.0) or 0.0, 3) if st else None,
        compile=round(getattr(prob, "compilation_time", 0.0) or 0.0, 3),
        status=str(status),
        peak_rss_gb=round(rss.peak / 1e9, 3),
        rss_delta_gb=round((rss.peak - base) / 1e9, 3),
    )


def study_reg(Ns, study_cap):
    sys.path.insert(0, os.path.join(REPO, "regression"))
    import cvxpy as cp
    from utils_p1 import (
        createproblem_hingeMIO_kappa, create_scenario_hinge,
        generate_classification_data, MOSEK_PARAMS,
    )
    m, k, kappa = 50, 5, 10.0
    params = dict(MOSEK_PARAMS)
    params["MSK_DPAR_MIO_MAX_TIME"] = float(study_cap)
    params["MSK_DPAR_OPTIMIZER_MAX_TIME"] = float(study_cap)
    data, _ = generate_classification_data(n_total=20000, m=m, k_true=k,
                                           noise_std=3.0, seed=12345)
    rows = []
    for N in Ns:
        samples = data[:N, :m + 1]
        w = np.ones(N) / N
        # full-DRO MI-SOCP
        prob, beta, z, lmbda, s, dat, eps, kap, wp = createproblem_hingeMIO_kappa(N, m, k, p=2)
        dat.value = samples; wp.value = w; eps.value = 0.5 / (N ** 0.25); kap.value = kappa
        r_dro = timed_solve(prob, solver=cp.MOSEK, ignore_dpp=True, verbose=False, mosek_params=params)
        # SAA scenario MI
        sp, sx, sz = create_scenario_hinge(samples, m, N, k)
        r_saa = timed_solve(sp, solver=cp.MOSEK, ignore_dpp=True, verbose=False, mosek_params=params)
        rows.append(dict(N=N, dro=r_dro, saa=r_saa))
        print(f"N={N:5d} | DRO {r_dro['wall']:7.2f}s ({r_dro['status']}) "
              f"compile={r_dro['compile']:.2f} rss={r_dro['rss_delta_gb']:.3f}GB | "
              f"SAA {r_saa['wall']:7.2f}s ({r_saa['status']})", flush=True)
    return rows


def study_port(Ns, study_cap):
    sys.path.insert(0, os.path.join(REPO, "port_new"))
    import cvxpy as cp
    import pandas as pd
    from utils import createproblem_portLP_p2, worst_case_p2
    m = 150
    datname = os.path.join(REPO, "port_new", "synthetic_200_1.csv")
    full = pd.read_csv(datname).to_numpy()[:, 1:][:, :m]
    rows = []
    for N in Ns:
        samples = full[:N]
        w = np.ones(N) / N
        # outer LP (rebuilt each interval in the driver)
        prob, x, s, tau, lam, dat, eps, wp = createproblem_portLP_p2(N, m)
        dat.value = samples; wp.value = w; eps.value = 0.05
        r_lp = timed_solve(prob, solver=cp.CLARABEL, verbose=False)
        # worst-case dual (z in R^{N x m}) -- the heavy one
        wc = worst_case_p2(N, m, samples)
        prob2 = wc[0]
        # set the parameter iterate(s); worst_case_p2 returns (prob, s,lam,x,tau,eps,w)
        try:
            _, s_d, lam_d, x_d, tau_d, eps_d, w_d = wc
            x_d.value = np.ones(m) / m
            tau_d.value = 0.0
            eps_d.value = 0.05
            w_d.value = w
        except Exception:
            pass
        r_wc = timed_solve(prob2, solver=cp.CLARABEL, verbose=False)
        rows.append(dict(N=N, lp=r_lp, worstcase=r_wc))
        print(f"N={N:5d} | LP {r_lp['wall']:7.2f}s ({r_lp['status']}) compile={r_lp['compile']:.2f} | "
              f"WC {r_wc['wall']:7.2f}s ({r_wc['status']}) compile={r_wc['compile']:.2f} "
              f"rss={r_wc['rss_delta_gb']:.3f}GB", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["reg_dro", "port_dro"])
    ap.add_argument("--Ns", default="250,500,1000,1500,2000")
    ap.add_argument("--study_cap", type=float, default=300.0,
                    help="per-solve MOSEK time cap for the study (prod uses 1500)")
    args = ap.parse_args()
    Ns = [int(x) for x in args.Ns.split(",")]
    rows = study_reg(Ns, args.study_cap) if args.variant == "reg_dro" else study_port(Ns, args.study_cap)
    out = os.path.join(HERE, "results", f"scaling_{args.variant}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
