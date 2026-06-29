#!/usr/bin/env python
"""Profile one online-MRO experiment variant as a *single* (seed, eps) task.

This drives the REAL driver scripts (no copy of their logic) so the profile
reflects the exact code paths that run under SLURM. We make three interventions,
all before the driver executes:

1. ``joblib.Parallel`` is replaced by a shim that runs only the first
   ``--tasks`` items of the sweep, in-process and sequentially. The drivers all
   do ``Parallel(n_jobs=...)(delayed(fn)(...) for ...)``; running in-process lets
   cProfile and the RSS sampler see everything regardless of the driver's
   internal globals.
2. ``cvxpy.Problem.solve`` is wrapped to record, per solve, the wall time, the
   solver's own ``solve_time``, the CVXPY ``compilation_time`` (canonicalization),
   and the problem size. ``wall - solve_time - compile`` is CVXPY/Python overhead.
3. A psutil daemon thread samples RSS to get the peak resident memory.

Usage:
    python profiling/profile_run.py --variant reg_dro --T 301 --tasks 1
"""
import argparse
import os
import sys
import time
import json
import runpy
import threading
import cProfile
import pstats
from io import StringIO

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# (script path, default CLI args mirroring the slurm command, array idx or None)
VARIANTS = {
    # portfolio, our clustered MRO method (closed-form subgradient, CLARABEL)
    "port_ours": dict(
        script="port_new/port_p2.py",
        args="--R 1 --T {T} --fixed_time {T} --interval 1 --Q 500 --K 15 "
             "--N_init 5 --r_start 0 --m 50 --eta_0 0.1 --rmse_mult 1.25 "
             "--no-line_search --cluster_interval 1",
        array_idx=0,  # K_arr=[15,25,50,2000] -> K=15
    ),
    # portfolio, full DRO (rebuilds LP each interval, CLARABEL)
    "port_dro": dict(
        script="port_new/port_DRO_orig_p2.py",
        args="--R 1 --T {T} --interval 100 --interval_SAA 100 --N_init 5 "
             "--r_start 0 --m 150",
        array_idx=None,
    ),
    # regression, our clustered MRO method (3 MI-SOCPs/interval on N=K, MOSEK)
    "reg_ours": dict(
        script="regression/reg_orig_p2.py",
        args="--R 1 --T {T} --fixed_time {T} --interval 100 --Q 500 --K 10 "
             "--N_init 5 --r_start 0 --m 50 --noise 3 --rmse_mult 1.1 --k 5 "
             "--k_true 5 --power 0.033 --p 2 --kappa 10",
        array_idx=0,  # K_arr=[10,15,25] -> K=10
    ),
    # regression, full DRO (2 MI-SOCPs/interval on N=num_dat, MOSEK)
    "reg_dro": dict(
        script="regression/reg_DRO_orig_p2.py",
        args="--R 1 --T {T} --interval 100 --interval_SAA 100 --N_init 5 "
             "--r_start 0 --m 50 --noise 3 --k 5 --k_true 5 --p 2 "
             "--power 0.25 --kappa 10",
        array_idx=None,
    ),
}

# --------------------------------------------------------------------------- #
# Intervention 1: limit the joblib sweep to the first N tasks, in-process.
# --------------------------------------------------------------------------- #
TASK_LIMIT = [1]


def _install_parallel_shim():
    import joblib

    class LimitedParallel:
        def __init__(self, *a, **k):
            pass

        def __call__(self, iterable):
            out = []
            for i, task in enumerate(iterable):
                if i >= TASK_LIMIT[0]:
                    break
                func, args, kwargs = task  # delayed(fn)(*a, **k) -> (fn, a, k)
                out.append(func(*args, **kwargs))
            return out

    joblib.Parallel = LimitedParallel


# --------------------------------------------------------------------------- #
# Intervention 2: split each CVXPY solve into compile vs solve vs overhead.
# --------------------------------------------------------------------------- #
SOLVE_RECORDS = []


def _install_solve_probe():
    import cvxpy as cp

    orig = cp.Problem.solve

    def probed(self, *a, **k):
        t0 = time.perf_counter()
        res = orig(self, *a, **k)
        wall = time.perf_counter() - t0
        st = getattr(self, "solver_stats", None)
        solve_t = getattr(st, "solve_time", None) if st is not None else None
        compile_t = getattr(self, "compilation_time", None)
        try:
            nvars = sum(v.size for v in self.variables())
            nbool = sum(v.size for v in self.variables() if v.attributes.get("boolean"))
        except Exception:
            nvars, nbool = None, None
        SOLVE_RECORDS.append(dict(
            wall=wall, solve_time=solve_t, compile=compile_t,
            nvars=nvars, nbool=nbool, ncon=len(self.constraints),
            solver=k.get("solver", None) and str(k.get("solver")),
        ))
        return res

    cp.Problem.solve = probed


# --------------------------------------------------------------------------- #
# Intervention 3: peak-RSS sampler.
# --------------------------------------------------------------------------- #
class RSSSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stopflag = threading.Event()
        self._proc = psutil.Process(os.getpid())

    def run(self):
        while not self._stopflag.is_set():
            try:
                rss = self._proc.memory_info().rss
                # include children just in case joblib spawns any
                for c in self._proc.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak = max(self.peak, rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stopflag.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--T", type=int, default=301)
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    TASK_LIMIT[0] = args.tasks
    spec = VARIANTS[args.variant]
    os.makedirs(args.outdir, exist_ok=True)

    # Output the experiment writes to (kept out of the repo tree).
    work = os.path.join(args.outdir, "work", args.variant) + "/"
    os.makedirs(work, exist_ok=True)

    if spec["array_idx"] is not None:
        os.environ["SLURM_ARRAY_TASK_ID"] = str(spec["array_idx"])
    # Force single-process accounting inside get_n_processes(); we run in-proc.
    os.environ["SLURM_CPUS_PER_TASK"] = "1"

    script = os.path.join(REPO, spec["script"])
    cli = spec["args"].format(T=args.T)
    sys.argv = [script, "--foldername", work] + cli.split()

    _install_parallel_shim()
    _install_solve_probe()

    sampler = RSSSampler()
    sampler.start()

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    post_error = None
    prof.enable()
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        # Expected: the driver's __main__ post-processing reindexes the full
        # R*M sweep; with --tasks limited it raises IndexError after the timed
        # experiment work is already done. Record it but keep the results.
        post_error = repr(e)
    finally:
        prof.disable()
        wall = time.perf_counter() - t0
        sampler.stop()
        sampler.join(timeout=1.0)

    # ---- write cProfile dump + top functions ----
    base = os.path.join(args.outdir, args.variant)
    prof.dump_stats(base + ".pstats")
    s = StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    s2 = StringIO()
    pstats.Stats(prof, stream=s2).sort_stats("tottime").print_stats(35)

    # ---- summarize the solve records ----
    def _sum(key):
        vals = [r[key] for r in SOLVE_RECORDS if r.get(key) is not None]
        return sum(vals) if vals else 0.0

    n_solves = len(SOLVE_RECORDS)
    tot_solve = _sum("solve_time")
    tot_comp = _sum("compile")
    tot_wall_solve = _sum("wall")
    summary = dict(
        variant=args.variant, T=args.T, tasks=args.tasks,
        total_wall_s=round(wall, 2),
        peak_rss_gb=round(sampler.peak / 1e9, 3),
        n_cvxpy_solves=n_solves,
        cvxpy_total_wall_s=round(tot_wall_solve, 2),
        cvxpy_solve_time_s=round(tot_solve, 2),
        cvxpy_compile_time_s=round(tot_comp, 2),
        cvxpy_overhead_s=round(tot_wall_solve - tot_solve - tot_comp, 2),
        non_cvxpy_wall_s=round(wall - tot_wall_solve, 2),
        post_processing_note=post_error,
    )
    import numpy as _np

    def _native(o):
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        return str(o)

    with open(base + ".summary.json", "w") as f:
        json.dump(dict(summary=summary, solves=SOLVE_RECORDS), f, indent=2,
                  default=_native)

    with open(base + ".profile.txt", "w") as f:
        f.write("=== SUMMARY ===\n")
        f.write(json.dumps(summary, indent=2) + "\n\n")
        f.write("=== TOP BY CUMULATIVE TIME ===\n")
        f.write(s.getvalue() + "\n")
        f.write("=== TOP BY SELF (tottime) ===\n")
        f.write(s2.getvalue() + "\n")

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
    print("wrote:", base + ".profile.txt")


if __name__ == "__main__":
    main()
