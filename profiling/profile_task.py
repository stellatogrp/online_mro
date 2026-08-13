"""Profile a single (seed, epsilon) task of a consolidated driver.

Measures wall time and peak RSS of one task at several horizons T, fits a
quadratic cost model, and extrapolates to the production horizon to size
SLURM requests (formula from the parent repo's PROFILING.md:
walltime = ceil(tasks/cpus) * max_task_time * 1.3, mem = peak_RSS * 1.3).

Usage (from paper_experiments/):
  uv run python -m profiling.profile_task --experiment portfolio --method mro \
      --solver subgrad --horizons 51 101 201 401 [--eps_index 0] [--cprofile]

With --cprofile it additionally records a .pstats dump and prints the top
cumulative-time functions at the largest horizon.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = HERE / "results"


def run_one(experiment, method, solver, T, eps_index, extra, cprofile):
    """Run one single-task driver invocation; return wall time and peak RSS."""
    workdir = RESULTS / "work" / f"{experiment}_{method}_{solver}_T{T}"
    tag = f"{experiment}_{method}_{solver}_T{T}"
    cmd = [sys.executable]
    if cprofile:
        cmd += ["-m", "cProfile", "-o", str(RESULTS / f"{tag}.pstats")]
    cmd += ["-m", f"{experiment}.run",
            "--method", method, "--solver", solver,
            "--results_dir", str(workdir),
            "--T", str(T), "--R", "1"]
    if eps_index is not None and method != "true_saa":
        cmd += ["--eps_index", str(eps_index)]
    cmd += extra

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1",
               # run the single task inline (joblib n_jobs=1) so cProfile and
               # RSS see the actual work, not a loky worker
               SLURM_CPUS_PER_TASK="1")

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=PROJECT, env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    peak = {"rss": 0}
    stop = threading.Event()

    def poll():
        ps = psutil.Process(proc.pid)
        while not stop.is_set():
            try:
                rss = ps.memory_info().rss
                for ch in ps.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak["rss"] = max(peak["rss"], rss)
            except psutil.NoSuchProcess:
                return
            time.sleep(0.05)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    _, err = proc.communicate()
    stop.set()
    th.join(timeout=1)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} failed (rc={proc.returncode}):\n"
                           + err.decode()[-2000:])
    return wall, peak["rss"]


def fit_and_extrapolate(horizons, walls, T_prod):
    """Fit wall(T) = a + b*T + c*T^2 and evaluate at T_prod."""
    coef = np.polyfit(np.array(horizons, dtype=float), np.array(walls), 2)
    return float(np.polyval(coef, T_prod)), coef


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   choices=["portfolio", "svm"])
    p.add_argument("--method", required=True,
                   choices=["mro", "dro", "true_saa"])
    p.add_argument("--solver", default="subgrad",
                   choices=["exact", "subgrad"])
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[51, 101, 201, 401])
    p.add_argument("--eps_index", type=int, default=0,
                   help="which epsilon to profile (0 = largest)")
    p.add_argument("--T_prod", type=int, default=2001)
    p.add_argument("--cprofile", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra args passed through to the driver")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for T in args.horizons:
        cp = args.cprofile and T == max(args.horizons)
        wall, rss = run_one(args.experiment, args.method, args.solver, T,
                            args.eps_index, args.extra, cp)
        rows.append({"T": T, "wall_s": round(wall, 2),
                     "peak_rss_mb": round(rss / 2**20, 1)})
        print(f"T={T:5d}  wall={wall:8.1f}s  peakRSS={rss/2**20:8.1f}MB",
              flush=True)

    walls = [r["wall_s"] for r in rows]
    proj, coef = fit_and_extrapolate(args.horizons, walls, args.T_prod)
    out = {
        "experiment": args.experiment, "method": args.method,
        "solver": args.solver, "eps_index": args.eps_index,
        "rows": rows, "quadratic_coef": list(coef),
        "projected_wall_s_at_T_prod": round(proj, 1),
        "T_prod": args.T_prod,
        "peak_rss_mb_max": max(r["peak_rss_mb"] for r in rows),
    }
    tag = f"{args.experiment}_{args.method}_{args.solver}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nprojected single-task wall @ T={args.T_prod}: {proj/3600:.2f} h"
          f"  (fit {coef})")

    if args.cprofile:
        import pstats
        st = pstats.Stats(
            str(RESULTS / f"{tag}_T{max(args.horizons)}.pstats"))
        st.sort_stats("cumulative")
        st.print_stats(25)


if __name__ == "__main__":
    main()
