# Profiling and SLURM sizing (2026-08-08)

Single-task profiles (`profiling/profile_task.py`, one `(seed 0, eps_index 0)`
task, threads pinned to 1) on an Apple M4 Pro, calibrated to cluster `cpu`
nodes with pilot jobs.

## Measurements

Mac sweep, wall time per single task (quadratic fit to T=2001):

| path | T=51 | T=101 | T=201 | T=401 | fit @ T=2001 |
|---|---|---|---|---|---|
| port mro subgrad | 24.6s | 62.0s | 211.4s | 824.1s | 5.74 h |
| port dro subgrad | 14.1s | 40.1s | 131.8s | 475.7s | 3.03 h |
| port mro exact   | 9.2s  | 11.3s | 17.4s  | 38.9s  | 0.18 h |
| port dro exact   | 4.0s  | 6.1s  | 13.7s  | 46.0s  | 0.31 h |
| svm mro subgrad  | 3.5s  | 5.2s  | 15.0s  | 62.6s  | 0.49 h |
| svm mro exact    | ~3.3s flat | | | | ~0.01 h |
| svm dro subgrad  | ~3.2s flat | | | | ~0.00 h |
| svm dro exact    | 4.3s  | 5.0s  | 6.1s   | 9.5s   | 0.02 h |

cluster calibration pilots (1 cpu, single task):
- port mro subgrad T=401: **2785 s vs 824 s Mac → della/Mac ratio 3.4**;
  MaxRSS 323 MB.
- svm dro exact T=2001: 3.6 min, MaxRSS 279 MB (production path is trivial).

## Hotspots (cProfile, inline single task)

- **port mro subgrad**: 99% of wall time is `worst_case_value_box`
  (`utils.py:189`) → `_worstcase_piece2` (`utils.py:148`), the 60x80 nested
  bisection, called ~8x per timestep (subgradient steps for online + MRO,
  plus 4 diagnostic/regret evaluations on the full running sample at
  interval=1). All calls feed plotted series (regret uses prev-iterate values
  re-evaluated on the *current* sample set, so they cannot be cached). Cost
  is inherent; sizing accommodates it, so no cadence change was made.
- **svm mro subgrad**: the per-step cluster-SAA MOSEK MILP (K=15 rows) is
  subsecond; total path cost 0.49 h/task on Mac. No cadence change needed.
- The legacy per-step full-DataFrame checkpoint rewrite (O(T^2) at
  interval=1) is gated to every 200 logged steps in the consolidated
  drivers (final CSVs bit-identical; verified by the equivalence suite).

## SLURM sizing

Formula: `walltime = ceil(tasks/cpus) x task_time(della) x 1.3-1.5`,
`mem-per-cpu = peak worker RSS x ~3`. Tasks = seeds x epsilons.
The two heavy portfolio subgrad paths are split into seed-halves
(5 seeds x 6 eps = 30 tasks <= 32 cpus → one joblib batch per job).

Final table: see `slurm/submit_all.sh`. Headline requests:

| job | tasks | cpus | mem/cpu | time | est. run |
|---|---|---|---|---|---|
| port_mro_subgrad_s{0,5} | 30 each | 32 | 1G | 30h | ~19.5h |
| port_dro_subgrad_s{0,5} | 20 each | 32 | 1G | 16h | ~10.5h |
| port_mro_exact | 60 | 32 | 3G | 3h | ~1.2h |
| port_dro_exact | 70 | 20 | 8G | 5h | ~4h (mem-bound; RSS grows with N) |
| port_true_saa | 10 seeds serial | 4 | 10G | 12h | conservative (unprofiled) |
| svm_mro_subgrad | 60 | 32 | 1G | 6h | ~4.3h |
| svm_{mro,dro}_exact, svm_dro_subgrad | 60 | 32 | 1G | 1-2h | minutes |
| svm_true_saa | 10 seeds serial | 4 | 8G | 8h | conservative (1500s MILP cap x 10) |


## Update: accelerated implementations (2026-08-08)

Two changes removed the dominant costs (both validated against the untouched
Python/CVXPY oracles; the legacy-equivalence gate pins them off):

1. **C worst-case kernel** (`portfolio/cworst/`): the nested-bisection
   worst-case evaluation, previously numpy memory-bandwidth-bound
   (~47 GB/s sustained, ~120 GB traffic per call at N=2000), is now a C
   kernel that precomputes per-sample saturation breakpoints once per call
   and solves the same fixed point exactly per outer iteration.
   ~240x on value calls, ~110x on state-producing calls
   (2-79 ms vs 5.4-8.0 s at N=2000 (workstation/cluster)). Parity: 1e-15 (literal
   bisection mode) / 2e-10 (exact mode) on F; SOCP oracle within 1e-6.
2. **Direct Clarabel SOCP** (`portfolio/direct_socp.py`): dro exact now
   assembles Clarabel's native cone form directly (0.03 s assembly at
   N=2000) instead of CVXPY canonicalization (102 s): 2.5x per solve.

Re-profiled single-task walls (Mac, fit @ T=2001): mro subgrad 5.74 h -> ~50 s;
dro subgrad 3.03 h -> ~45 s; mro exact 0.18 h -> ~40 s; dro exact 0.31 h
(9.85 GB) -> ~10 min (<2 GB projected). Final production requests: portfolio
subgrad/exact jobs 2-3 h; dro exact seed-split 8 h at 14 cpu x 12G.

## Update: periodic re-anchoring (final)

The portfolio subgradient variants re-anchor with an exact solve every
`solve_interval` = 200 steps (plus early steps [5, 25, 50, 100]); the solve
time is charged into the recorded per-iteration time. Measured cost per
re-anchor at production sizes (m = 300): clustered problem (K = 15,
Clarabel via CVXPY) ~0.13 s; full-sample problem (direct Clarabel,
N = 2000) ~115 s, growing with N. At T = 5001 this adds minutes per task to
the clustered variants and ~1-2 h per task to the full-sample subgradient
variant (run as a one-seed-per-element array, 4 cpus x 10G, 12 h).
