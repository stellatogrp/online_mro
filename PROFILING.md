# Profiling report & SLURM sizing — online-MRO experiments

Branch: `profiling-slurm-sizing`. Profiled the four latest p2 variants
(DRO = full DRO; "ours" = clustered MRO method). All measurements are a **single
`(seed, eps)` task run in-process** with one solver thread, via
`profiling/profile_run.py` (cProfile + canonicalization-vs-solve split + peak-RSS
sampler) and `profiling/scaling_study.py` (solve cost & memory vs problem size N).

## TL;DR

| Variant | File | Dominant bottleneck | Fixable? |
|---|---|---|---|
| **port_ours** | `port_new/port_p2.py` | **`df.to_csv` every timestep on a frame with array-valued cells → `str(ndarray)`**; O(T²). ~76% of runtime at T=151, projected **~1.9 h/task** at T=2001. | **Yes — big win** |
| **port_dro** | `port_new/port_DRO_orig_p2.py` | CVXPY **canonicalization** of the rebuilt LP (compile > solve), grows with N. | Partly |
| **reg_ours** | `regression/reg_orig_p2.py` | **MOSEK mixed-integer solve** (50 binary feature-select vars), 2-3.5 s each, ~99% of runtime. Not clustering/canon. | Inherent (MI) |
| **reg_dro** | `regression/reg_DRO_orig_p2.py` | **MOSEK mixed-integer solve** on growing N, 4-35 s each, ~99% of runtime. Memory grows with N. | Inherent (MI) |

The current SLURM requests are **5–20× larger than needed** on memory and time,
which is the main reason jobs sit PENDING. Recommended requests in the last
section cut all four to ≤ ~2 GB/cpu and ≤ a few hours.

## Environment note (action needed)

- The committed `mosek/mosek.lic` is **expired** (17-mar-2026). The SLURM scripts
  point at `/scratch/.../low-rank-dro/mosek/mosek.lic`, which is valid to
  18-mar-2027, so cluster jobs still work — but the repo copy is stale and
  misleading. Profiling used the scratch license.
- The shell has `OMP_NUM_THREADS=invalid` set globally (libgomp warns and ignores
  it). Harmless here (we set it to 1) but worth cleaning up in your dotfiles.
- `lropt_rev` belongs to `iywang`; profiling used a fresh `uv` venv
  (`profiling/.venv`, Python 3.11, numpy 1.26 / cvxpy 1.5.4 to match the
  pre-numpy-2 behavior the code assumes — see the `w2_dist` note below).

## Measured per-task numbers (1 thread)

| Variant | T | total (s) | MOSEK/solver solve (s) | CVXPY compile (s) | non-CVXPY (s) | #solves |
|---|--:|--:|--:|--:|--:|--:|
| port_ours | 151 | 50.3 | 2.2 | 4.5 | **42.9** (CSV 38) | 906 |
| port_ours | 301 | 177.9 | ~6.5 | 7.8 | **163.6** (CSV 157) | 1806 |
| port_dro  | 201 | 5.4 | 1.6 | **2.8** | 0.9 | 26 |
| reg_ours  | 201 | 79.4 | **77.4** | 0.6 | 1.2 | 51 |
| reg_dro   | 201 | 97.7 | **96.7** | 0.4 | 0.6 | 34 |

## Per-variant findings

### port_ours (`port_p2.py`) — CSV writing dominates
- `df.to_csv(...)` is called **every active timestep** (`port_p2.py:476`); with
  `--interval 1` that is all 2001 steps. The frame is rewritten in full each time
  (O(T²) rows written total) and several columns hold **numpy arrays per cell**
  (`x`, `MRO_x`, `weights`, `MRO_weights`, `weights_q`). pandas stringifies each
  array cell (`pandas .../csvs.py:_save_chunk → numpy array2string → dragon4`).
- Confirmed quadratic: non-CVXPY time 42.9 s → 163.6 s for T 151 → 301 (×3.8 ≈
  (301/151)²). **Projection to T=2001: ~6900 s ≈ 1.9 h/task just for CSV.**
- The p2 "ours" method is **not** solve-free (unlike p1): it solves the
  `worst_case_p2` dual 4×/step for metrics + cluster warm-starts (906 solves at
  T=151), but these are cheap (CLARABEL, ~7 s total at T=151; compile > solve).
- Fix: write the CSV only at the end + sparse checkpoints, and/or keep the
  vector-valued columns out of the per-step CSV (save them once as `.npy`). This
  removes ~76% of the runtime.

### port_dro (`port_DRO_orig_p2.py`) — canonicalization-bound, moderate
- The LP `createproblem_portLP_p2(N, m=150)` is rebuilt each interval; **compile
  time dominates the solve** and grows with N: compile 1.0/2.3/5.7/10.1/15.5 s and
  solve ~0.5–3.9 s at N=250/500/1000/1500/2000.
- The worst-case dual (`worst_case_p2`, the metric) is cheap (≤0.23 s at N=2000),
  despite the N×m structure — not a bottleneck.
- Memory is small (sparse LP, CLARABEL): < ~0.3 GB/worker.
- Per-task full-T ≈ 5–10 min (21 interval LP solves at growing N + `t_list`).

### reg_ours (`reg_orig_p2.py`) — MI solver-bound (small clusters)
- Each interval solves **3 MOSEK problems**: DRO MI-SOCP on K micro-clusters, DRO
  MI-SOCP on K macro-clusters, and an SAA scenario MI. The two DRO MI-SOCPs cost
  **2–3.5 s of pure solver time each**; the SAA one is ~0.01–0.02 s.
- The cost is the **50 binary feature-selection variables** (cardinality k=5
  best-subset), **not** N (clusters are tiny: K=10/15/25) and **not**
  canonicalization (compile ~0.015 s). KMeans/clustering is negligible (0.33 s
  total over the run).
- Per-task full-T ≈ 5–6 min (≈37 active intervals × 3 solves).

### reg_dro (`reg_DRO_orig_p2.py`) — MI solver-bound (growing N)
- Each interval solves the DRO MI-SOCP and the SAA MI on **N = num_dat** (grows to
  ~2000). 99% of runtime is MOSEK solve time; compile is negligible (≤0.06 s).
- Solve time at N=2000 ranges **4–35 s across the eps grid** (worst at eps≈0.7);
  it does **not** approach the 1500 s MOSEK cap for these instances. MI time is
  instance-dependent, so keep walltime margin.
- **Memory grows with N**: peak RSS 0.70 → 1.49 GB for N=250 → 2000. This sets
  `--mem-per-cpu` for the regression jobs (~1.5 GB per concurrent worker).
- Per-task full-T ≈ 10–25 min typical (dominated by the large-N `t_list`
  checkpoints 1249…2000).

## Cross-cutting issues

1. **Oversized SLURM requests** (see table below) — the primary cause of long
   PENDING waits. `portfolio_new_p2.sh` asks 175 GB / 24 h; `regression.sh` asks
   140 GB / 23 h. Actual peaks are tens of GB and ≤ ~30 min/task.
2. **MOSEK threads vs joblib.** MOSEK defaults to all cores; with 35 joblib
   workers that is 35× oversubscription. Pin `MSK_IPAR_NUM_THREADS = 1` so each
   worker uses one core. (Profiling above was single-thread; production currently
   oversubscribes and is slower per solve.)
3. **Dataset broadcast is NOT a memory problem.** joblib/loky auto-memmaps the
   `synthetic_*` array (24 MB > 1 MB threshold) read-only across workers, so it is
   shared, not copied 35×. Memory is dominated by the per-solve MOSEK working set.
4. **`ignore_dpp=True` + rebuild-every-interval** removes canonicalization reuse.
   For full DRO this is mostly unavoidable (N changes); for reg_ours the structure
   is fixed (N=K) so a cached parametric problem would remove the (small) compile.
5. **`w2_dist` (`port_new/utils.py:475`) does `float()` on a size-1 array** in the
   `k1['K']>K` branch — raises under numpy ≥ 2.0. The code assumes numpy < 2.

## Recommended SLURM requests

Concurrency model: a job fans out `R × len(eps_init)` joblib tasks across
`njobs = min(100, cpus)` cores. Sizing assumes `--cpus-per-task=35` and
**MOSEK pinned to 1 thread/worker**. Walltime ≈ `ceil(tasks/cpus) ×
max_task_time × 1.3`; memory ≈ `cpus × peak_worker_RSS × 1.3`.

| Job (variant) | tasks (R×M) | peak RSS/worker | **mem-per-cpu** | **time** | (current) |
|---|--:|--:|--:|--:|--:|
| `portfolio_new_p2.sh` → port_dro | 5×5=25 | ~0.3 GB | **1G** | **1:00:00** | 5G / 24h |
| port_ours (after CSV fix) | 5×7=35 | ~0.5 GB | **1G** | **1:30:00** | — |
| port_ours (current code) | 5×7=35 | ~0.5 GB | 1G | 4:00:00 | — |
| `regression.sh` → reg_dro | 5×9=45 | ~1.5 GB | **2G** | **4:00:00** | 4G / 23h |
| reg_ours | 10×9=90 | ~1.0 GB | **2G** | **2:00:00** | — |

Notes:
- Memory was the wildly over-provisioned axis: 35 cpu × the recommended
  mem-per-cpu gives 35 GB (portfolio) / 70 GB (regression) vs the current
  175 GB / 140 GB. Smaller `--mem` and `--time` are the main queue-time win.
- If queue waits persist, dropping `--cpus-per-task` to ~20 schedules even faster
  at the cost of ~one extra task-batch of walltime.
- Keep margin on `reg_dro` walltime because MI solve time is instance-dependent.

## How to reproduce

```bash
source profiling/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic
python profiling/profile_run.py  --variant reg_dro --T 201        # per-variant profile
python profiling/scaling_study.py --variant reg_dro --Ns 250,500,1000,1500,2000   # cost & mem vs N
```
Outputs land in `profiling/results/` (`*.profile.txt`, `*.summary.json`,
`scaling_*.json`).
