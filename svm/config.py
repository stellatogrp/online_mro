"""Defaults for the SPARSE-SVM experiment, keyed by (method, solver).

Provenance: the values ACTUALLY LAUNCHED by ``slurm/svm1.sh`` (branch
paper-experiments), falling back to the legacy argparse defaults of
``svm1/svm_grad.py`` / ``svm1/svm_orig.py`` / ``svm1/svm_DRO_grad.py`` /
``svm1/svm_DRO_orig.py`` / ``svm1/svm_true_saa.py`` where the SLURM script
did not override them.

Notes
-----
* K: the legacy MRO drivers hard-required ``SLURM_ARRAY_TASK_ID`` and picked
  ``K = [10, 15, 25][idx]``; ``slurm/svm1.sh`` ran array index 1, i.e. K=15.
  ``run.py`` replaces that with an explicit ``--K`` (default 15).
* fixed_time: the SLURM launches set ``--fixed_time 2001`` = T (clusters are
  never frozen).  We encode that as ``fixed_time = None`` -> "use T".
* epsilon lists are EXACTLY the legacy per-driver lists (including their
  T-dependent switches), keyed by (method, solver).
"""

# Logging / warm-start step lists (identical across the legacy drivers).
T_LIST = [4, 5, 24, 25, 49, 50, 99, 100]
T_SOLVE_LIST = [5, 25, 50, 100]

# Per-(method, solver) defaults.  ``fixed_time=None`` means "equal to T".
DEFAULTS = {
    ('mro', 'subgrad'): dict(
        dataset='ijcnn1', K=15, T=2001, R=10, r_start=0,
        k=10, Q=500, fixed_time=None, interval=1, N_init=5,
        rmse_mult=1.25, cluster_interval=50, power=0.02, p=1,
        eta=0.1, solve_interval=200,
    ),
    ('mro', 'exact'): dict(
        dataset='ijcnn1', K=15, T=2001, R=10, r_start=0,
        k=10, Q=500, fixed_time=None, interval=200, N_init=5,
        rmse_mult=1.25, cluster_interval=50, power=0.02, p=1,
    ),
    ('dro', 'subgrad'): dict(
        dataset='ijcnn1', T=2001, R=10, r_start=0,
        k=10, interval=1, N_init=5, power=0.25, p=1,
        eta=0.1, solve_interval=200,
    ),
    ('dro', 'exact'): dict(
        dataset='ijcnn1', T=2001, R=10, r_start=0,
        k=10, interval=200, interval_SAA=100, N_init=5, power=0.25, p=1,
    ),
    ('true_saa', None): dict(
        dataset='ijcnn1', k=10, N_true=20000, R=10, r_start=0, T=2001,
    ),
}


def eps_list(method, solver, T):
    """Legacy per-driver epsilon sweeps (verbatim, including T switches)."""
    if method == 'mro' and solver == 'subgrad':
        # svm1/svm_grad.py
        eps_init = [0.18, 0.16, 0.15, 0.12, 0.1, 0.095]
        if T >= 5000:
            eps_init = [0.15, 0.12, 0.1, 0.095, 0.09, 0.85]
        return eps_init
    if method == 'mro' and solver == 'exact':
        # svm1/svm_orig.py
        eps_init = [0.18, 0.16, 0.15, 0.12, 0.1, 0.095]
        if T >= 5000:
            eps_init = [0.18, 0.16, 0.15, 0.12, 0.1, 0.095, 0.09, 0.85]
        return eps_init
    if method == 'dro' and solver == 'subgrad':
        # svm1/svm_DRO_grad.py
        if T >= 10000:
            eps_init = [0.1, 0.05, 0.02]
        else:
            eps_init = [0.18, 0.16, 0.15, 0.12, 0.1, 0.095]
        return eps_init
    if method == 'dro' and solver == 'exact':
        # svm1/svm_DRO_orig.py
        if T >= 10000:
            eps_init = [0.15, 0.12, 0.1, 0.095, 0.09]
        else:
            eps_init = [0.18, 0.16, 0.15, 0.12, 0.1, 0.095]
        return eps_init
    raise ValueError(f"no epsilon list for method={method!r}, solver={solver!r}")
