"""Problem-agnostic helpers for the SPARSE-SVM experiment.

Provenance: verbatim copy of the pieces of legacy ``svm/utils.py`` that
``utils_svm.py`` and the SVM drivers actually use (run metadata, file
cleanup, SLURM-aware process count, distance / simplex helpers).  The MOSEK
solver constants live in ``solver_params.py``; the SVM-specific machinery
(problem builders, label-pure clustering, regret bookkeeping) lives in
``utils_svm.py``.
"""
import json
import os

import joblib
import numpy as np
import ot
from scipy.spatial.distance import cdist


def save_run_metadata(metadata, paths):
    """Write a run-metadata JSON + human-readable .txt into one or more dirs.

    Called as the *first* thing in each experiment ``__main__`` block so the
    metadata is persisted even if the run later crashes mid-experiment.
    """
    lines = ["# Run metadata", ""]
    for k, v in metadata.items():
        if isinstance(v, (list, tuple)):
            lines.append(f"- {k}: {', '.join(str(x) for x in v)}")
        else:
            lines.append(f"- {k}: {v}")
    txt = "\n".join(lines) + "\n"

    for entry in paths:
        if isinstance(entry, str):
            d, jname = entry, 'metadata.json'
        else:
            d, jname = entry
            jname = jname or 'metadata.json'
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, jname), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        with open(os.path.join(d, 'metadata.txt'), 'w') as f:
            f.write(txt)


def remove_files(paths):
    """Remove exactly the given files, skipping any that don't exist.

    Used at the end of a run to discard the specific now-redundant
    intermediate files that *this* run generated, without touching anything
    else that may share the same directory (e.g. another concurrent run's
    output, or files left over from an unrelated experiment)."""
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def get_n_processes(max_n=np.inf):
    """Number of processes to use (SLURM-aware)."""
    try:
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()
    return max(min(max_n, n_cpus), 1)


def calc_rmse(dat, mean):
    rmse = 0
    for d in dat:
        rmse += np.linalg.norm(d - mean, 2) ** 2
    return rmse


def project_simplex(v):
    """Euclidean projection onto {x >= 0, sum x = 1} (Duchi et al. 2008)."""
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, n + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def w2_dist(k1, k2, m):
    K = k2['K']
    val = 0
    for k in range(K):
        val += np.abs(k1["w"][k] - k2["w"][k]) * np.linalg.norm(k1["d"][k] - k2["d"][k])
    if k1['K'] > K:
        dists = cdist(k1['d'][K].reshape((1, m)), k2['d'][:K])
        # NUMPY 2.x FIX (behavior-preserving): legacy did ``val += dists @ ...``
        # (a shape-(1,) array) and then ``float(val)``, which numpy >= 2 rejects
        # ("only 0-dimensional arrays can be converted").  Indexing [0] extracts
        # the identical scalar; the arithmetic is unchanged.
        val += (dists @ np.abs(k2['w'][:K] - k1['w'][:K]))[0]
    return float(val)


def wasserstein(samples_p, samples_q):
    """Wasserstein-1 distance between two empirical distributions."""
    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)
    N = samples_p.shape[0]
    M = samples_q.shape[0]
    weights_p = np.ones(N) / N
    weights_q = np.ones(M) / M
    cost_matrix = ot.dist(samples_p, samples_q, metric='euclidean')
    return ot.emd2(weights_p, weights_q, cost_matrix)
