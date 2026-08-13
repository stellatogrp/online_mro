"""Legacy-vs-consolidated numerical equivalence tests (all marked slow).

For each of the 10 experiment paths, runs the legacy driver (from the repo
root, exactly as the SLURM scripts did -- including SLURM_ARRAY_TASK_ID for
the legacy MRO drivers, whose K comes from ``K_arr[idx]``, NOT from ``--K``)
and the consolidated ``python -m {portfolio,svm}.run`` driver at IDENTICAL
settings, then compares the final per-seed CSVs column by column.

Wall-clock columns (any name containing "time") are excluded; everything
else must match to rtol 1e-12 / atol 1e-14.

Matched settings (T=30, R=2 everywhere):
  * Legacy portfolio MRO forces K = [15, 25, 30][SLURM_ARRAY_TASK_ID]; we use
    idx=0 -> K=15 and pass --K 15 to the new driver.
  * Legacy svm MRO forces K = [10, 15, 25][idx]; idx=0 -> K=10 with
    N_init=20, which also keeps cluster growth away from the legacy w2_dist
    branch that crashes under numpy>=2 (fixed only in the new copy).
  * Portfolio runs use --box_q 0.05: heavier clipping (the 0.3-0.4 defaults)
    creates duplicate corner rows at m=8, and micro-cluster init with
    N_init=20 micro-clusters then crashes in BOTH drivers (KMeans finds
    fewer than N_init distinct clusters).
  * Every flag whose legacy argparse default differs from the consolidated
    config default is passed explicitly to both sides.
  * Neither side gets --eps_index: the legacy drivers cannot restrict the
    sweep, so both run the identical full epsilon sweep.

Legacy drivers write (and then delete) intermediate files plus metadata; only
the final ``df_K{K}R{r}.csv`` (or ``true_saa_*.csv``) files are compared.
All legacy output goes under the pytest tmp dir via --foldername; the legacy
``__main__`` blocks write nowhere else (verified by reading them).
"""
import numpy as np
import pandas as pd
import pytest

from .conftest import PAPER_DIR, REPO_ROOT, assert_ran, run_cli, skip_without_mosek

# The legacy reference drivers are not distributed with this repository.
# When they are absent the gate auto-skips; with a checkout of the original
# research codebase, place this repo at <original>/paper_experiments (or
# symlink) to run it.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REPO_ROOT / "port_box").is_dir() or not (REPO_ROOT / "svm1").is_dir(),
        reason="legacy reference drivers not present"),
]

T, R = 30, 2
TDIR_LEGACY = f"T{T - 1}R{R}"   # legacy final dir: T{T-1}R{R}/
TDIR_NEW = f"T{T - 1}"          # consolidated final dir: T{T-1}/ (no R suffix)

# --- matched portfolio settings (see module docstring re box_q) -------------
PORT_COMMON = ["--T", T, "--R", R, "--m", "8", "--n_total", "20000",
               "--power", "0.025", "--box_q", "0.05"]
PORT_MRO = ["--Q", "500", "--fixed_time", "1500", "--interval", "1",
            "--N_init", "20", "--rmse_mult", "1.25", "--cluster_interval", "1"]

# --- matched svm settings ---------------------------------------------------
SVM_COMMON = ["--T", T, "--R", R, "--k", "10", "--dataset", "ijcnn1",
              "--power", "0.5", "--N_init", "20", "--interval", "1"]
SVM_MRO = ["--Q", "500", "--fixed_time", "1500", "--rmse_mult", "2",
           "--cluster_interval", "1"]

SLURM0 = {"SLURM_ARRAY_TASK_ID": "0"}


def _df_pairs(K):
    """(legacy_rel, new_rel, index_col) triples for the per-seed CSVs."""
    return [(f"{TDIR_LEGACY}/df_K{K}R{r}.csv", f"{TDIR_NEW}/df_K{K}R{r}.csv", 0)
            for r in range(R)]


TRUE_SAA_PAIRS = [(f"true_saa_R{r}.csv", f"true_saa_R{r}.csv", None)
                  for r in range(R)] + [("true_saa_all.csv", "true_saa_all.csv", None)]

# Each spec:
#   id, needs_mosek, legacy script, legacy extra args, legacy env,
#   new module, new args, list of (legacy_rel, new_rel, index_col)
SPECS = [
    (
        "portfolio-mro-exact", False,
        "port_box/port_box_orig.py", PORT_COMMON + PORT_MRO, SLURM0,
        "portfolio.run",
        ["--method", "mro", "--solver", "exact", "--K", "15"]
        + PORT_COMMON + PORT_MRO,
        _df_pairs(15),
    ),
    (
        "portfolio-mro-subgrad", False,
        "port_box/port_box.py",
        PORT_COMMON + PORT_MRO + ["--eta_0", "0.01", "--line_search"], SLURM0,
        "portfolio.run",
        ["--method", "mro", "--solver", "subgrad", "--K", "15"]
        + PORT_COMMON + PORT_MRO + ["--eta_0", "0.01", "--line_search",
                                    "--solve_interval", "0"],
        _df_pairs(15),
    ),
    (
        "portfolio-dro-exact", False,
        "port_box/port_box_DRO_orig.py",
        PORT_COMMON + ["--interval", "1", "--interval_SAA", "1",
                       "--N_init", "20", "--alpha", "0.8"], {},
        "portfolio.run",
        ["--method", "dro", "--solver", "exact"]
        + PORT_COMMON + ["--interval", "1", "--interval_SAA", "1",
                         "--N_init", "20", "--alpha", "0.8"],
        _df_pairs(0),
    ),
    (
        "portfolio-dro-subgrad", False,
        "port_box/port_box_DRO.py",
        PORT_COMMON + ["--interval", "1", "--N_init", "20", "--alpha", "0.8",
                       "--eta_0", "0.1", "--line_search"], {},
        "portfolio.run",
        ["--method", "dro", "--solver", "subgrad"]
        + PORT_COMMON + ["--interval", "1", "--N_init", "20", "--alpha", "0.8",
                         "--eta_0", "0.1", "--line_search",
                         "--solve_interval", "0"],
        _df_pairs(0),
    ),
    (
        "portfolio-true_saa", True,
        "port_box/port_box_true_saa.py",
        ["--m", "8", "--alpha", "0.8", "--N_true", "500", "--n_total", "20000",
         "--R", R, "--r_start", "0", "--box_q", "0.05"], {},
        "portfolio.run",
        ["--method", "true_saa", "--m", "8", "--alpha", "0.8",
         "--N_true", "500", "--n_total", "20000", "--R", R, "--r_start", "0",
         "--box_q", "0.05"],
        TRUE_SAA_PAIRS,
    ),
    (
        "svm-mro-exact", True,
        "svm1/svm_orig.py", SVM_COMMON + SVM_MRO + ["--p", "1"], SLURM0,
        "svm.run",
        ["--method", "mro", "--solver", "exact", "--K", "10"]
        + SVM_COMMON + SVM_MRO + ["--p", "1"],
        _df_pairs(10),
    ),
    (
        "svm-mro-subgrad", True,
        "svm1/svm_grad.py",
        SVM_COMMON + SVM_MRO + ["--p", "2", "--eta", "0.01",
                                "--solve_interval", "500"], SLURM0,
        "svm.run",
        ["--method", "mro", "--solver", "subgrad", "--K", "10"]
        + SVM_COMMON + SVM_MRO + ["--p", "2", "--eta", "0.01",
                                  "--solve_interval", "500"],
        _df_pairs(10),
    ),
    (
        "svm-dro-exact", True,
        "svm1/svm_DRO_orig.py",
        SVM_COMMON + ["--interval_SAA", "1", "--p", "1"], {},
        "svm.run",
        ["--method", "dro", "--solver", "exact"]
        + SVM_COMMON + ["--interval_SAA", "1", "--p", "1"],
        _df_pairs(0),
    ),
    (
        "svm-dro-subgrad", True,
        "svm1/svm_DRO_grad.py",
        SVM_COMMON + ["--p", "2", "--eta", "0.01", "--solve_interval", "500"], {},
        "svm.run",
        ["--method", "dro", "--solver", "subgrad"]
        + SVM_COMMON + ["--p", "2", "--eta", "0.01", "--solve_interval", "500"],
        _df_pairs(0),
    ),
    (
        "svm-true_saa", True,
        "svm1/svm_true_saa.py",
        ["--dataset", "ijcnn1", "--k", "10", "--N_true", "60", "--R", R,
         "--r_start", "0"], {},
        "svm.run",
        ["--method", "true_saa", "--dataset", "ijcnn1", "--k", "10",
         "--N_true", "60", "--R", R, "--r_start", "0"],
        TRUE_SAA_PAIRS,
    ),
]


def _compare_csvs(legacy_csv, new_csv, index_col):
    """Non-time columns must agree: names, order, and values."""
    a = pd.read_csv(legacy_csv, index_col=index_col)
    b = pd.read_csv(new_csv, index_col=index_col)
    cols_a = [c for c in a.columns if "time" not in c.lower()]
    cols_b = [c for c in b.columns if "time" not in c.lower()]
    assert cols_a == cols_b, (
        f"non-time column mismatch for {legacy_csv.name}:\n"
        f"legacy: {cols_a}\nnew:    {cols_b}")
    assert len(a) == len(b), (
        f"row-count mismatch for {legacy_csv.name}: {len(a)} vs {len(b)}")
    for c in cols_a:
        x, y = a[c], b[c]
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            np.testing.assert_allclose(
                y.to_numpy(dtype=float), x.to_numpy(dtype=float),
                rtol=1e-12, atol=1e-14, equal_nan=True,
                err_msg=f"{legacy_csv.name}, column {c}")
        else:
            assert x.astype(str).fillna("nan").tolist() == \
                y.astype(str).fillna("nan").tolist(), \
                f"{legacy_csv.name}, non-numeric column {c} differs"


@pytest.mark.parametrize(
    "needs_mosek,legacy_script,legacy_args,legacy_env,module,new_args,pairs",
    [pytest.param(*s[1:], id=s[0],
                  marks=[pytest.mark.mosek] if s[1] else [])
     for s in SPECS],
)
def test_legacy_equivalence(tmp_path, needs_mosek, legacy_script, legacy_args,
                            legacy_env, module, new_args, pairs):
    if needs_mosek:
        skip_without_mosek()

    legacy_dir = tmp_path / "legacy"
    new_dir = tmp_path / "new"
    legacy_dir.mkdir()
    new_dir.mkdir()

    # Legacy driver: run from the repo root; --foldername needs the trailing
    # slash the legacy string concatenation assumes.
    proc = run_cli(
        [legacy_script, "--foldername", f"{legacy_dir}/", *legacy_args],
        cwd=REPO_ROOT, env_extra=legacy_env)
    assert_ran(proc, label=f"legacy {legacy_script}")

    # Consolidated driver at identical settings.  Pin the accelerated
    # implementations off: this gate validates the consolidation against the
    # legacy numerics; the C worst-case kernel and the direct Clarabel SOCP
    # have their own oracle tests (test_cworst.py, test_direct_socp.py).
    proc = run_cli(
        ["-m", module, "--results_dir", str(new_dir), *new_args],
        cwd=PAPER_DIR,
        env_extra={"PORT_WORSTCASE_IMPL": "python", "PORT_DRO_DIRECT": "0"})
    assert_ran(proc, label=f"new {module}")

    for legacy_rel, new_rel, index_col in pairs:
        legacy_csv = legacy_dir / legacy_rel
        new_csv = new_dir / new_rel
        assert legacy_csv.exists(), f"legacy final CSV missing: {legacy_csv}"
        assert new_csv.exists(), f"new final CSV missing: {new_csv}"
        _compare_csvs(legacy_csv, new_csv, index_col)
