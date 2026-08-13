"""End-to-end smoke tests for the consolidated drivers.

Runs ``python -m {portfolio,svm}.run`` as a subprocess (same interpreter,
cwd=paper_experiments) at the minimum viable problem sizes for each of the
10 paths (2 experiments x {mro,dro} x {exact,subgrad} + true_saa) and checks
exit code, final CSV existence, non-emptiness, and absence of all-NaN
columns.  One representative path per experiment is additionally run twice
into different directories to check bit-identical (non-time-column)
determinism.

All svm paths and portfolio true_saa require MOSEK and are marked
``mosek``; they self-skip without a license.
"""
import pandas as pd
import pytest

from .conftest import PAPER_DIR, assert_ran, run_cli, skip_without_mosek

# (id, module, args (without --results_dir), needs_mosek, final CSV rel path)
SMOKE_PATHS = [
    (
        "portfolio-mro-exact", "portfolio.run",
        ["--method", "mro", "--solver", "exact", "--T", "6", "--R", "1",
         "--K", "3", "--m", "8", "--N_init", "3", "--interval", "1",
         "--eps_index", "0"],
        False, "T5/df_K3R0.csv",
    ),
    (
        "portfolio-mro-subgrad", "portfolio.run",
        ["--method", "mro", "--solver", "subgrad", "--T", "6", "--R", "1",
         "--K", "3", "--m", "8", "--N_init", "3", "--interval", "1",
         "--eps_index", "0"],
        False, "T5/df_K3R0.csv",
    ),
    (
        "portfolio-dro-exact", "portfolio.run",
        ["--method", "dro", "--solver", "exact", "--T", "6", "--R", "1",
         "--m", "8", "--N_init", "3", "--interval", "1", "--interval_SAA", "1",
         "--eps_index", "0"],
        False, "T5/df_K0R0.csv",
    ),
    (
        "portfolio-dro-subgrad", "portfolio.run",
        ["--method", "dro", "--solver", "subgrad", "--T", "6", "--R", "1",
         "--m", "8", "--N_init", "3", "--interval", "1", "--eps_index", "0"],
        False, "T5/df_K0R0.csv",
    ),
    (
        "portfolio-true_saa", "portfolio.run",
        ["--method", "true_saa", "--R", "2", "--N_true", "500", "--m", "8"],
        True, "true_saa_all.csv",
    ),
    (
        "svm-mro-exact", "svm.run",
        ["--method", "mro", "--solver", "exact", "--T", "6", "--R", "1",
         "--K", "3", "--eps_index", "0", "--interval", "1"],
        True, "T5/df_K3R0.csv",
    ),
    (
        "svm-mro-subgrad", "svm.run",
        ["--method", "mro", "--solver", "subgrad", "--T", "6", "--R", "1",
         "--K", "3", "--eps_index", "0", "--interval", "1"],
        True, "T5/df_K3R0.csv",
    ),
    (
        "svm-dro-exact", "svm.run",
        ["--method", "dro", "--solver", "exact", "--T", "6", "--R", "1",
         "--eps_index", "0", "--interval", "1", "--interval_SAA", "1"],
        True, "T5/df_K0R0.csv",
    ),
    (
        "svm-dro-subgrad", "svm.run",
        ["--method", "dro", "--solver", "subgrad", "--T", "6", "--R", "1",
         "--eps_index", "0", "--interval", "1"],
        True, "T5/df_K0R0.csv",
    ),
    (
        "svm-true_saa", "svm.run",
        ["--method", "true_saa", "--R", "2", "--N_true", "60"],
        True, "true_saa_all.csv",
    ),
]


def _params():
    out = []
    for pid, module, args, needs_mosek, final in SMOKE_PATHS:
        marks = [pytest.mark.mosek] if needs_mosek else []
        out.append(pytest.param(module, args, needs_mosek, final,
                                id=pid, marks=marks))
    return out


def _run_driver(module, args, results_dir):
    proc = run_cli(["-m", module, "--results_dir", str(results_dir), *args],
                   cwd=PAPER_DIR)
    assert_ran(proc, label=module)
    return proc


def _read_final(csv_path):
    # true_saa CSVs are written index=False; the per-seed df_K*R*.csv carry a
    # leading unnamed index column.  index_col=0 handles the latter; for the
    # former the first column is a real value column, so read plainly.
    if csv_path.name.startswith("true_saa"):
        return pd.read_csv(csv_path)
    return pd.read_csv(csv_path, index_col=0)


@pytest.mark.parametrize("module,args,needs_mosek,final", _params())
def test_smoke_path(tmp_path, module, args, needs_mosek, final):
    if needs_mosek:
        skip_without_mosek()
    results_dir = tmp_path / "results"
    _run_driver(module, args, results_dir)

    csv_path = results_dir / final
    assert csv_path.exists(), f"final CSV missing: {csv_path}"
    df = _read_final(csv_path)
    assert len(df) > 0, "final CSV has no rows"
    all_nan = [c for c in df.columns if df[c].isna().all()]
    assert not all_nan, f"all-NaN columns in final CSV: {all_nan}"


# ---------------------------------------------------------------------------
# Determinism: run one representative path per experiment twice into
# different directories and require bit-identical non-time columns.
# ---------------------------------------------------------------------------
DETERMINISM_PATHS = {
    "portfolio": next(p for p in SMOKE_PATHS if p[0] == "portfolio-mro-exact"),
    "svm": next(p for p in SMOKE_PATHS if p[0] == "svm-mro-exact"),
}


def _nontime_text(csv_path):
    """The final CSV as raw text columns, with wall-clock columns dropped."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    drop = [c for c in df.columns if "time" in c.lower()]
    return df.drop(columns=drop)


@pytest.mark.parametrize(
    "experiment",
    [
        pytest.param("portfolio", id="portfolio-mro-exact"),
        pytest.param("svm", id="svm-mro-exact", marks=pytest.mark.mosek),
    ],
)
def test_determinism(tmp_path, experiment):
    _, module, args, needs_mosek, final = DETERMINISM_PATHS[experiment]
    if needs_mosek:
        skip_without_mosek()
    frames = []
    for sub in ("run_a", "run_b"):
        results_dir = tmp_path / sub
        _run_driver(module, args, results_dir)
        frames.append(_nontime_text(results_dir / final))
    assert frames[0].columns.tolist() == frames[1].columns.tolist()
    assert frames[0].equals(frames[1]), (
        "re-running the driver produced different non-time output"
    )
