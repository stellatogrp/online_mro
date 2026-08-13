"""Unit tests for plotting.plots_utils.setup_dfs seed aggregation.

Hand-builds tiny per-seed CSVs (including a ragged seed and a K-collision
decoy) and checks the mean / 25% / 75% aggregates against hand-computed
values, the numeric seed-sort of the glob, and the init=False cache
round-trip.
"""
import os

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from plotting.plots_utils import _seed_csvs, setup_dfs


def _write(dirpath, name, t, a, b):
    pd.DataFrame({"t": t, "a": a, "b": b}).to_csv(dirpath / name)


@pytest.fixture()
def seed_dir(tmp_path):
    d = tmp_path / "seeds"
    d.mkdir()
    # Three seeds for K=3; seed 10 is ragged (2 rows instead of 3) and its
    # two-digit index exercises the numeric (not lexicographic) sort.
    _write(d, "df_K3R0.csv", [0, 1, 2], [1.0, 2.0, 10.0], [4.0, 4.0, 10.0])
    _write(d, "df_K3R1.csv", [0, 1, 2], [2.0, 3.0, 30.0], [8.0, 6.0, 20.0])
    _write(d, "df_K3R10.csv", [0, 1], [4.0, 7.0], [12.0, 14.0])
    # Decoy: same prefix 'df_K3' but K=30; must NOT be globbed for K=3.
    _write(d, "df_K30R0.csv", [0, 1, 2], [1000.0] * 3, [1000.0] * 3)
    return d


# Hand-computed expectations (pandas linear-interpolation quantiles):
#   'a' row 0: [1, 2, 4]   mean 7/3, q25 1.5,  q75 3.0
#   'a' row 1: [2, 3, 7]   mean 4,   q25 2.5,  q75 5.0
#   'a' row 2: [10, 30]    mean 20,  q25 15.0, q75 25.0   (seed 10 dropped out)
#   'b' row 0: [4, 8, 12]  mean 8,   q25 6.0,  q75 10.0
#   'b' row 1: [4, 6, 14]  mean 8,   q25 5.0,  q75 10.0
#   'b' row 2: [10, 20]    mean 15,  q25 12.5, q75 17.5
EXPECTED = {
    "mean": {"a": [7.0 / 3.0, 4.0, 20.0], "b": [8.0, 8.0, 15.0]},
    25: {"a": [1.5, 2.5, 15.0], "b": [6.0, 5.0, 12.5]},
    75: {"a": [3.0, 5.0, 25.0], "b": [10.0, 10.0, 17.5]},
}


def test_seed_glob_numeric_sort_and_decoy(seed_dir):
    names = [os.path.basename(p) for p in _seed_csvs(str(seed_dir), 3)]
    # Seed 10 is found and sorted numerically after 0 and 1; the K=30 decoy
    # is excluded.
    assert names == ["df_K3R0.csv", "df_K3R1.csv", "df_K3R10.csv"]


def _check_aggregates(df, quantiles):
    assert 3 in df and 3 in quantiles
    mean_df = df[3]
    # Output length is the max replication length (ragged seed 10 has 2 rows).
    assert len(mean_df) == 3
    for col in ("a", "b"):
        np.testing.assert_allclose(
            mean_df[col].to_numpy(dtype=float), EXPECTED["mean"][col],
            rtol=1e-12, err_msg=f"mean, column {col}")
        for q in (25, 75):
            np.testing.assert_allclose(
                quantiles[3][q][col].to_numpy(dtype=float), EXPECTED[q][col],
                rtol=1e-12, err_msg=f"q{q}, column {col}")
    # The t axis survives aggregation (rows 0..2, seed 10 missing row 2).
    np.testing.assert_allclose(mean_df["t"].to_numpy(dtype=float), [0, 1, 2])


def test_setup_dfs_init_true(seed_dir, tmp_path):
    out = tmp_path / "agg"
    out.mkdir()
    df, quantiles = setup_dfs(
        folderout=str(out) + "/", foldername=str(seed_dir) + "/",
        K_list=[3], quant_list=[25, 75], init=True)
    _check_aggregates(df, quantiles)
    # Cache files were written.
    for name in ("df_K3.csv", "quantiles_25K3.csv", "quantiles_75K3.csv"):
        assert (out / name).exists()

    # If the decoy had leaked in, row 0 of 'a' would average 1000 in; make the
    # exclusion explicit.
    assert df[3]["a"].iloc[0] < 100


def test_setup_dfs_init_false_roundtrip(seed_dir, tmp_path):
    out = tmp_path / "agg"
    out.mkdir()
    df1, quant1 = setup_dfs(
        folderout=str(out) + "/", foldername=str(seed_dir) + "/",
        K_list=[3], quant_list=[25, 75], init=True)
    # init=False must reproduce the same aggregates purely from the cache
    # (foldername no longer needed).
    df2, quant2 = setup_dfs(
        folderout=str(out) + "/", foldername=None,
        K_list=[3], quant_list=[25, 75], init=False)
    _check_aggregates(df2, quant2)
    pd.testing.assert_frame_equal(df1[3], df2[3])
    for q in (25, 75):
        pd.testing.assert_frame_equal(quant1[3][q], quant2[3][q])


def test_setup_dfs_missing_k_skipped(seed_dir, tmp_path):
    out = tmp_path / "agg"
    out.mkdir()
    # K=7 has no per-seed files: it is skipped, K=3 still aggregates.
    df, quantiles = setup_dfs(
        folderout=str(out) + "/", foldername=str(seed_dir) + "/",
        K_list=[3, 7], quant_list=[25, 75], init=True)
    assert 7 not in df and 7 not in quantiles
    _check_aggregates(df, quantiles)
