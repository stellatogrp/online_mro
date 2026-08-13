"""Unit tests for the SPARSE-SVM data loader (svm/utils_svm.py::load_svm_dataset)
and the simplex projection helper (svm/utils.py::project_simplex).

What the loader standardizes (read off the code, asserted below): it z-scores
each feature over the CONCATENATED train+test pool (LIBSVM ``ijcnn1`` +
``ijcnn1.t`` stacked), not over the train split alone -- the pool is re-split
randomly at experiment time, so LIBSVM's own split is not load-bearing.

Self-sufficient: no conftest fixtures are used.
"""
import pathlib
import sys

import numpy as np
import cvxpy as cp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from svm.utils import project_simplex  # noqa: E402
from svm.utils_svm import load_svm_dataset  # noqa: E402

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "svm" / "data"
M_IJCNN1 = 22

# Load once at module scope (0.3 s each); raw + standardized variants.
_STD_CACHE = {}


def _load(standardize):
    if standardize not in _STD_CACHE:
        _STD_CACHE[standardize] = load_svm_dataset(
            "ijcnn1", standardize=standardize)
    return _STD_CACHE[standardize]


def _count_lines(path):
    """One sample per line in svmlight format."""
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))


# --------------------------------------------------------------------------- #
# load_svm_dataset
# --------------------------------------------------------------------------- #
def test_ijcnn1_shapes_match_files():
    n_train = _count_lines(DATA_DIR / "ijcnn1")
    n_test = _count_lines(DATA_DIR / "ijcnn1.t")
    data, m = _load(True)
    assert m == M_IJCNN1
    assert data.shape == (n_train + n_test, M_IJCNN1 + 1)
    # loader concatenates train first, then test
    assert n_train > 0 and n_test > 0


def test_ijcnn1_labels_are_plus_minus_one():
    data, m = _load(True)
    labels = np.unique(data[:, m])
    assert set(labels).issubset({-1.0, 1.0})
    # both classes present
    assert set(labels) == {-1.0, 1.0}


def test_ijcnn1_standardization_is_over_full_pool():
    """standardize=True z-scores each feature over the whole train+test pool:
    pool means ~0, pool stds ~1, and the transform is exactly
    (raw - pool_mean) / pool_std applied to the raw concatenated pool."""
    data_std, m = _load(True)
    X = data_std[:, :m]
    np.testing.assert_allclose(X.mean(axis=0), np.zeros(m), atol=1e-10)
    np.testing.assert_allclose(X.std(axis=0), np.ones(m), atol=1e-8)

    data_raw, m_raw = _load(False)
    assert m_raw == m
    X_raw = data_raw[:, :m]
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    assert np.all(sigma > 1e-12)   # no degenerate ijcnn1 columns
    np.testing.assert_allclose(X, (X_raw - mu) / sigma, atol=1e-10)
    # labels are untouched by standardization
    np.testing.assert_array_equal(data_std[:, m], data_raw[:, m])


def test_ijcnn1_raw_features_are_not_standardized():
    data_raw, m = _load(False)
    X_raw = data_raw[:, :m]
    # sanity: the raw pool is NOT already z-scored (so the test above is
    # actually exercising the standardization branch)
    assert np.max(np.abs(X_raw.mean(axis=0))) > 1e-3


# --------------------------------------------------------------------------- #
# project_simplex (svm/utils.py copy, tested independently)
# --------------------------------------------------------------------------- #
def test_project_simplex_matches_cvxpy_qp():
    rng = np.random.default_rng(17)
    for _ in range(5):
        v = 2.0 * rng.standard_normal(8)
        out = project_simplex(v)

        x = cp.Variable(8)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - v)),
                          [x >= 0, cp.sum(x) == 1])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(out, x.value, atol=1e-6, rtol=0)


def test_project_simplex_output_on_simplex():
    rng = np.random.default_rng(23)
    v = rng.standard_normal(10)
    out = project_simplex(v)
    assert np.all(out >= 0)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)


def test_project_simplex_fixed_point_on_simplex():
    w = np.random.default_rng(29).dirichlet(np.ones(6))
    np.testing.assert_allclose(project_simplex(w), w, atol=1e-12)
