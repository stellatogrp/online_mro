"""Unit tests for the online / batch MRO clustering machinery and the
synthetic data generator in ``portfolio.utils``.

Invariants tested (verified against the algorithm, not assumed):
  * micro (q_dict) and macro (k_dict) weights always sum to 1;
  * membership index lists partition the running sample array (counts
    conserved, no duplicates);
  * every micro/macro centroid equals the exact mean of its member samples,
    and every weight equals member_count / num_samples -- the absorption,
    spawn, merge, and re-cluster updates all preserve exact running means;
  * the whole stream update is deterministic once the global numpy seed is
    fixed (KMeans is called without random_state, so it draws from the
    global legacy RNG).

NOT asserted: stability of cluster *assignments* across different streams or
K-means initializations -- the algorithm does not guarantee that.
"""
import numpy as np

from portfolio.utils import (
    fixed_cluster,
    generate_returns,
    online_cluster_init_online,
    online_cluster_update_online,
    project_simplex,  # noqa: F401  (imported by sibling tests; keep utils warm)
    wasserstein,
)

K, Q, M = 3, 8, 4
N_INIT, STREAM_N = 12, 40


def _run_stream(seed=0, fix_time=10**9):
    """Init on N_INIT points then stream STREAM_N more, one at a time."""
    np.random.seed(seed)  # KMeans(random_state=None) uses the global RNG
    data, lb, ub = generate_returns(N_INIT + STREAM_N, M, seed=11)
    q_dict, k_dict, _ = online_cluster_init_online(K, Q, data[:N_INIT], M)
    num = N_INIT
    for t, i in enumerate(range(N_INIT, N_INIT + STREAM_N)):
        q_dict, k_dict, _ = online_cluster_update_online(
            K, data[i], q_dict, k_dict, num, t, fix_time, M, Q)
        num += 1
    return q_dict, k_dict, num, data


def _check_state(q_dict, k_dict, num, running):
    cur_Q = q_dict["cur_Q"]
    cur_K = int(k_dict["K"])
    # weights are probability vectors
    np.testing.assert_allclose(q_dict["w"][:cur_Q].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(k_dict["w"][:cur_K].sum(), 1.0, atol=1e-12)
    assert np.all(q_dict["w"][:cur_Q] >= 0)
    assert np.all(k_dict["w"][:cur_K] >= 0)
    # membership indices partition {0, ..., num-1} at both levels
    q_idx = [i for q in range(cur_Q) for i in q_dict["idx"][q]]
    k_idx = [i for k in range(cur_K) for i in k_dict["idx"][k]]
    assert sorted(q_idx) == list(range(num))
    assert sorted(k_idx) == list(range(num))
    # centroids are exact means of members; weights are exact counts / num
    for level, cur in ((q_dict, cur_Q), (k_dict, cur_K)):
        for j in range(cur):
            members = np.asarray(level["idx"][j], dtype=int)
            assert members.size > 0
            np.testing.assert_allclose(
                level["d"][j], running[members].mean(axis=0), atol=1e-8)
            np.testing.assert_allclose(
                level["w"][j], members.size / num, atol=1e-12)


def test_init_state_invariants():
    np.random.seed(0)
    data, _, _ = generate_returns(N_INIT, M, seed=11)
    q_dict, k_dict, _ = online_cluster_init_online(K, Q, data, M)
    _check_state(q_dict, k_dict, N_INIT, data)
    assert q_dict["cur_Q"] <= Q
    assert int(k_dict["K"]) == min(K, N_INIT)


def test_stream_update_invariants():
    """After 40 online updates (absorptions, spawns, merges, re-clusters) the
    exact-running-mean and conservation invariants still hold."""
    q_dict, k_dict, num, data = _run_stream(seed=0)
    assert num == N_INIT + STREAM_N
    _check_state(q_dict, k_dict, num, data[:num])
    assert q_dict["cur_Q"] <= Q


def test_stream_update_deterministic_under_seed():
    qa, ka, _, _ = _run_stream(seed=0)
    qb, kb, _, _ = _run_stream(seed=0)
    np.testing.assert_array_equal(qa["d"], qb["d"])
    np.testing.assert_array_equal(qa["w"], qb["w"])
    np.testing.assert_array_equal(ka["d"], kb["d"])
    np.testing.assert_array_equal(ka["w"], kb["w"])
    assert qa["cur_Q"] == qb["cur_Q"]
    for k in range(int(ka["K"])):
        assert list(ka["idx"][k]) == list(kb["idx"][k])


def test_fixed_cluster_conserves_mass_and_counts():
    """The frozen-cluster path (t >= fix_time): each new point is absorbed by
    the nearest macro center; weights stay a probability vector, the absorbed
    centroid stays the exact member mean, and counts are conserved."""
    q_dict, k_dict, num, data = _run_stream(seed=0)
    cur_K = int(k_dict["K"])
    extra, _, _ = generate_returns(6, M, seed=99)
    running = np.vstack([data, extra])
    for j in range(6):
        k_dict, _ = fixed_cluster(k_dict, extra[j], num, M)
        num += 1
        np.testing.assert_allclose(k_dict["w"][:cur_K].sum(), 1.0, atol=1e-12)
        k_idx = [i for k in range(cur_K) for i in k_dict["idx"][k]]
        assert sorted(k_idx) == list(range(num))
        for k in range(cur_K):
            members = np.asarray(k_dict["idx"][k], dtype=int)
            np.testing.assert_allclose(
                k_dict["d"][k], running[members].mean(axis=0), atol=1e-8)


# --------------------------------------------------------------------------- #
# Synthetic data generator
# --------------------------------------------------------------------------- #
def test_generate_returns_deterministic_and_in_box():
    d1, lb1, ub1 = generate_returns(30, 6, seed=5)
    d2, lb2, ub2 = generate_returns(30, 6, seed=5)
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(lb1, lb2)
    np.testing.assert_array_equal(ub1, ub2)
    assert d1.shape == (30, 6)
    assert np.all(lb1 < ub1)
    # data are clipped into the box
    assert np.all(d1 >= lb1[None, :] - 1e-15)
    assert np.all(d1 <= ub1[None, :] + 1e-15)
    # a different seed gives different data
    d3, _, _ = generate_returns(30, 6, seed=6)
    assert not np.array_equal(d1, d3)


def test_generate_returns_box_binds():
    """box_q = 0.40 keeps the central 20% band, so a large fraction of the
    samples must sit exactly on the box boundary (this is what makes the
    box-aware worst-case value differ from the unbounded closed form)."""
    d, lb, ub = generate_returns(200, 5, seed=1)
    on_boundary = np.mean((d == lb[None, :]) | (d == ub[None, :]))
    assert on_boundary > 0.5


# --------------------------------------------------------------------------- #
# Wasserstein helper
# --------------------------------------------------------------------------- #
def test_wasserstein_sanity():
    # two point masses: W1 = Euclidean distance
    p = np.array([[0.0, 0.0]])
    q = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(wasserstein(p, q), 5.0, atol=1e-12)
    # identical empirical distributions: W1 = 0
    samp = np.random.default_rng(1).normal(size=(20, 3))
    np.testing.assert_allclose(wasserstein(samp, samp), 0.0, atol=1e-12)
    # symmetry
    a = np.random.default_rng(2).normal(size=(10, 2))
    b = np.random.default_rng(3).normal(size=(15, 2))
    np.testing.assert_allclose(wasserstein(a, b), wasserstein(b, a), atol=1e-10)
