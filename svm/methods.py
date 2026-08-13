"""Worker functions for the SPARSE-SVM experiment (one per method/solver path).

Provenance: verbatim copies of the per-task worker bodies of the legacy
drivers (branch paper-experiments):

  * ``task_mro_subgrad``  <- ``svm_experiments`` in ``svm1/svm_grad.py``
  * ``task_mro_exact``    <- ``svm_experiments`` in ``svm1/svm_orig.py``
  * ``task_dro_subgrad``  <- ``svm_experiments`` in ``svm1/svm_DRO_grad.py``
  * ``task_dro_exact``    <- ``svm_experiments`` in ``svm1/svm_DRO_orig.py``
  * ``run_true_saa``      <- ``svm1/svm_true_saa.py`` (solve loop + summary)

The only changes vs. the legacy workers:
  * the legacy module-level globals (eps_init, list_inds, m, foldername, ...)
    are passed as explicit keyword arguments;
  * the two ``compute_cumulative_regret`` aliases are disambiguated to
    ``compute_cumulative_regret_online_hinge`` /
    ``compute_cumulative_regret_dro_hinge`` (pure rename);
  * the mid-loop per-task checkpoint CSVs of the *exact* paths (which legacy
    rewrote at every logged step) are gated to every ``checkpoint_every``-th
    logged step.  Those blocks only read loop state and write a CSV -- no
    side effects feed back into the loop -- so the final CSV is identical.

Everything else (computation order, RNG call order, MOSEK parameters,
history keys, DataFrame columns, per-task file names) is unchanged.
"""
import copy
import os
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from .utils_svm import (
    createproblem_hingeMIO,
    create_scenario_hinge,
    evaluate_expected_cost_hinge,
    fixed_cluster,
    label_aware_kmeans,
    load_svm_dataset,
    MOSEK_PARAMS,
    w2_dist,
    wasserstein,
    worst_case_hinge,
)
from .utils_svm import calc_cluster_val_hinge as calc_cluster_val
from .utils_svm import compute_cumulative_regret_online_hinge
from .utils_svm import compute_cumulative_regret_dro_hinge
from .utils_svm import online_cluster_init_online as online_cluster_init
from .utils_svm import online_cluster_update_online as online_cluster_update

output_stream = sys.stdout


# ---------------------------------------------------------------------------
# IHT helpers  (identical in svm1/svm_grad.py and svm1/svm_DRO_grad.py)
# ---------------------------------------------------------------------------

def _iht(beta, k):
    """Project onto {v : ||v||_0 <= k} by zeroing all but the k largest |entries|."""
    if k >= len(beta):
        return beta.copy()
    v = beta.copy()
    v[np.argsort(np.abs(v))[:-k]] = 0.0
    return v


def _q_step(dat, m, beta, delta):
    """Perturb data under worst-case Q*(beta) for p=2 Wasserstein DRO.

    x~_i = x_i - delta y_i beta / ||beta||_2  for active samples (y_i beta^T x_i < 1)
    x~_i = x_i                                for inactive samples
    """
    X = dat[:, :m]
    y = dat[:, m]
    beta_norm = np.linalg.norm(beta, 2)
    dat_tilde = dat.copy()
    if beta_norm < 1e-10:
        return dat_tilde
    active = y * (X @ beta) < 1
    dat_tilde[active, :m] -= delta * np.outer(y[active], beta) / beta_norm
    return dat_tilde


def _iht_grad_step(dat_tilde, m, beta, eta, k, weights=None):
    """One subgradient + IHT step on f_{Q*}(beta) = weighted hinge on perturbed data.

    g = -sum w_i y_i x~_i  (active set under perturbed data)
    beta <- H_k(beta - eta g)
    """
    X_tilde = dat_tilde[:, :m]
    y_tilde = dat_tilde[:, m]
    n = dat_tilde.shape[0]
    active = y_tilde * (X_tilde @ beta) < 1
    if not np.any(active):
        return beta.copy()
    if weights is None:
        g = -np.sum(y_tilde[active, None] * X_tilde[active], axis=0) / n
    else:
        g = -np.sum(weights[active, None] * y_tilde[active, None] * X_tilde[active], axis=0)
    return _iht(beta - eta * g, k)


def _milp_warmstart(dat, m, k, p, radius, weights, beta_hint):
    """Run full MI-SOCP warm-started from beta_hint; return (beta_opt, obj, time) or None."""
    N = dat.shape[0]
    prob, bvar, zvar, dpar, epar, wpar = createproblem_hingeMIO(N, m, k, p=p)
    dpar.value = dat
    epar.value = radius
    wpar.value = weights
    bvar.value = beta_hint
    zvar.value = (np.abs(beta_hint) > 1e-6).astype(float)
    prob.solve(solver=cp.MOSEK, warm_start=True, ignore_dpp=True,
               verbose=False, mosek_params=MOSEK_PARAMS)
    if bvar.value is None:
        return None
    return bvar.value.copy(), float(prob.objective.value), float(prob.solver_stats.solve_time)


# ---------------------------------------------------------------------------
# MRO subgradient-IHT worker  (svm1/svm_grad.py::svm_experiments, verbatim)
# ---------------------------------------------------------------------------

def task_mro_subgrad(r_input, K, T, N_init, full_data, power, p, r_start, *,
                     m, train_size, test_size, eps_init, list_inds, init_ind,
                     Q, k, eta, fixed_time, cluster_interval, interval, t_list,
                     solve_interval, t_solve_list, rmse_mult, foldername):
    """Online MRO subgradient-IHT vs. batch-MRO-IHT vs. cluster-SAA (p=2 hinge).

    Mirrors svm_orig_p1.py: online micro-clusters (k_dict) drive the 'online'
    IHT method; fresh k-means clusters (new_k_dict) drive the 'batch-MRO' IHT
    method; cluster_SAA solves the non-robust best-subset on those same clusters.
    ``full_data`` is a real LIBSVM binary-classification dataset (see
    ``load_svm_dataset``) rather than a synthetic sparse-linear model.

    History keys ('x', 'MRO_x', 'obj_values', 'MRO_obj_values', ...) match
    svm_orig_p1.py so that compute_cumulative_regret_online_hinge is reused.
    """
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            full_data[:, :m + 1], train_size=train_size, test_size=test_size,
            random_state=r_start + r)
        init_eps = eps_init[epsnum]
        num_dat = N_init
        dim = m + 1

        q_dict, k_dict, weight_update_time = online_cluster_init(
            K, Q, dat[init_ind:(init_ind + num_dat)], dim)
        k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict = None
        init_samples = dat[init_ind:(init_ind + N_init)]

        # Running IHT iterates
        x_current = np.zeros(m)       # online-MRO-IHT
        MRO_x_current = np.zeros(m)   # batch-MRO-IHT
        x_prev = np.zeros(m)
        MRO_x_prev = np.zeros(m)
        cluster_SAA_x_current = np.zeros(m)
        init_radius_val = init_eps * (1 / (num_dat ** power))
        cluster_time = 0.0

        history = {
            'x': [],
            'obj_values': [],
            'MRO_x': [],
            'MRO_obj_values': [],
            'cluster_SAA_x': [],
            'cluster_SAA_obj_values': [],
            'worst_values': [],
            'worst_values_MRO': [],
            'epsilon': [],
            'online_computation_times': {
                'weight_update': [],
                'iht_step': [],
                'total_iteration': [],
            },
            'MRO_computation_times': {
                'clustering': [],
                'iht_step': [],
                'total_iteration': [],
            },
            'square_val': [],
            'sig_val': [],
            'square_val_MRO': [],
            'sig_val_MRO': [],
            'satisfy': [],
            'MRO_satisfy': [],
            'worst_times': [],
            'MRO_worst_times': [],
            'MRO_worst_values': [],
            't': [],
            'MRO_worst_values_regret': [],
            'worst_values_regret': [],
            'MRO_worst_times_regret': [],
            'worst_times_regret': [],
            'regret_bound': [],
            'MRO_regret_bound': [],
            'regret_K': [],
            'MRO_regret_K': [],
            'cluster_SAA_time': [],
            # periodic full MI-SOCP solve results
            'solve_x': [],
            'solve_obj_values': [],
            'solve_time_list': [],
            'solve_t': [],
        }

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            radius = init_eps * (1 / (num_dat ** power))
            eta_t = eta / np.sqrt(t + 1)
            running_samples = dat[init_ind:(init_ind + num_dat)]
            cur_K = int(np.minimum(num_dat, K))

            # ---- online MRO: IHT step every t ----
            online_iht_start = time.time()
            online_dat_tilde = _q_step(k_dict['d'][:cur_K], m, x_current, radius)
            x_current = _iht_grad_step(
                online_dat_tilde, m, x_current, eta_t, k,
                weights=k_dict['w'][:cur_K])
            online_iht_time = time.time() - online_iht_start

            # ---- batch MRO: re-cluster if needed, then IHT step ----
            if t <= fixed_time:
                start_clust = time.time()
                cur_K_batch = int(np.minimum(K, num_dat))
                if (new_k_dict is None) or (t % cluster_interval == 0) \
                        or (new_k_dict['d'].shape[0] != cur_K_batch):
                    new_centers, klabels, wk = label_aware_kmeans(
                        running_samples, cur_K_batch, dim)
                else:
                    new_centers = new_k_dict['d']
                    samp_lab = np.round(running_samples[:, dim - 1]).astype(int)
                    cen_lab = np.round(new_centers[:, dim - 1]).astype(int)
                    Dd = cdist(running_samples, new_centers)
                    Dd = np.where(samp_lab[:, None] != cen_lab[None, :], np.inf, Dd)
                    klabels = np.argmin(Dd, axis=1)
                    wk = np.bincount(klabels, minlength=new_centers.shape[0]) / num_dat
                cur_K_batch = new_centers.shape[0]
                cluster_time = (time.time() - start_clust) if K < num_dat else 0.0
                new_k_dict = {
                    'K': cur_K_batch, 'data': {}, 'a': new_centers,
                    'd': new_centers, 'w': wk,
                }
                for kk in range(cur_K_batch):
                    new_k_dict['data'][kk] = running_samples[klabels == kk]

            mro_iht_start = time.time()
            mro_dat_tilde = _q_step(new_k_dict['d'], m, MRO_x_current, radius)
            MRO_x_current = _iht_grad_step(
                mro_dat_tilde, m, MRO_x_current, eta_t, k,
                weights=new_k_dict['w'])
            mro_iht_time = time.time() - mro_iht_start

            # ---- periodic MI-SOCP warm-start ----
            milp_time_online = 0.0
            milp_time_mro = 0.0
            if solve_interval > 0 and ((t > 0 and t % solve_interval == 0) or t in t_solve_list):
                # online-MRO warm-start
                _t0 = time.time()
                res = _milp_warmstart(
                    k_dict['d'][:cur_K], m, k, p, radius,
                    k_dict['w'][:cur_K], x_current)
                milp_time_online = time.time() - _t0
                if res is not None:
                    x_current, sol_obj, sol_time = res
                    history['solve_x'].append(x_current.copy())
                    history['solve_obj_values'].append(sol_obj)
                    history['solve_time_list'].append(sol_time)
                    history['solve_t'].append(t)

                # batch-MRO warm-start
                _t0 = time.time()
                res_mro = _milp_warmstart(
                    new_k_dict['d'], m, k, p, radius,
                    new_k_dict['w'], MRO_x_current)
                milp_time_mro = time.time() - _t0
                if res_mro is not None:
                    MRO_x_current = res_mro[0]

            # ---- ingest new sample, update online clusters ----
            # Save pre-ingest state: new_k_dict was built from these, so
            # calc_cluster_val and create_scenario_hinge for new_k_dict must
            # use running_samples_pre / num_dat_pre to keep cur_K consistent.
            running_samples_pre = running_samples
            num_dat_pre = num_dat
            new_sample = dat[init_ind + num_dat]
            q_dict, k_dict, weight_update_time = online_cluster_update(
                K, new_sample, q_dict, k_dict, num_dat, t, fixed_time,
                dim, Q, rmse_mult, cluster_interval=cluster_interval)
            if t >= fixed_time:
                new_k_dict, cluster_time = fixed_cluster(
                    new_k_dict, new_sample, num_dat=num_dat, m=dim)
            num_dat += 1
            running_samples = dat[init_ind:(init_ind + num_dat)]

            # ---- logging (mirrors reg_orig_p1.py cadence) ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):

                    min_obj = worst_case_hinge(
                        k_dict['d'][:cur_K], m, x_current, radius,
                        weights=k_dict['w'][:cur_K], p=p)
                    MRO_min_obj = worst_case_hinge(
                        new_k_dict['d'], m, MRO_x_current, radius,
                        weights=new_k_dict['w'], p=p)
                    square_val_mro, sig_val_mro = calc_cluster_val(
                        K, new_k_dict, num_dat_pre, MRO_x_current,
                        running_samples_pre, m)[1:]

                    # cluster_SAA: non-robust MI best-subset on same centroids
                    cs_prob, cs_x, cs_z = create_scenario_hinge(
                        new_k_dict['d'], m, num_dat_pre, k,
                        weights=new_k_dict['w'])
                    cs_prob.solve(solver=cp.MOSEK, ignore_dpp=True,
                                  verbose=False, mosek_params=MOSEK_PARAMS)
                    cluster_SAA_x_current = cs_x.value
                    cluster_SAA_obj = cs_prob.objective.value
                    cluster_SAA_min_time = cs_prob.solver_stats.solve_time

                    history['online_computation_times']['iht_step'].append(online_iht_time)
                    history['online_computation_times']['total_iteration'].append(
                        online_iht_time + weight_update_time + milp_time_online)
                    history['online_computation_times']['weight_update'].append(weight_update_time)
                    history['t'].append(t)

                    history['MRO_computation_times']['iht_step'].append(mro_iht_time)
                    history['MRO_computation_times']['total_iteration'].append(
                        mro_iht_time + cluster_time + milp_time_mro)
                    history['MRO_computation_times']['clustering'].append(cluster_time)
                    history['cluster_SAA_time'].append(cluster_SAA_min_time + cluster_time)

                    # worst-case hinge on full running data
                    wc_start = time.time()
                    new_worst = worst_case_hinge(running_samples, m, x_current, radius, p=p)
                    worst_time = time.time() - wc_start
                    history['worst_values'].append(new_worst)
                    history['worst_times'].append(worst_time)

                    wc_start = time.time()
                    new_worst_MRO = worst_case_hinge(
                        running_samples, m, MRO_x_current, radius, p=p)
                    MRO_worst_time = time.time() - wc_start
                    history['MRO_worst_values'].append(new_worst_MRO)
                    history['MRO_worst_times'].append(MRO_worst_time)

                    square_val, sig_val = calc_cluster_val(
                        K, k_dict, num_dat, x_current, running_samples, m)[1:]

                    # worst-case on previous iterates for regret
                    wc_start = time.time()
                    new_worst_reg = worst_case_hinge(
                        running_samples, m, x_prev, radius, p=p)
                    worst_time_reg = time.time() - wc_start
                    history['worst_values_regret'].append(new_worst_reg)
                    history['worst_times_regret'].append(worst_time_reg)

                    wc_start = time.time()
                    new_worst_MRO_reg = worst_case_hinge(
                        running_samples, m, MRO_x_prev, radius, p=p)
                    MRO_worst_time_reg = time.time() - wc_start
                    history['MRO_worst_values_regret'].append(new_worst_MRO_reg)
                    history['MRO_worst_times_regret'].append(MRO_worst_time_reg)

                    MRO_x_prev = MRO_x_current.copy()
                    x_prev = x_current.copy()

                    # regret bound bookkeeping
                    N_dist_cur = wasserstein(init_samples, running_samples)
                    history['regret_K'].append(w2_dist(k_dict, k_dict_prev, dim))
                    history['MRO_regret_K'].append(w2_dist(new_k_dict, new_k_dict_prev, dim))
                    regret_bound = (
                        np.sum(history['regret_K']) + N_dist_cur + init_radius_val - radius
                    ) / (t + 1)
                    MRO_regret_bound = (
                        np.sum(history['MRO_regret_K']) + N_dist_cur + init_radius_val - radius
                    ) / (t + 1)
                    history['regret_bound'].append(regret_bound)
                    history['MRO_regret_bound'].append(MRO_regret_bound)
                    k_dict_prev = copy.deepcopy(k_dict)
                    new_k_dict_prev = copy.deepcopy(new_k_dict)

                    history['sig_val'].append(sig_val)
                    history['square_val'].append(square_val)
                    history['sig_val_MRO'].append(sig_val_mro)
                    history['square_val_MRO'].append(square_val_mro)
                    history['x'].append(x_current.copy())
                    history['obj_values'].append(min_obj)
                    history['MRO_x'].append(MRO_x_current.copy())
                    history['MRO_obj_values'].append(MRO_min_obj)
                    history['cluster_SAA_x'].append(cluster_SAA_x_current)
                    history['cluster_SAA_obj_values'].append(cluster_SAA_obj)
                    history['epsilon'].append(radius)

                    print(f"Current delta: {radius}")

        # ---- final CSV ----
        MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s = \
            compute_cumulative_regret_online_hinge(history, dateval, m)

        df = pd.DataFrame({
            'obj_values': np.array(history['obj_values']),
            'MRO_obj_values': np.array(history['MRO_obj_values']),
            'cluster_SAA_obj_values': np.array(history['cluster_SAA_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'online_time': np.array(history['online_computation_times']['total_iteration']),
            'MRO_time': np.array(history['MRO_computation_times']['total_iteration']),
            'cluster_SAA_time': np.array(history['cluster_SAA_time']),
            'MRO_square_val': np.array(history['square_val_MRO']),
            'MRO_sig_val': np.array(history['sig_val_MRO']),
            'square_val': np.array(history['square_val']),
            'sig_val': np.array(history['sig_val']),
            'worst_values': np.array(history['worst_values']),
            'MRO_worst_values': np.array(history['MRO_worst_values']),
            'worst_times': np.array(history['worst_times']),
            'MRO_worst_times': np.array(history['MRO_worst_times']),
            'worst_values_regret': np.array(history['worst_values_regret']),
            'MRO_worst_values_regret': np.array(history['MRO_worst_values_regret']),
            'worst_times_regret': np.array(history['worst_times_regret']),
            'MRO_worst_times_regret': np.array(history['MRO_worst_times_regret']),
            't': np.array(history['t']),
            'regret_bound': history['regret_bound'],
            'MRO_regret_bound': history['MRO_regret_bound'],
        })
        colnames = ['MRO_eval', 'MRO_satisfy', 'O_eval', 'O_satisfy',
                    'O_worst_satisfy', 'MRO_worst_satisfy',
                    'cluster_SAA_eval', 'cluster_SAA_satisfy']
        colvals = [MRO_e, MRO_s, online_e, online_s,
                   online_ws, MRO_ws, CSA_e, CSA_s]
        for i in range(len(colnames)):
            for j in range(2):
                df[colnames[i] + str(j)] = np.array(colvals[i][j])
        df.to_csv(foldername + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        return df

    except Exception as e:
        import traceback
        print(f"Exception in svm_experiments (r_input={locals().get('r_input', None)}): {e}",
              file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


# ---------------------------------------------------------------------------
# MRO exact (MOSEK MILP) worker  (svm1/svm_orig.py::svm_experiments, verbatim)
# ---------------------------------------------------------------------------

def task_mro_exact(r_input, K, T, N_init, full_data, power, p, r_start, *,
                   m, train_size, test_size, eps_init, list_inds, init_ind,
                   Q, k, fixed_time, cluster_interval, interval, t_list,
                   rmse_mult, foldername, checkpoint_every):
    """Online mean-robust DRO sparse-SVM vs. batch (kmeans) MRO-SVM (p = 1).

    Real-data sibling of ``regression/reg_orig_p1.py``: same p=1 DRO sparse-SVM
    mixed-integer LP (``createproblem_hingeMIO``, solved with MOSEK) and same
    online/batch clustering machinery, but ``full_data`` is a real LIBSVM
    binary-classification dataset (see ``load_svm_dataset``) rather than a
    synthetic sparse-linear model. Data points are the joint (x, y) vectors of
    dimension ``m + 1`` (label y in {-1, +1}) clustered online / by kmeans; the
    worst-case expected hinge loss of a fixed beta has the p=1 closed form
    ``worst_case_hinge`` = empirical hinge + delta ||beta||_1.

    A third method, ``cluster_SAA``, reuses the batch-MRO kmeans clusters but
    solves the *non-robust* (delta=0) best subset on those weighted centroids,
    so it isolates the effect of clustering from the distributional robustness.
    """
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            full_data[:, :m + 1], train_size=train_size, test_size=test_size,
            random_state=r_start + r)
        init_eps = eps_init[epsnum]
        num_dat = N_init
        dim = m + 1  # joint (covariate, label) dimension used for clustering

        q_dict, k_dict, weight_update_time = online_cluster_init(K, Q, dat[init_ind:(init_ind + num_dat)], dim)
        k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict = None
        init_samples = dat[init_ind:(init_ind + N_init)]

        # Initialize solutions (beta vectors)
        MRO_x_prev = np.zeros(m)
        x_prev = np.zeros(m)
        # Pre-seed iterates so a solver failure at the first interval still
        # leaves x_current / MRO_x_current well defined for the worst-case eval.
        x_current = np.zeros(m)
        MRO_x_current = np.zeros(m)
        cluster_SAA_x_current = np.zeros(m)
        init_radius_val = init_eps * (1 / (num_dat**(power)))

        history = {
            'x': [],
            'obj_values': [],
            'MRO_x': [],
            'MRO_obj_values': [],
            'worst_values': [],
            'worst_values_MRO': [],
            'epsilon': [],
            'weights': [],
            'weights_q': [],
            'online_computation_times': {
                'weight_update': [],
                'min_problem': [],
                'total_iteration': []
            },
            'MRO_computation_times': {
                'clustering': [],
                'min_problem': [],
                'total_iteration': []
            },
            'distances': [],
            'square_val': [],
            'sig_val': [],
            'square_val_MRO': [],
            'sig_val_MRO': [],
            "satisfy": [],
            "MRO_satisfy": [],
            'worst_times': [],
            'MRO_worst_times': [],
            'MRO_worst_values': [],
            't': [],
            'MRO_weights': [],
            'MRO_worst_values_regret': [],
            'worst_values_regret': [],
            'MRO_worst_times_regret': [],
            'worst_times_regret': [],
            'regret_bound': [],
            'MRO_regret_bound': [],
            'regret_K': [],
            'MRO_regret_K': [],
            'cluster_SAA_x': [],
            'cluster_SAA_obj_values': [],
            'cluster_SAA_time': []
        }

        n_logged = 0  # logged-step counter for checkpoint gating (new)

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            # radius represents delta (order-1); RWPI: delta_n ~ 1/sqrt(n).
            radius = init_eps * (1 / (num_dat**(power)))
            running_samples = dat[init_ind:(init_ind + num_dat)]

            # ---- solve online MRO sparse-SVM best-subset problem ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    cur_K = int(np.minimum(num_dat, K))
                    online_problem, online_x, online_z, data_train, eps_train, w_train = createproblem_hingeMIO(cur_K, m, k, p=p)
                    data_train.value = k_dict['d'][:cur_K]
                    eps_train.value = radius
                    w_train.value = k_dict['w'][:cur_K]

                    online_problem.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
                                         mosek_params=MOSEK_PARAMS)
                    x_current = online_x.value
                    min_obj = online_problem.objective.value
                    min_time = online_problem.solver_stats.solve_time

                    history['online_computation_times']['min_problem'].append(min_time)
                    history['online_computation_times']['total_iteration'].append(min_time + weight_update_time)
                    history['online_computation_times']['weight_update'].append(weight_update_time)
                    history['t'].append(t)

            # ---- solve batch MRO sparse-SVM with fresh kmeans clusters ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    if t <= fixed_time:
                        start_time = time.time()
                        cur_K = int(np.minimum(K, num_dat))
                        # label-pure batch k-means: the cur_K budget is split
                        # across the two labels, each clustered separately, so
                        # every centroid is label-pure and the total is cur_K.
                        if (new_k_dict is None) or (t % cluster_interval == 0) \
                                or (new_k_dict['d'].shape[0] != cur_K):
                            new_centers, klabels, wk = label_aware_kmeans(running_samples, cur_K, dim)
                        else:
                            # cheap update: assign each datapoint to its nearest
                            # same-label center from the last full re-cluster.
                            new_centers = new_k_dict['d']
                            samp_lab = np.round(running_samples[:, dim - 1]).astype(int)
                            cen_lab = np.round(new_centers[:, dim - 1]).astype(int)
                            Dd = cdist(running_samples, new_centers)
                            Dd = np.where(samp_lab[:, None] != cen_lab[None, :], np.inf, Dd)
                            klabels = np.argmin(Dd, axis=1)
                            wk = np.bincount(klabels, minlength=new_centers.shape[0]) / num_dat
                        cur_K = new_centers.shape[0]
                        cluster_time = (time.time() - start_time) if K < num_dat else 0.0
                        new_k_dict = {}
                        new_k_dict['K'] = cur_K
                        new_k_dict['data'] = {}
                        new_k_dict['a'] = new_centers
                        new_k_dict['d'] = new_centers
                        new_k_dict['w'] = wk
                        for kk in range(cur_K):
                            new_k_dict['data'][kk] = running_samples[klabels == kk]

                    cur_K_mro = new_k_dict['d'].shape[0]
                    MRO_problem, MRO_x, MRO_z, MRO_data_train, MRO_eps_train, MRO_w_train = createproblem_hingeMIO(cur_K_mro, m, k, p=p)
                    MRO_data_train.value = new_k_dict['d']
                    MRO_w_train.value = new_k_dict['w']
                    MRO_eps_train.value = radius
                    MRO_problem.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
                                      mosek_params=MOSEK_PARAMS)
                    MRO_x_current = MRO_x.value
                    MRO_min_obj = MRO_problem.objective.value
                    MRO_min_time = MRO_problem.solver_stats.solve_time
                    square_val_mro, sig_val_mro = calc_cluster_val(K, new_k_dict, num_dat, MRO_x_current, running_samples, m)[1:]

                    # cluster_SAA: non-robust BSS-SVM on the same weighted kmeans
                    # centroids (delta = 0), so it shares the clustering cost.
                    cs_prob, cs_x, cs_z = create_scenario_hinge(new_k_dict['d'], m, num_dat, k, weights=new_k_dict['w'])
                    cs_prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
                                  mosek_params=MOSEK_PARAMS)
                    cluster_SAA_x_current = cs_x.value
                    cluster_SAA_obj = cs_prob.objective.value
                    cluster_SAA_min_time = cs_prob.solver_stats.solve_time

                    history['MRO_computation_times']['min_problem'].append(MRO_min_time)
                    history['MRO_computation_times']['total_iteration'].append(MRO_min_time + cluster_time)
                    history['MRO_computation_times']['clustering'].append(cluster_time)
                    history['MRO_weights'].append(new_k_dict['w'])
                    history['cluster_SAA_time'].append(cluster_SAA_min_time + cluster_time)

            # ---- online MRO worst-case hinge (wrt full, non-clustered data) ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    wc_start = time.time()
                    new_worst = worst_case_hinge(running_samples, m, x_current, radius, p=p)
                    worst_time = time.time() - wc_start
                    history['worst_values'].append(new_worst)
                    history['worst_times'].append(worst_time)

                    if t <= T or (t in t_list):
                        wc_start = time.time()
                        new_worst_MRO = worst_case_hinge(running_samples, m, MRO_x_current, radius, p=p)
                        MRO_worst_time = time.time() - wc_start

                        square_val, sig_val = calc_cluster_val(K, k_dict, num_dat, x_current, running_samples, m)[1:]

                        history['MRO_worst_values'].append(new_worst_MRO)
                        history['MRO_worst_times'].append(MRO_worst_time)

            # ---- worst-case hinge wrt previous-stage solutions (for regret) ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    wc_start = time.time()
                    new_worst = worst_case_hinge(running_samples, m, x_prev, radius, p=p)
                    worst_time = time.time() - wc_start
                    history['worst_values_regret'].append(new_worst)
                    history['worst_times_regret'].append(worst_time)

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    wc_start = time.time()
                    new_worst_MRO = worst_case_hinge(running_samples, m, MRO_x_prev, radius, p=p)
                    MRO_worst_time = time.time() - wc_start
                    history['MRO_worst_values_regret'].append(new_worst_MRO)
                    history['MRO_worst_times_regret'].append(MRO_worst_time)

                    MRO_x_prev = MRO_x_current
                    x_prev = x_current

            # ---- ingest new sample, update clusters ----
            new_sample = dat[init_ind + num_dat]
            q_dict, k_dict, weight_update_time = online_cluster_update(K, new_sample, q_dict, k_dict, num_dat, t, fixed_time, dim, Q, rmse_mult, cluster_interval=cluster_interval)
            if t >= fixed_time:
                new_k_dict, cluster_time = fixed_cluster(new_k_dict, new_sample, num_dat=num_dat, m=dim)
            num_dat += 1

            # ---- regret bound bookkeeping ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    N_dist_cur = wasserstein(init_samples, running_samples)

                    history['regret_K'].append(w2_dist(k_dict, k_dict_prev, dim))
                    history['MRO_regret_K'].append(w2_dist(new_k_dict, new_k_dict_prev, dim))
                    regret_bound = (np.sum(history['regret_K']) + N_dist_cur + init_radius_val - radius) / (t + 1)
                    MRO_regret_bound = (np.sum(history['MRO_regret_K']) + N_dist_cur + init_radius_val - radius) / (t + 1)
                    history["regret_bound"].append(regret_bound)
                    history["MRO_regret_bound"].append(MRO_regret_bound)
                    k_dict_prev = copy.deepcopy(k_dict)
                    new_k_dict_prev = copy.deepcopy(new_k_dict)

                    history['sig_val'].append(sig_val)
                    history['square_val'].append(square_val)
                    history['sig_val_MRO'].append(sig_val_mro)
                    history['square_val_MRO'].append(square_val_mro)
                    history['x'].append(x_current)
                    history['obj_values'].append(min_obj)
                    history['MRO_x'].append(MRO_x_current)
                    history['MRO_obj_values'].append(MRO_min_obj)
                    history['cluster_SAA_x'].append(cluster_SAA_x_current)
                    history['cluster_SAA_obj_values'].append(cluster_SAA_obj)
                    history['weights'].append(k_dict['w'].copy())
                    history['weights_q'].append(q_dict['w'].copy())
                    history['epsilon'].append(radius)

                    print(f"Current delta: {radius}")

            # ---- write running CSV (checkpoint) ----
            # Legacy rewrote this per-task checkpoint CSV at EVERY logged step.
            # The block only reads loop state (history, dateval) and writes a
            # CSV -- nothing feeds back into the loop -- so it is gated to
            # every ``checkpoint_every``-th logged step.  The final CSV written
            # after the loop is identical either way.
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    n_logged += 1
                    if n_logged % checkpoint_every == 0:

                        MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s = compute_cumulative_regret_online_hinge(
                            history, dateval, m)

                        df = pd.DataFrame({
                            'obj_values': np.array(history['obj_values']),
                            'MRO_obj_values': np.array(history['MRO_obj_values']),
                            'cluster_SAA_obj_values': np.array(history['cluster_SAA_obj_values']),
                            'epsilon': np.array(history['epsilon']),
                            'weights': history['weights'],
                            'MRO_weights': history['MRO_weights'],
                            'online_time': np.array(history['online_computation_times']['total_iteration']),
                            'MRO_time': np.array(history['MRO_computation_times']['total_iteration']),
                            'cluster_SAA_time': np.array(history['cluster_SAA_time']),
                            'MRO_square_val': np.array(history['square_val_MRO']),
                            'MRO_sig_val': np.array(history['sig_val_MRO']),
                            'square_val': np.array(history['square_val']),
                            'sig_val': np.array(history['sig_val']),
                            "worst_values": np.array(history['worst_values']),
                            "MRO_worst_values": np.array(history['MRO_worst_values']),
                            "worst_times": np.array(history['worst_times']),
                            "MRO_worst_times": np.array(history['MRO_worst_times']),
                            "worst_values_regret": np.array(history['worst_values_regret']),
                            "MRO_worst_values_regret": np.array(history['MRO_worst_values_regret']),
                            "worst_times_regret": np.array(history['worst_times_regret']),
                            "MRO_worst_times_regret": np.array(history['MRO_worst_times_regret']),
                            't': np.array(history['t']),
                            'regret_bound': history["regret_bound"],
                            'MRO_regret_bound': history["MRO_regret_bound"]
                        })
                        colnames = ['MRO_eval', "MRO_satisfy", 'O_eval', "O_satisfy", "O_worst_satisfy", "MRO_worst_satisfy", "cluster_SAA_eval", "cluster_SAA_satisfy"]
                        colvals = [MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s]
                        for i in range(len(colnames)):
                            for j in range(2):
                                df[colnames[i] + str(j)] = np.array(colvals[i][j])
                        df.to_csv(foldername + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s = compute_cumulative_regret_online_hinge(
            history, dateval, m)

        df = pd.DataFrame({
            'obj_values': np.array(history['obj_values']),
            'MRO_obj_values': np.array(history['MRO_obj_values']),
            'cluster_SAA_obj_values': np.array(history['cluster_SAA_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'weights': history['weights'],
            'MRO_weights': history['MRO_weights'],
            'online_time': np.array(history['online_computation_times']['total_iteration']),
            'MRO_time': np.array(history['MRO_computation_times']['total_iteration']),
            'cluster_SAA_time': np.array(history['cluster_SAA_time']),
            'MRO_square_val': np.array(history['square_val_MRO']),
            'MRO_sig_val': np.array(history['sig_val_MRO']),
            'square_val': np.array(history['square_val']),
            'sig_val': np.array(history['sig_val']),
            "worst_values": np.array(history['worst_values']),
            "MRO_worst_values": np.array(history['MRO_worst_values']),
            "worst_times": np.array(history['worst_times']),
            "MRO_worst_times": np.array(history['MRO_worst_times']),
            "worst_values_regret": np.array(history['worst_values_regret']),
            "MRO_worst_values_regret": np.array(history['MRO_worst_values_regret']),
            "worst_times_regret": np.array(history['worst_times_regret']),
            "MRO_worst_times_regret": np.array(history['MRO_worst_times_regret']),
            't': np.array(history['t']),
            'regret_bound': history["regret_bound"],
            'MRO_regret_bound': history["MRO_regret_bound"]
        })
        colnames = ['MRO_eval', "MRO_satisfy", 'O_eval', "O_satisfy", "O_worst_satisfy", "MRO_worst_satisfy", "cluster_SAA_eval", "cluster_SAA_satisfy"]
        colvals = [MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s]
        for i in range(len(colnames)):
            for j in range(2):
                df[colnames[i] + str(j)] = np.array(colvals[i][j])
        df.to_csv(foldername + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        return df
    except Exception as e:
        import traceback
        print(f"Exception in svm_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


# ---------------------------------------------------------------------------
# DRO subgradient-IHT worker  (svm1/svm_DRO_grad.py::svm_experiments, verbatim)
# ---------------------------------------------------------------------------

def task_dro_subgrad(r_input, T, N_init, full_data, power, r_start, p, *,
                     m, train_size, test_size, eps_init, list_inds, init_ind,
                     k, eta, solve_interval, t_solve_list, interval, t_list,
                     newfoldername):
    """Full-data DRO-IHT sparse-SVM (p=2 hinge).

    DRO-IHT runs an alternating Q-step / beta-step at every timestep,
    accumulating data as in the streaming setting. ``full_data`` is a real
    LIBSVM binary-classification dataset (see ``load_svm_dataset``) rather
    than a synthetic sparse-linear model.

    History keys match svm_DRO_orig_p1.py ('DRO_x', ...).
    """
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            full_data[:, :m + 1], train_size=train_size, test_size=test_size,
            random_state=r_start + r)

        init_eps = eps_init[epsnum]
        num_dat = N_init

        DRO_x_current = np.zeros(m)   # DRO-IHT running iterate

        history = {
            'DRO_x': [],
            'DRO_obj_values': [],
            'epsilon': [],
            'DRO_computation_times': {'total_iteration': [], 'iht_step': []},
            't': [],
            # periodic full MI-SOCP results
            'solve_x': [],
            'solve_obj_values': [],
            'solve_time_list': [],
            'solve_t': [],
        }

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            radius = init_eps * (1 / (num_dat ** power))
            eta_t = eta / np.sqrt(t + 1)
            running_samples = dat[init_ind:(init_ind + num_dat)]
            n = num_dat

            # ---- DRO-IHT: Q-step + beta-step every t ----
            iht_start = time.time()
            dat_tilde = _q_step(running_samples, m, DRO_x_current, radius)
            DRO_x_current = _iht_grad_step(
                dat_tilde, m, DRO_x_current, eta_t, k, weights=None)
            iht_time = time.time() - iht_start

            # ---- periodic full MI-SOCP warm-start ----
            milp_time = 0.0
            if solve_interval > 0 and ((t > 0 and t % solve_interval == 0) or t in t_solve_list):
                weights_uniform = np.ones(n) / n
                _t0 = time.time()
                res = _milp_warmstart(
                    running_samples, m, k, p, radius, weights_uniform, DRO_x_current)
                milp_time = time.time() - _t0
                if res is not None:
                    DRO_x_current, sol_obj, sol_time = res
                    history['solve_x'].append(DRO_x_current.copy())
                    history['solve_obj_values'].append(sol_obj)
                    history['solve_time_list'].append(sol_time)
                    history['solve_t'].append(t)

            # New sample
            num_dat += 1

            # ---- logging at interval steps ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):

                    DRO_min_obj = worst_case_hinge(
                        running_samples, m, DRO_x_current, radius, p=p)

                    history['DRO_computation_times']['iht_step'].append(iht_time)
                    history['DRO_computation_times']['total_iteration'].append(iht_time + milp_time)
                    history['DRO_x'].append(DRO_x_current.copy())
                    history['DRO_obj_values'].append(DRO_min_obj)
                    history['epsilon'].append(radius)
                    history['t'].append(t)

        # ---- final CSV ----
        T_hist = len(history['t'])
        DRO_eval, DRO_satisfy = [], []
        for j in range(2):
            eval_samples = dateval[(j * 200):(j + 1) * 200, :m + 1]
            eval_values = np.array([
                evaluate_expected_cost_hinge(eval_samples, m, history['DRO_x'][t])
                for t in range(T_hist)
            ])
            DRO_eval.append(eval_values)
            DRO_satisfy.append(
                (np.array(history['DRO_obj_values']) >= eval_values).astype(float))

        df = pd.DataFrame({
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time': np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval1': DRO_eval[0],
            'DRO_eval2': DRO_eval[1],
            'DRO_satisfy1': DRO_satisfy[0],
            'DRO_satisfy2': DRO_satisfy[1],
            't': np.array(history['t']),
        })
        df.to_csv(newfoldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        return df

    except Exception as e:
        import traceback
        print(f"Exception in svm_experiments (r_input={locals().get('r_input', None)}): {e}",
              file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


# ---------------------------------------------------------------------------
# DRO exact (MOSEK MILP) worker  (svm1/svm_DRO_orig.py::svm_experiments, verbatim)
# ---------------------------------------------------------------------------

def task_dro_exact(r_input, T, N_init, full_data, power, r_start, p, *,
                   m, train_size, test_size, eps_init, list_inds, init_ind,
                   k, interval, interval_SAA, t_list, newfoldername,
                   checkpoint_every):
    """Full-data DRO sparse-SVM vs. sample-average (SAA) SVM (p = 1, hinge loss).

    Real-data sibling of ``regression/reg_DRO_orig_p1.py``: same p=1 DRO
    sparse-SVM mixed-integer LP (``createproblem_hingeMIO``) solved with
    MOSEK, and the same non-robust cardinality-constrained SAA hinge
    minimization (``create_scenario_hinge``), but ``full_data`` is a real
    LIBSVM binary-classification dataset (see ``load_svm_dataset``) rather
    than a synthetic sparse-linear model.
    """
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            full_data[:, :m + 1], train_size=train_size, test_size=test_size,
            random_state=r_start + r)

        init_eps = eps_init[epsnum]
        num_dat = N_init

        # Pre-seed iterates so a solver failure at the first interval still
        # leaves them defined for the history append.
        DRO_x_current = np.zeros(m)
        SA_x_current = np.zeros(m)

        history = {
            'DRO_x': [],
            'DRO_obj_values': [],
            'epsilon': [],
            'DRO_computation_times': {'total_iteration': []},
            'SA_computation_times': [],
            'SA_obj_values': [],
            'SA_x': [],
            't': [],
        }

        n_logged = 0  # logged-step counter for checkpoint gating (new)

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            # radius represents delta (order-1); RWPI picks delta_n ~ 1/sqrt(n)
            # so the induced ell_1 penalty is statistically principled.
            radius = init_eps * (1 / (num_dat**power))
            running_samples = dat[init_ind:(init_ind + num_dat)]

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    # solve full-data DRO sparse-SVM best subset (MILP via MOSEK)
                    DRO_problem, DRO_x, DRO_z, DRO_data, DRO_eps, DRO_w = createproblem_hingeMIO(num_dat, m, k, p=p)
                    DRO_data.value = running_samples
                    DRO_w.value = (1 / num_dat) * np.ones(num_dat)
                    DRO_eps.value = radius
                    DRO_problem.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
                                      mosek_params=MOSEK_PARAMS)
                    DRO_x_current = DRO_x.value
                    DRO_min_obj = DRO_problem.objective.value
                    DRO_min_time = DRO_problem.solver_stats.solve_time

            if t % interval_SAA == 0 or ((t - 1) % interval_SAA == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    # SAA (delta=0) doesn't depend on epsilon, and running_samples/
                    # num_dat only depend on r (train_test_split uses
                    # random_state=r_start+r), so the SAA solution is identical
                    # across every epsnum for a given (r, t) -- solve it only
                    # once, for the first eps_init value, and skip the redundant
                    # MOSEK call for every other epsnum.
                    if epsnum == 0:
                        s_prob, s_x, s_z = create_scenario_hinge(running_samples, m, num_dat, k)
                        s_prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
                                     mosek_params=MOSEK_PARAMS)
                        SA_x_current = s_x.value
                        SA_obj_current = s_prob.objective.value
                        SA_time = s_prob.solver_stats.solve_time
                    else:
                        SA_x_current = np.full(m, np.nan)
                        SA_obj_current = np.nan
                        SA_time = np.nan

                    history['DRO_computation_times']['total_iteration'].append(DRO_min_time)
                    history['DRO_x'].append(DRO_x_current)
                    history['DRO_obj_values'].append(DRO_min_obj)
                    history['epsilon'].append(radius)
                    history['t'].append(t)

                    history['SA_computation_times'].append(SA_time)
                    history['SA_x'].append(SA_x_current)
                    history['SA_obj_values'].append(SA_obj_current)

            # New sample
            num_dat += 1

            # ---- write running CSV (checkpoint) ----
            # Legacy rewrote this per-task checkpoint CSV at EVERY logged step.
            # The block only reads loop state (history, dateval) and writes a
            # CSV -- nothing feeds back into the loop -- so it is gated to
            # every ``checkpoint_every``-th logged step.  The final CSV written
            # after the loop is identical either way.
            if t % interval_SAA == 0 or ((t - 1) % interval_SAA == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    n_logged += 1
                    if n_logged % checkpoint_every == 0:

                        DRO_eval, DRO_satisfy, SA_eval, SA_satisfy = compute_cumulative_regret_dro_hinge(
                            history, dateval, m)

                        df = pd.DataFrame({
                            'DRO_obj_values': np.array(history['DRO_obj_values']),
                            'epsilon': np.array(history['epsilon']),
                            'DRO_time': np.array(history['DRO_computation_times']['total_iteration']),
                            'DRO_eval1': DRO_eval[0],
                            'DRO_eval2': DRO_eval[1],
                            "DRO_satisfy1": DRO_satisfy[0],
                            "DRO_satisfy2": DRO_satisfy[1],
                            'SA_eval1': SA_eval[0],
                            'SA_eval2': SA_eval[1],
                            'SA_satisfy1': SA_satisfy[0],
                            'SA_satisfy2': SA_satisfy[1],
                            'SA_obj_values': np.array(history['SA_obj_values']),
                            'SA_time': np.array(history['SA_computation_times']),
                            't': np.array(history['t'])
                        })
                        df.to_csv(newfoldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        DRO_eval, DRO_satisfy, SA_eval, SA_satisfy = compute_cumulative_regret_dro_hinge(
            history, dateval, m)

        df = pd.DataFrame({
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time': np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval1': DRO_eval[0],
            'DRO_eval2': DRO_eval[1],
            "DRO_satisfy1": DRO_satisfy[0],
            "DRO_satisfy2": DRO_satisfy[1],
            'SA_eval1': SA_eval[0],
            'SA_eval2': SA_eval[1],
            'SA_satisfy1': SA_satisfy[0],
            'SA_satisfy2': SA_satisfy[1],
            'SA_obj_values': np.array(history['SA_obj_values']),
            'SA_time': np.array(history['SA_computation_times']),
            't': np.array(history['t'])
        })
        df.to_csv(newfoldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        return df
    except Exception as e:
        import traceback
        print(f"Exception in svm_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


# ---------------------------------------------------------------------------
# Oracle SAA baseline  (svm1/svm_true_saa.py, verbatim)
# ---------------------------------------------------------------------------

def solve_true_saa(full_data, m, N_true, k, r, r_start, test_size, train_size):
    """Solve SAA best-subset SVM on the first N_true samples for seed r.

    Uses the same train/test split convention as svm_orig.py / svm_grad.py so
    that dateval is identical to the one used during online experiments.

    Returns a dict with keys:
        r, insample_obj, outsample_eval0, outsample_eval1,
        solve_time, converged
    """
    dat, dateval = train_test_split(
        full_data[:, :m + 1], train_size=train_size, test_size=test_size,
        random_state=r_start + r)

    # Take the first N_true points in the same order the online experiments
    # stream them (init_ind=0).  This is the oracle training set.
    dat_true = dat[:N_true]

    t0 = time.time()
    prob, beta_var, z_var = create_scenario_hinge(dat_true, m, N_true, k)
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
               mosek_params=MOSEK_PARAMS)
    solve_time = time.time() - t0

    if beta_var.value is None:
        print(f"  r={r}: solver failed (status: {prob.status})")
        return {
            'r': r,
            'insample_obj': np.nan,
            'outsample_eval0': np.nan,
            'outsample_eval1': np.nan,
            'solve_time': solve_time,
            'converged': False,
        }

    beta = beta_var.value
    insample = float(prob.objective.value)
    # Two disjoint 200-sample windows -- same windows compute_cumulative_regret uses.
    oos0 = evaluate_expected_cost_hinge(dateval[0:200],   m, beta)
    oos1 = evaluate_expected_cost_hinge(dateval[200:400], m, beta)

    print(f"  r={r}: in-sample={insample:.5f}  oos0={oos0:.5f}  oos1={oos1:.5f}"
          f"  time={solve_time:.1f}s")
    return {
        'r': r,
        'insample_obj': insample,
        'outsample_eval0': oos0,
        'outsample_eval1': oos1,
        'solve_time': solve_time,
        'converged': True,
    }


def run_true_saa(foldername, dataset, k, N_true, R, r_start):
    """Oracle SAA sparse-SVM (svm1/svm_true_saa.py ``__main__`` body, verbatim).

    Solves the cardinality-constrained best-subset hinge SVM on N_true
    training samples for seeds r=0..R-1, using the SAME train/test split as
    the online experiments (same random_state=r_start+r).  Saves:

      {foldername}/true_saa_R{r}.csv    -- per-seed results
      {foldername}/true_saa_all.csv     -- all seeds stacked
      {foldername}/true_saa_summary.csv -- mean/q25/q75 across seeds
    """
    os.makedirs(foldername, exist_ok=True)

    full_data, m = load_svm_dataset(dataset)
    n_total = full_data.shape[0]
    test_size = max(1000, n_total // 6)
    train_size = n_total - test_size
    N_true = min(N_true, train_size)

    print(f"Dataset: {dataset}  n_total={n_total}  m={m}")
    print(f"train_size={train_size}  test_size={test_size}  N_true={N_true}  k={k}")
    print(f"Output: {foldername}")
    print()

    records = []
    for r in range(R):
        print(f"Seed r={r}/{R - 1} ...")
        rec = solve_true_saa(full_data, m, N_true, k, r, r_start,
                             test_size, train_size)
        records.append(rec)
        pd.DataFrame([rec]).to_csv(
            foldername + f'true_saa_R{r}.csv', index=False)

    df_all = pd.DataFrame(records)
    df_all.to_csv(foldername + 'true_saa_all.csv', index=False)

    # Summary: mean and 25/75 quantiles across seeds -- used by the pct-diff plot.
    valid = df_all.dropna(subset=['insample_obj'])
    summary = {
        'dataset': dataset,
        'N_true': N_true,
        'k': k,
        'R': R,
        # in-sample objective
        'insample_obj_mean': valid['insample_obj'].mean(),
        'insample_obj_q25':  valid['insample_obj'].quantile(0.25),
        'insample_obj_q75':  valid['insample_obj'].quantile(0.75),
        # out-of-sample window j=0
        'outsample_eval0_mean': valid['outsample_eval0'].mean(),
        'outsample_eval0_q25':  valid['outsample_eval0'].quantile(0.25),
        'outsample_eval0_q75':  valid['outsample_eval0'].quantile(0.75),
        # out-of-sample window j=1 (matches eval1 / eval2 columns in experiment CSVs)
        'outsample_eval1_mean': valid['outsample_eval1'].mean(),
        'outsample_eval1_q25':  valid['outsample_eval1'].quantile(0.25),
        'outsample_eval1_q75':  valid['outsample_eval1'].quantile(0.75),
        # timing
        'solve_time_mean': df_all['solve_time'].mean(),
        'n_converged': int(valid.shape[0]),
    }
    pd.DataFrame([summary]).to_csv(
        foldername + 'true_saa_summary.csv', index=False)

    print()
    print("=== Summary ===")
    for key, val in summary.items():
        fmt = f'{val:.6f}' if isinstance(val, float) else str(val)
        print(f"  {key}: {fmt}")
    print(f"\nSaved to {foldername}")
