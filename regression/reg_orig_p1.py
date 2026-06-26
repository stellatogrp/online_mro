import argparse
import os
import sys

# Ensure local package imports work when run from SLURM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cvxpy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import time
import itertools
import copy

from utils_p1 import (
    createproblem_hingeMIO,
    create_scenario_hinge,
    fixed_cluster,
    generate_classification_data,
    get_n_processes,
    label_aware_kmeans,
    MOSEK_PARAMS,
    save_run_metadata,
    w2_dist,
    wasserstein,
    worst_case_hinge,
)
from utils_p1 import calc_cluster_val_hinge as calc_cluster_val
from utils_p1 import compute_cumulative_regret_online_hinge as compute_cumulative_regret
from utils_p1 import online_cluster_init_online as online_cluster_init
from utils_p1 import online_cluster_update_online as online_cluster_update


output_stream = sys.stdout


def reg_experiments(r_input, K, T, N_init, synthetic_data, power,p,r_start):
    """Online mean-robust DRO sparse-SVM vs. batch (kmeans) MRO-SVM (p = 1).

    p = 1 hinge-loss sibling of ``reg_orig.py``.  The p=2 perturbed-covariates
    DRO best-subset MI-SOCP is replaced by the p=1 DRO sparse-SVM mixed-integer
    LP (``createproblem_hingeMIO``, solved with MOSEK).  Data points are the
    joint (x, y) vectors of dimension ``m + 1`` (label y in {-1, +1}) clustered
    online / by kmeans; the worst-case expected hinge loss of a fixed beta has
    the p=1 closed form ``worst_case_hinge`` = empirical hinge + delta ||beta||_1.

    A third method, ``cluster_SAA``, reuses the batch-MRO kmeans clusters but
    solves the *non-robust* (delta=0) best subset on those weighted centroids,
    so it isolates the effect of clustering from the distributional robustness.
    """
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            synthetic_data[:, :m + 1], train_size=19000, test_size=1000, random_state=r_start + r)
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

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            # radius represents delta (order-1); RWPI: delta_n ~ 1/sqrt(n).
            radius = init_eps * (1 / (num_dat**(power)))
            running_samples = dat[init_ind:(init_ind + num_dat)]

            # ---- solve online MRO sparse-SVM best-subset problem ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= 8001 or (t in t_list):
                    cur_K = int(np.minimum(num_dat, K))
                    online_problem, online_x, online_z, data_train, eps_train, w_train = createproblem_hingeMIO(cur_K, m, k,p)
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
                if t <= 8001 or (t in t_list):
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
                    MRO_problem, MRO_x, MRO_z, MRO_data_train, MRO_eps_train, MRO_w_train = createproblem_hingeMIO(cur_K_mro, m, k,p)
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
                if t <= 8001 or (t in t_list):
                    wc_start = time.time()
                    new_worst = worst_case_hinge(running_samples, m, x_current, radius,p)
                    worst_time = time.time() - wc_start
                    history['worst_values'].append(new_worst)
                    history['worst_times'].append(worst_time)

                    if t <= 8001 or (t in t_list):
                        wc_start = time.time()
                        new_worst_MRO = worst_case_hinge(running_samples, m, MRO_x_current, radius,p)
                        MRO_worst_time = time.time() - wc_start

                        square_val, sig_val = calc_cluster_val(K, k_dict, num_dat, x_current, running_samples, m)[1:]

                        history['MRO_worst_values'].append(new_worst_MRO)
                        history['MRO_worst_times'].append(MRO_worst_time)

            # ---- worst-case hinge wrt previous-stage solutions (for regret) ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= 8001 or (t in t_list):
                    wc_start = time.time()
                    new_worst = worst_case_hinge(running_samples, m, x_prev, radius,p)
                    worst_time = time.time() - wc_start
                    history['worst_values_regret'].append(new_worst)
                    history['worst_times_regret'].append(worst_time)

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= 8001 or (t in t_list):
                    wc_start = time.time()
                    new_worst_MRO = worst_case_hinge(running_samples, m, MRO_x_prev, radius,p)
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
                if t <= 8001 or (t in t_list):
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

            # ---- write running CSV ----
            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= 8001 or (t in t_list):

                    MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s = compute_cumulative_regret(
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

        MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, CSA_e, CSA_s = compute_cumulative_regret(
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
        print(f"Exception in reg_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


if __name__ == '__main__':

    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--T', type=int, default=2001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=50)        # covariate dimension d
    parser.add_argument('--k', type=int, default=10)         # cardinality budget
    parser.add_argument('--k_true', type=int, default=10)    # true sparsity
    parser.add_argument('--Q', type=int, default=500)
    parser.add_argument('--fixed_time', type=int, default=1500)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--N_init', type=int, default=50)
    parser.add_argument('--rmse_mult', type=float, default=2)
    parser.add_argument('--cluster_interval', type=int, default=1,
                        help='Full re-cluster every this many steps for both the '
                             'online (micro->macro) and batch-MRO methods; between '
                             'full re-clusters a cheap nearest-center assignment is '
                             'used. Default 1 = re-cluster every step.')
    parser.add_argument('--r_start', type=int, default=0)
    parser.add_argument('--noise', type=float, default=3.0)
    parser.add_argument('--power', type=float, default=0.5)
    parser.add_argument('--p', type=int, default=1)


    arguments = parser.parse_args()
    foldername = arguments.foldername
    K = arguments.K
    R = arguments.R
    m = arguments.m
    k = arguments.k
    k_true = arguments.k_true
    Q = arguments.Q
    T = arguments.T
    p = arguments.p
    power = arguments.power
    noise_std = arguments.noise
    r_start = arguments.r_start
    fixed_time = arguments.fixed_time
    interval = arguments.interval
    N_init = arguments.N_init
    rmse_mult = arguments.rmse_mult
    cluster_interval = arguments.cluster_interval
    K_arr = [10,15,25]
    K = K_arr[idx]
    newfoldername = foldername + 'K' + str(K) + '_R' + str(R) + '_T' + str(T - 1) + '/'
    os.makedirs(newfoldername, exist_ok=True)
    print(newfoldername)

    # Synthetic classification data: one fixed dataset; each seed draws its own
    # train/test split inside reg_experiments.
    synthetic_data, beta_true = generate_classification_data(
        n_total=20000, m=m, k_true=k_true, noise_std=noise_std, seed=12345)

    init_ind = 0
    njobs = get_n_processes(100)
    # eps_init values are the delta prefactors; radius = delta = init_eps / sqrt(n).
    # Range brackets the out-of-sample optimum up through the validity threshold
    # (where the worst-case hinge objective becomes a valid upper bound on the
    # out-of-sample hinge loss); large init_eps over-shrinks beta (toward 0).
    # Empirically for m=20, k=5 (see visualize_radius_p1.py): the OOS-hinge
    # optimum sits at init_eps ~0.1-0.3 and the validity threshold runs from
    # ~0.2 (large n) up to ~1.5 (small n), so the sweep is centred there.
    eps_init = [0.7, 0.4, 0.3, 0.2,0.18,0.15,0.12,0.1,0.05]
    if T >= 5000:
        # long horizon reaches large n, where both the optimum and the validity
        # threshold are small; weight the sweep toward the low end.
        eps_init = [1.0, 0.5, 0.3, 0.1]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R), np.arange(M)))
    t_list = [4, 5, 9, 10, 14, 15, 19, 20, 29,30, 59,60, 1249, 1250, 1499, 1500, 1749, 1750, 1999, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    newdatname = foldername + 'T' + str(T - 1) + 'R' + str(R) + '/'

    save_run_metadata(
        {
            'filename': os.path.basename(__file__),
            'K': K, 'T': T, 'R': R, 'm': m, 'k': k, 'k_true': k_true,
            'noise_std': noise_std,
            'Q': Q,
            'fixed_time': fixed_time, 'interval': interval, 'N_init': N_init,
            'rmse_mult': rmse_mult, 'cluster_interval': cluster_interval,
            'r_start': r_start,
            'epsilon_values': [float(e) for e in eps_init],
            'num_epsilon_values': len(eps_init),
            'num_random_seeds': R,
            'total_test_combinations': len(eps_init) * R,
            'beta_true': [float(b) for b in beta_true],
            'power': power,
            'p':p
        },
        [newfoldername, (newdatname, f'metadata_K{K}.json')],
    )

    results = Parallel(n_jobs=njobs)(delayed(reg_experiments)(
        r_input, K, T, N_init, synthetic_data, power,p, r_start) for r_input in range(len(list_inds)))

    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r, epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))], ignore_index=True)
        findfs[r].to_csv(newfoldername + 'df_' + str(r + r_start) + '.csv')

    for r in range(R):
        findfs[r] = findfs[r].drop(columns=['weights', 'MRO_weights'])
        findfs[r].to_csv(newdatname + 'df_' + 'K' + str(K) + 'R' + str(r + r_start) + '.csv')

    print("DONE")
