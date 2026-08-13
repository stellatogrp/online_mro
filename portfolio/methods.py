"""Worker functions for the portfolio box-support W1-DRO experiments.

Provenance (bodies copied verbatim from the legacy drivers in ``port_box/``,
with module-level globals turned into explicit parameters and the mid-loop
checkpoint CSV rewrite gated behind ``checkpoint_every``):

  task_mro_subgrad  <- port_experiments in port_box.py
  task_mro_exact    <- port_experiments in port_box_orig.py (incl. cluster-SAA;
                       compute_saa_regret also from port_box_orig.py)
  task_dro_subgrad  <- port_experiments in port_box_DRO.py
  task_dro_exact    <- port_experiments in port_box_DRO_orig.py (incl. SAA for
                       epsnum == 0)
  run_true_saa      <- solve_true_saa in port_box_true_saa.py

Only intended deviations from the legacy bodies:
  * former globals are now function parameters (same names);
  * the mid-loop "recompute regret + rewrite whole checkpoint CSV" block only
    runs every ``checkpoint_every`` logged steps (the block has no side effects
    on loop state; the final CSV content is unchanged);
  * in task_dro_exact the calls to utils.compute_cumulative_regret use the
    alias ``compute_cumulative_regret_dro`` (the plain name is taken by the
    online variant used by the MRO paths, mirroring the per-file imports).
"""
import copy
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from .utils import (
    _expected_cost,
    box_dro_subgrad_step,
    compute_cumulative_regret_dro_only,
    create_scenario,
    create_scenario_cluster,
    createproblem_box_DRO,
    fixed_cluster,
    safe_solve,
    w2_dist,
    wasserstein,
    worst_case_value_box,
)
from .utils import calc_cluster_val_online as calc_cluster_val
from .utils import compute_cumulative_regret as compute_cumulative_regret_dro
from .utils import compute_cumulative_regret_online as compute_cumulative_regret
from .utils import online_cluster_init_online as online_cluster_init
from .utils import online_cluster_update_online as online_cluster_update


output_stream = sys.stdout


def task_mro_subgrad(r_input, K, T, N_init, synthetic_returns, lb, ub, eta_0, r_start,
                     newfoldername, power, list_inds, eps_init, m, train_size, test_size,
                     init_ind, Q, interval, t_list, fixed_time, rmse_mult, cluster_interval,
                     line_search, checkpoint_every, solve_interval=0, t_solve_list=()):
    """port_experiments from port_box.py (online/batch MRO, subgradient steps).

    ``solve_interval`` > 0 enables periodic exact re-anchor solves (mirrors the
    svm driver's ``_milp_warmstart`` gate and port_box_comb.py): whenever
    ``t % solve_interval == 0`` (t > 0) or ``t in t_solve_list``, the exact
    clustered box SOCP (``createproblem_box_DRO``, same builder/solver as
    ``task_mro_exact``) replaces the online and the reclustering (x, tau)
    iterates, and its wall time is charged into that step's recorded
    total_iteration time.  ``solve_interval`` 0/None keeps every code path
    identical to the legacy pure-subgradient behavior.
    """
    solve_interval = 0 if solve_interval is None else int(solve_interval)
    try:
        r,epsnum = list_inds[r_input]
        np.random.seed(r_start+r)
        dat, dateval = train_test_split(
             synthetic_returns[:, :m], train_size=train_size, test_size=test_size, random_state=r_start+r)
        # dat_indices = np.random.choice(48000,48000,replace=False)
        # dat = dat[dat_indices]
        init_eps = eps_init[epsnum]
        num_dat = N_init
        q_dict, k_dict,weight_update_time= online_cluster_init(K, Q, dat[init_ind:(init_ind+num_dat)], m)
        k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict = None
        init_samples = dat[init_ind:(init_ind+N_init)]


        # Initialize solutions
        MRO_x_prev = np.zeros(m)
        MRO_tau_prev = 0
        tau_prev = 0
        x_prev = np.zeros(m)
        init_radius_val = init_eps*(1/(num_dat**power))

        # Saddle-point scheme: maintain (x, tau) iterates updated by one
        # projected-subgradient step per interval using Danskin gradients from
        # the inner worst-case dual.  Step size from the classical bound
        # eta_t = (D_x / L_x) / sqrt(t+1).
        a_const = -5
        D_x = np.sqrt(2)
        R = np.linalg.norm(dateval, axis=1).mean()
        L_x = abs(a_const) * R
        # eta_0 = D_x / L_x

        x_current = np.ones(m) / m
        tau_current = 0.0
        MRO_x_current = np.ones(m) / m
        MRO_tau_current = 0.0

        # History for analysis
        history = {
            'x': [],
            'tau': [],
            'obj_values': [],
            'MRO_x': [],
            'MRO_tau': [],
            'MRO_obj_values': [],
            'worst_values': [],
            'worst_values_MRO':[],
            'epsilon': [],
            'weights': [],
            'weights_q': [],
            'online_computation_times': {
                'weight_update': [],
                'min_problem': [],
                'gradient_step': [],
                'total_iteration': []
            },
            'MRO_computation_times':{
            'clustering': [],
            'min_problem': [],
            'gradient_step': [],
            'total_iteration':[]
            },
            'distances':[],
            'mean_val':[],
            'square_val': [],
            'sig_val': [],
            'mean_val_MRO':[],
            'square_val_MRO': [],
            'sig_val_MRO': [],
            "satisfy":[],
            "MRO_satisfy":[],
            'worst_times':[],
            'MRO_worst_times':[],
            'MRO_worst_values':[],
            'worst_values':[],
            't':[],
            'MRO_weights':[],
            'MRO_worst_values_regret':[],
            'worst_values_regret':[],
            'MRO_worst_times_regret':[],
            'worst_times_regret':[],
            'regret_bound':[],
            'MRO_regret_bound':[],
            'regret_K': [],
            'MRO_regret_K':[]
        }

        ckpt_count = 0

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            radius = init_eps*(1/(num_dat**power))
            running_samples = dat[init_ind:(init_ind+num_dat)]

            # Periodic exact re-anchor gate (svm _milp_warmstart / port_box_comb
            # semantics: raw timestep t, plus the early t_solve_list steps).
            do_resolve = solve_interval > 0 and (
                (t > 0 and t % solve_interval == 0) or (t in t_solve_list))

            # solve online MRO problem
            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                online_dat = k_dict['d'][:num_dat]      # at most K micro-clusters
                online_w = k_dict['w'][:num_dat]

                grad_time = 0.0
                if t == 0:
                    # SAA warm-start on the clustered data to seed (x, tau).
                    seed_prob, seed_x, seed_tau = create_scenario_cluster(
                        online_dat, m, online_dat.shape[0], online_w, a=a_const)
                    _t0 = time.time()
                    if safe_solve(seed_prob, name='seed(online,t=0)', t=t,
                                  solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        x_current = seed_x.value
                        tau_current = float(seed_tau.value)
                    min_obj = worst_case_value_box(x_current, tau_current, online_dat,
                                                   online_w, radius, lb, ub, a=a_const)
                    min_time = time.time() - _t0
                else:
                    # Closed-form worst-case value + Danskin (sub)gradient, then
                    # one projected step (no inner CVXPY solve; strong duality).
                    eta = eta_0 / np.sqrt(t + 1)
                    grad_start = time.time()
                    x_current, tau_current, min_obj = box_dro_subgrad_step(
                        x_current, tau_current, online_dat, online_w, radius, lb, ub, eta,
                        a=a_const, line_search=line_search,
                    )
                    grad_time = time.time() - grad_start
                    min_time = grad_time

                # Periodic exact re-anchor: solve the clustered box SOCP on the
                # online cluster state (same builder/solver as task_mro_exact)
                # and replace the iterate; on solver failure keep the current
                # subgradient iterate.  Wall time (build + solve) is charged
                # into this step's total_iteration below, as the svm driver
                # charges milp_time.
                online_resolve_time = 0.0
                if do_resolve:
                    _t0 = time.time()
                    re_prob, re_x, re_tau, re_dat, re_eps, re_w = createproblem_box_DRO(
                        online_dat.shape[0], m, lb, ub, a=a_const)
                    re_dat.value = online_dat
                    re_eps.value = radius
                    re_w.value = online_w
                    if safe_solve(re_prob, name='online_reanchor', t=t,
                                  ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        x_current = re_x.value
                        tau_current = float(re_tau.value)
                    online_resolve_time = time.time() - _t0

                # Evaluate the online worst-case objective at the last iterate.
                min_obj = worst_case_value_box(x_current, tau_current, online_dat,
                                               online_w, radius, lb, ub, a=a_const)

                # Store timing information
                history['online_computation_times']['min_problem'].append(min_time)
                history['online_computation_times']['gradient_step'].append(grad_time)
                history['online_computation_times']['total_iteration'].append(min_time+weight_update_time+online_resolve_time)
                history['online_computation_times']['weight_update'].append(weight_update_time)
                history['t'].append(t)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                # solve MRO problem with new clusters
                if t <= fixed_time:
                    start_time = time.time()
                    cur_K = np.minimum(K,num_dat)
                    # Full KMeans every cluster_interval steps; otherwise reuse the
                    # previous full re-cluster's centers and just assign each
                    # datapoint to its nearest center (cluster_interval=1 => full
                    # KMeans every step, the original behavior).
                    do_full = (new_k_dict is None) or (t % cluster_interval == 0) \
                        or (new_k_dict['d'].shape[0] != cur_K)
                    if do_full:
                        if new_k_dict is not None and (num_dat > (interval+N_init)) and new_k_dict['d'].shape[0] == cur_K:
                            kmeans = KMeans(n_clusters=cur_K, init=new_k_dict['d'],n_init=1).fit(running_samples)
                        else:
                            kmeans = KMeans(n_clusters=cur_K,init="k-means++", n_init=1).fit(running_samples)
                        new_centers = kmeans.cluster_centers_
                        labels = kmeans.labels_
                    else:
                        new_centers = new_k_dict['d']
                        labels = np.argmin(cdist(running_samples, new_centers), axis=1)
                    wk = np.bincount(labels, minlength=new_centers.shape[0]) / num_dat
                    cluster_time = time.time()-start_time  if K < num_dat else 0.0
                    new_k_dict = {}
                    new_k_dict['K'] = cur_K
                    new_k_dict['data'] = {}
                    new_k_dict['a'] = new_centers
                    new_k_dict['d'] = new_centers
                    new_k_dict['w'] = wk
                    for k in range(new_centers.shape[0]):
                        new_k_dict['data'][k] = running_samples[labels==k]


                cur_K_mro = new_k_dict['d'].shape[0]

                MRO_grad_time = 0.0
                if t == 0:
                    # SAA warm-start on the clustered centroids to seed MRO (x, tau).
                    seed_prob, seed_x, seed_tau = create_scenario_cluster(
                        new_k_dict['d'], m, cur_K_mro, new_k_dict['w'], a=a_const)
                    _t0 = time.time()
                    if safe_solve(seed_prob, name='seed(MRO,t=0)', t=t,
                                  solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        MRO_x_current = seed_x.value
                        MRO_tau_current = float(seed_tau.value)
                    MRO_min_obj = worst_case_value_box(MRO_x_current, MRO_tau_current,
                                                       new_k_dict['d'], new_k_dict['w'], radius, lb, ub, a=a_const)
                    MRO_min_time = time.time() - _t0
                else:
                    # Closed-form worst-case value + Danskin (sub)gradient, then
                    # one projected step (no inner CVXPY solve; strong duality).
                    eta = eta_0 / np.sqrt(t + 1)
                    grad_start = time.time()
                    MRO_x_current, MRO_tau_current, MRO_min_obj = box_dro_subgrad_step(
                        MRO_x_current, MRO_tau_current, new_k_dict['d'], new_k_dict['w'],
                        radius, lb, ub, eta, a=a_const, line_search=line_search,
                    )
                    MRO_grad_time = time.time() - grad_start
                    MRO_min_time = MRO_grad_time

                # Periodic exact re-anchor for the reclustering iterate (same
                # semantics as the online branch above).
                MRO_resolve_time = 0.0
                if do_resolve:
                    _t0 = time.time()
                    re_prob, re_x, re_tau, re_dat, re_eps, re_w = createproblem_box_DRO(
                        cur_K_mro, m, lb, ub, a=a_const)
                    re_dat.value = new_k_dict['d']
                    re_eps.value = radius
                    re_w.value = new_k_dict['w']
                    if safe_solve(re_prob, name='MRO_reanchor', t=t,
                                  ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        MRO_x_current = re_x.value
                        MRO_tau_current = float(re_tau.value)
                    MRO_resolve_time = time.time() - _t0

                # Evaluate the MRO worst-case objective at the last iterate.
                MRO_min_obj = worst_case_value_box(MRO_x_current, MRO_tau_current,
                                                   new_k_dict['d'], new_k_dict['w'], radius, lb, ub, a=a_const)

                mean_val_mro, square_val_mro, sig_val_mro = calc_cluster_val(K, new_k_dict,num_dat,MRO_x_current,running_samples)

                history['MRO_computation_times']['min_problem'].append(MRO_min_time)
                history['MRO_computation_times']['gradient_step'].append(MRO_grad_time)
                history['MRO_computation_times']['total_iteration'].append(MRO_min_time+cluster_time+MRO_resolve_time)
                history['MRO_computation_times']['clustering'].append(cluster_time)
                history['MRO_weights'].append(new_k_dict['w'])


            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                # compute online MRO worst value (wrt non clustered data)
                uni_w = (1/num_dat)*np.ones(num_dat)
                t0 = time.time()
                new_worst = worst_case_value_box(x_current, tau_current, running_samples, uni_w, radius, lb, ub, a=a_const)
                worst_time = time.time() - t0

                history['worst_values'].append(new_worst)
                history['worst_times'].append(worst_time)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                t0 = time.time()
                new_worst_MRO = worst_case_value_box(MRO_x_current, MRO_tau_current, running_samples, uni_w, radius, lb, ub, a=a_const)
                MRO_worst_time = time.time() - t0

                mean_val, square_val, sig_val = calc_cluster_val(K, k_dict,num_dat,x_current,running_samples)

                history['MRO_worst_values'].append(new_worst_MRO)
                history['MRO_worst_times'].append(MRO_worst_time)


            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                # compute online worst value (wrt prev stage sols
                t0 = time.time()
                new_worst = worst_case_value_box(x_prev, tau_prev, running_samples, uni_w, radius, lb, ub, a=a_const)
                worst_time = time.time() - t0

                history['worst_values_regret'].append(new_worst)
                history['worst_times_regret'].append(worst_time)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                t0 = time.time()
                new_worst_MRO = worst_case_value_box(MRO_x_prev, MRO_tau_prev, running_samples, uni_w, radius, lb, ub, a=a_const)
                MRO_worst_time = time.time() - t0

                history['MRO_worst_values_regret'].append(new_worst_MRO)
                history['MRO_worst_times_regret'].append(MRO_worst_time)

                MRO_x_prev = MRO_x_current
                MRO_tau_prev = MRO_tau_current
                x_prev = x_current
                tau_prev = tau_current


            # New sample
            new_sample = dat[init_ind+num_dat]
            q_dict, k_dict, weight_update_time = online_cluster_update(K, new_sample, q_dict, k_dict, num_dat, t, fixed_time, m, Q, rmse_mult, cluster_interval=cluster_interval)
            if t >= fixed_time:
                new_k_dict, cluster_time = fixed_cluster(new_k_dict, new_sample, num_dat=num_dat, m=m)
            num_dat += 1
            # history['online_computation_times']['weight_update'].append(weight_update_time)
            # history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                N_dist_cur = wasserstein(init_samples,running_samples)

                history['regret_K'].append(w2_dist(k_dict, k_dict_prev, m)+ 2*radius )
                history['MRO_regret_K'].append(w2_dist(new_k_dict, new_k_dict_prev, m)+ 2*radius)
                regret_bound = (np.sum(history['regret_K']) + N_dist_cur+ radius + init_radius_val)/(t+1)
                MRO_regret_bound = (np.sum(history['MRO_regret_K']) + N_dist_cur+ radius + init_radius_val)/(t+1)
                history["regret_bound"].append(regret_bound)
                history["MRO_regret_bound"].append(MRO_regret_bound)
                k_dict_prev = copy.deepcopy(k_dict)
                new_k_dict_prev = copy.deepcopy(new_k_dict)

                history['mean_val'].append(mean_val)
                history['sig_val'].append(sig_val)
                history['square_val'].append(square_val)
                history['mean_val_MRO'].append(mean_val_mro)
                history['sig_val_MRO'].append(sig_val_mro)
                history['square_val_MRO'].append(square_val_mro)
                history['x'].append(x_current)
                history['tau'].append(tau_current)
                history['obj_values'].append(min_obj)
                history['MRO_x'].append(MRO_x_current)
                history['MRO_tau'].append(MRO_tau_current)
                history['MRO_obj_values'].append(MRO_min_obj)
                history['weights'].append(k_dict['w'].copy())
                history['weights_q'].append(q_dict['w'].copy())
                history['epsilon'].append(radius)


                # print(f"Current allocation: {x_current}")
                print(f"Current epsilon: {radius}")
                # print(f"Weight sum: {np.sum(k_dict['w'])}")

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= T or (t in t_list)):
                ckpt_count += 1
                if ckpt_count % checkpoint_every == 0:

                    MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = compute_cumulative_regret(
                    history, dateval, m)

                    df = pd.DataFrame({
                    'x': history['x'],
                    'tau': np.array(history['tau']),
                    'obj_values': np.array(history['obj_values']),
                    'MRO_x': history['MRO_x'],
                    'MRO_tau':np.array(history['MRO_tau']),
                    'MRO_obj_values': np.array(history['MRO_obj_values']),
                    'epsilon': np.array(history['epsilon']),
                    'weights':  history['weights'],
                    'MRO_weights': history['MRO_weights'],
                    'weights_q': history['weights_q'],
                    'online_time':  np.array(history['online_computation_times']['total_iteration']),
                    'MRO_time':  np.array(history['MRO_computation_times']['total_iteration']),
                    'online_gradient_time': np.array(history['online_computation_times']['gradient_step']),
                    'MRO_gradient_time': np.array(history['MRO_computation_times']['gradient_step']),
                    'MRO_mean_val': np.array(history['mean_val_MRO']),
                    'MRO_square_val': np.array(history['square_val_MRO']),
                    'MRO_sig_val': np.array(history['sig_val_MRO']),
                    'mean_val': np.array(history['mean_val']),
                    'square_val': np.array(history['square_val']),
                    'sig_val': np.array(history['sig_val']),
                    "worst_values":np.array(history['worst_values']),
                    "MRO_worst_values":np.array(history['MRO_worst_values']),
                    "worst_times":np.array(history['worst_times']),
                    "MRO_worst_times":np.array(history['MRO_worst_times']),
                    "worst_values_regret":np.array(history['worst_values_regret']),
                    "MRO_worst_values_regret":np.array(history['MRO_worst_values_regret']),
                    "worst_times_regret":np.array(history['worst_times_regret']),
                    "MRO_worst_times_regret":np.array(history['MRO_worst_times_regret']),
                    't': np.array(history['t']),
                    'regret_bound': history["regret_bound"],
                    'MRO_regret_bound': history["MRO_regret_bound"]
                    })
                    colnames = ['MRO_eval', "MRO_satisfy",'O_eval',"O_satisfy", "O_worst_satisfy", "MRO_worst_satisfy"]
                    colvals = [MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws]
                    for i in range(len(colnames)):
                        for j in range(2):
                            df[colnames[i]+str(j)] = np.array(colvals[i][j])
                    # print(f"Weights: {q_dict['w'], np.sum(q_dict['w']) }")
                    df.to_csv(newfoldername + 'df_' + str(r_start+r_input) +'.csv')

        MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = compute_cumulative_regret(
                history, dateval, m)

        df = pd.DataFrame({
        # 'x': history['x'],
        # 'tau': np.array(history['tau'][1:]),
        'obj_values': np.array(history['obj_values']),
        # 'MRO_x': history['MRO_x'],
        # 'MRO_tau':np.array(history['MRO_tau'][1:]),
        'MRO_obj_values': np.array(history['MRO_obj_values']),
        'epsilon': np.array(history['epsilon']),
        'weights':  history['weights'],
        'MRO_weights': history['MRO_weights'],
        # 'weights_q': history['weights_q'],
        'online_time':  np.array(history['online_computation_times']['total_iteration']),
        'MRO_time':  np.array(history['MRO_computation_times']['total_iteration']),
        'online_gradient_time': np.array(history['online_computation_times']['gradient_step']),
        'MRO_gradient_time': np.array(history['MRO_computation_times']['gradient_step']),
        'MRO_mean_val': np.array(history['mean_val_MRO']),
        'MRO_square_val': np.array(history['square_val_MRO']),
        'MRO_sig_val': np.array(history['sig_val_MRO']),
        'mean_val': np.array(history['mean_val']),
        'square_val': np.array(history['square_val']),
        'sig_val': np.array(history['sig_val']),
        "worst_values":np.array(history['worst_values']),
        "MRO_worst_values":np.array(history['MRO_worst_values']),
        "worst_times":np.array(history['worst_times']),
        "MRO_worst_times":np.array(history['MRO_worst_times']),
        "worst_values_regret":np.array(history['worst_values_regret']),
        "MRO_worst_values_regret":np.array(history['MRO_worst_values_regret']),
        "worst_times_regret":np.array(history['worst_times_regret']),
        "MRO_worst_times_regret":np.array(history['MRO_worst_times_regret']),
        't': np.array(history['t']),
        'regret_bound': history["regret_bound"],
                'MRO_regret_bound': history["MRO_regret_bound"]
        })
        colnames = ['MRO_eval', "MRO_satisfy",'O_eval',"O_satisfy", "O_worst_satisfy", "MRO_worst_satisfy"]
        colvals = [MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws]
        for i in range(len(colnames)):
            for j in range(2):
                df[colnames[i]+str(j)] = np.array(colvals[i][j])
            # df.to_csv('df.csv')

        return df
    except Exception as e:
        import traceback
        print(f"Exception in port_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


def compute_saa_regret(history, dateval, m):
    """cluster_SAA analogue of ``compute_cumulative_regret_online``.

    Evaluates the SAA-on-clustered-data solution (``SAA_x``/``SAA_tau``) on the
    held-out samples and reports the (in-sample objective >= out-of-sample
    cost) satisfaction indicators, matching the online/MRO bookkeeping.
    """
    def evaluate_expected_cost(d_eval, x, tau):
        return np.mean(np.maximum(-5*d_eval@x - 4*tau, tau))

    SAA_e = []
    SAA_s = []
    SAA_ws = []
    T = len(history['t'])
    for j in range(2):
        eval_values = np.zeros(T)
        eval_samples = dateval[(j*200):(j+1)*200, :m]
        for t in range(T):
            eval_values[t] = evaluate_expected_cost(
                eval_samples, history['SAA_x'][t], history['SAA_tau'][t])
        satisfy = np.array(history['SAA_obj_values'] >= eval_values).astype(float)
        worst_satisfy = np.array(
            np.array(history['SAA_obj_values']) + 5*np.array(history['sig_val_SAA']) >= eval_values).astype(float)
        SAA_e.append(eval_values)
        SAA_s.append(satisfy)
        SAA_ws.append(worst_satisfy)
    return SAA_e, SAA_s, SAA_ws


def task_mro_exact(r_input, K, T, N_init, synthetic_returns, lb, ub, r_start, power,
                   list_inds, eps_init, m, train_size, test_size, init_ind, Q, interval,
                   t_list, fixed_time, rmse_mult, cluster_interval, foldername,
                   checkpoint_every):
    """port_experiments from port_box_orig.py (exact SOCP solves + cluster-SAA)."""
    try:
        r,epsnum = list_inds[r_input]
        np.random.seed(r_start+r)
        dat, dateval = train_test_split(
             synthetic_returns[:, :m], train_size=train_size, test_size=test_size, random_state=r_start+r)
        # dat_indices = np.random.choice(48000,48000,replace=False)
        # dat = dat[dat_indices]
        init_eps = eps_init[epsnum]
        num_dat = N_init
        a_const = -5
        q_dict, k_dict,weight_update_time= online_cluster_init(K, Q, dat[init_ind:(init_ind+num_dat)], m)
        k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict_prev = copy.deepcopy(k_dict)
        new_k_dict = None
        init_samples = dat[init_ind:(init_ind+N_init)]


        # Initialize solutions
        MRO_x_prev = np.zeros(m)
        MRO_tau_prev = 0
        tau_prev = 0
        x_prev = np.zeros(m)
        # cluster_SAA: sample-average approximation on the *same* clustered data
        # as MRO but with radius 0 (no Wasserstein ball).
        SAA_x_prev = np.zeros(m)
        SAA_tau_prev = 0
        # Pre-seed the iterate so a solver failure at the first interval still
        # leaves x_current / tau_current well defined for the worst-case eval.
        x_current = np.ones(m) / m
        tau_current = 0.0
        MRO_x_current = np.ones(m) / m
        MRO_tau_current = 0.0
        SAA_x_current = np.ones(m) / m
        SAA_tau_current = 0.0
        init_radius_val = init_eps*(1/(num_dat**power))
        # Keep the online and MRO solves on *separate* problem instances so that
        # Clarabel's in-place update path only ever sees one parameter trajectory
        # per solver. Sharing one Problem between the two branches triggers
        # "Data formatting error" inside Clarabel's update() when the centers
        # switch between the online (k_dict) and MRO (new_k_dict) data sources.
        _cur0 = int(np.minimum(num_dat, K))
        online_problem, online_x, online_tau, data_train, eps_train, w_train = createproblem_box_DRO(_cur0, m, lb, ub, a=a_const)
        MRO_problem, MRO_x, MRO_tau, MRO_data_train, MRO_eps_train, MRO_w_train = createproblem_box_DRO(_cur0, m, lb, ub, a=a_const)

        # History for analysis
        history = {
            'x': [],
            'tau': [],
            'obj_values': [],
            'MRO_x': [],
            'MRO_tau': [],
            'MRO_obj_values': [],
            'worst_values': [],
            'worst_values_MRO':[],
            'epsilon': [],
            'weights': [],
            'weights_q': [],
            'online_computation_times': {
                'weight_update': [],
                'min_problem': [],
                'total_iteration': []
            },
            'MRO_computation_times':{
            'clustering': [],
            'min_problem': [],
            'total_iteration':[]
            },
            'SAA_x': [],
            'SAA_tau': [],
            'SAA_obj_values': [],
            'SAA_worst_values': [],
            'SAA_worst_times': [],
            'SAA_worst_values_regret': [],
            'SAA_worst_times_regret': [],
            'SAA_weights': [],
            'mean_val_SAA': [],
            'square_val_SAA': [],
            'sig_val_SAA': [],
            'SAA_computation_times':{
            'clustering': [],
            'min_problem': [],
            'total_iteration':[]
            },
            'distances':[],
            'mean_val':[],
            'square_val': [],
            'sig_val': [],
            'mean_val_MRO':[],
            'square_val_MRO': [],
            'sig_val_MRO': [],
            "satisfy":[],
            "MRO_satisfy":[],
            'worst_times':[],
            'MRO_worst_times':[],
            'MRO_worst_values':[],
            'worst_values':[],
            't':[],
            'MRO_weights':[],
            'MRO_worst_values_regret':[],
            'worst_values_regret':[],
            'MRO_worst_times_regret':[],
            'worst_times_regret':[],
            'regret_bound':[],
            'MRO_regret_bound':[],
            'regret_K': [],
            'MRO_regret_K':[]
        }

        ckpt_count = 0

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            radius = init_eps*(1/(num_dat**power))
            running_samples = dat[init_ind:(init_ind+num_dat)]

            # solve online MRO problem
            if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    # Rebuild the problem on every solve.  Reusing a cached
                    # CVXPY problem across many parameter updates can leave
                    # Clarabel's in-place `_solver.update(P, q, A, b)` path in a
                    # stale numerical state (first showing up as "Solution may
                    # be inaccurate" warnings, then hard-failing with
                    # `Exception: Data formatting error`).  Rebuilding each
                    # time costs one extra sub-millisecond CVXPY compile at
                    # K<=25 and gives Clarabel a fresh solver every iteration.
                    cur_K = np.minimum(num_dat,K)
                    online_problem, online_x, online_tau, data_train, eps_train, w_train = createproblem_box_DRO(cur_K, m, lb, ub, a=a_const)
                    data_train.value = k_dict['d'][:num_dat]
                    eps_train.value = radius
                    w_train.value = k_dict['w'][:num_dat]

                    if safe_solve(online_problem, name='online_problem', t=t,
                                  ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        x_current = online_x.value
                        tau_current = online_tau.value
                        min_obj = online_problem.objective.value
                        min_time = online_problem.solver_stats.solve_time
                    else:
                        # Keep prior iterate; record NaN sentinels.
                        min_obj = np.nan
                        min_time = np.nan


                    # Store timing information
                    history['online_computation_times']['min_problem'].append(min_time)
                    history['online_computation_times']['total_iteration'].append(min_time+weight_update_time)
                    history['online_computation_times']['weight_update'].append(weight_update_time)
                    history['t'].append(t)

            if t % interval == 0 or ((t-1) % interval == 0)  or (t in t_list) :
                if t <= T or (t in t_list):
                    # solve MRO problem with new clusters
                    if t <= fixed_time:
                        start_time = time.time()
                        cur_K = np.minimum(K,num_dat)
                        do_full = (new_k_dict is None) or (t % cluster_interval == 0) \
                            or (new_k_dict['d'].shape[0] != cur_K)
                        if do_full:
                            if new_k_dict is not None and (num_dat > (interval+N_init)) and new_k_dict['d'].shape[0] == cur_K:
                                kmeans = KMeans(n_clusters=cur_K, init=new_k_dict['d'],n_init=1).fit(running_samples)
                            else:
                                print("restart kmeans", cur_K, num_dat)
                                kmeans = KMeans(n_clusters=cur_K,init="k-means++", n_init=1).fit(running_samples)
                            new_centers = kmeans.cluster_centers_
                            labels = kmeans.labels_
                        else:
                            new_centers = new_k_dict['d']
                            labels = np.argmin(cdist(running_samples, new_centers), axis=1)
                        wk = np.bincount(labels, minlength=new_centers.shape[0]) / num_dat
                        # Only count clustering time when K < num_dat -- otherwise
                        # cur_K = num_dat and the kmeans call is a trivial pass-through.
                        cluster_time = (time.time() - start_time) if K < num_dat else 0.0
                        new_k_dict = {}
                        new_k_dict['K'] = cur_K
                        new_k_dict['data'] = {}
                        new_k_dict['a'] = new_centers
                        new_k_dict['d'] = new_centers
                        new_k_dict['w'] = wk
                        for k in range(new_centers.shape[0]):
                            new_k_dict['data'][k] = running_samples[labels==k]
                        new_k_dict['d'] = new_centers


                    # Rebuild the MRO problem on every solve -- same reason as
                    # the online branch above (avoid Clarabel's stale cached
                    # solver across many parameter updates).
                    cur_K_mro = new_k_dict['d'].shape[0]
                    MRO_problem, MRO_x, MRO_tau, MRO_data_train, MRO_eps_train, MRO_w_train = createproblem_box_DRO(cur_K_mro, m, lb, ub, a=a_const)
                    MRO_data_train.value = new_k_dict['d']
                    MRO_w_train.value = new_k_dict['w']
                    MRO_eps_train.value = radius
                    if safe_solve(MRO_problem, name='MRO_problem', t=t,
                                  ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        MRO_x_current = MRO_x.value
                        MRO_tau_current = MRO_tau.value
                        MRO_min_obj = MRO_problem.objective.value
                        MRO_min_time = MRO_problem.solver_stats.solve_time
                    else:
                        MRO_min_obj = np.nan
                        MRO_min_time = np.nan
                    mean_val_mro, square_val_mro, sig_val_mro = calc_cluster_val(K, new_k_dict,num_dat,MRO_x_current,running_samples)

                    history['MRO_computation_times']['min_problem'].append(MRO_min_time)
                    history['MRO_computation_times']['total_iteration'].append(MRO_min_time+cluster_time)
                    history['MRO_computation_times']['clustering'].append(cluster_time)
                    history['MRO_weights'].append(new_k_dict['w'])

                    # cluster_SAA: SAA on the clustered data (new_k_dict) for
                    # the given K -- the scenario/CVaR problem (same code as the
                    # plain SAA, create_scenario_*) over the K cluster centroids
                    # weighted by the cluster masses.
                    SAA_problem, SAA_x, SAA_tau = create_scenario_cluster(new_k_dict['d'], m, cur_K_mro, new_k_dict['w'])
                    if safe_solve(SAA_problem, name='SAA_problem', t=t,
                                  ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        SAA_x_current = SAA_x.value
                        SAA_tau_current = SAA_tau.value
                        SAA_min_obj = SAA_problem.objective.value
                        SAA_min_time = SAA_problem.solver_stats.solve_time
                    else:
                        SAA_min_obj = np.nan
                        SAA_min_time = np.nan
                    mean_val_saa, square_val_saa, sig_val_saa = calc_cluster_val(K, new_k_dict,num_dat,SAA_x_current,running_samples)

                    history['SAA_computation_times']['min_problem'].append(SAA_min_time)
                    history['SAA_computation_times']['total_iteration'].append(SAA_min_time+cluster_time)
                    history['SAA_computation_times']['clustering'].append(cluster_time)
                    history['SAA_weights'].append(new_k_dict['w'])


            if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list) :
                if t <= T or (t in t_list):
                    # compute online MRO worst value (wrt non clustered data)

                    uni_w = (1/num_dat)*np.ones(num_dat)
                    _t0 = time.time()
                    new_worst = worst_case_value_box(x_current, tau_current, running_samples, uni_w, radius, lb, ub, a=a_const)
                    worst_time = time.time() - _t0

                    history['worst_values'].append(new_worst)
                    history['worst_times'].append(worst_time)

                if t % interval == 0 or ((t-1) % interval == 0)  or (t in t_list) :
                    if t <= T or (t in t_list):
                        _t0 = time.time()
                        new_worst_MRO = worst_case_value_box(MRO_x_current, MRO_tau_current, running_samples, uni_w, radius, lb, ub, a=a_const)
                        MRO_worst_time = time.time() - _t0

                        mean_val, square_val, sig_val = calc_cluster_val(K, k_dict,num_dat,x_current,running_samples)
                        # q_lens = [len(q_dict['data'][i]) for i in range(q_dict['cur_Q'])]
                        # k_lens = [len(k_dict['data'][i]) for i in range(k_dict['K'])]
                        # print("Q nums", q_lens, np.sum(q_lens), num_dat)
                        # print("K nums", k_lens, np.sum(k_lens), num_dat)

                        history['MRO_worst_values'].append(new_worst_MRO)
                        history['MRO_worst_times'].append(MRO_worst_time)

                        # cluster_SAA worst value (wrt non clustered data)
                        _t0 = time.time()
                        new_worst_SAA = worst_case_value_box(SAA_x_current, SAA_tau_current, running_samples, uni_w, radius, lb, ub, a=a_const)
                        SAA_worst_time = time.time() - _t0

                        history['SAA_worst_values'].append(new_worst_SAA)
                        history['SAA_worst_times'].append(SAA_worst_time)


            if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list) :
                if t <= T or (t in t_list):
                    # compute online worst value (wrt prev stage sols
                    _t0 = time.time()
                    new_worst = worst_case_value_box(x_prev, tau_prev, running_samples, uni_w, radius, lb, ub, a=a_const)
                    worst_time = time.time() - _t0

                    history['worst_values_regret'].append(new_worst)
                    history['worst_times_regret'].append(worst_time)

            if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)  :
                if t <= T or (t in t_list):
                    _t0 = time.time()
                    new_worst_MRO = worst_case_value_box(MRO_x_prev, MRO_tau_prev, running_samples, uni_w, radius, lb, ub, a=a_const)
                    MRO_worst_time = time.time() - _t0

                    # q_lens = [len(q_dict['data'][i]) for i in range(q_dict['cur_Q'])]
                    # k_lens = [len(k_dict['data'][i]) for i in range(k_dict['K'])]
                    # print("Q nums", q_lens, np.sum(q_lens), num_dat)
                    # print("K nums", k_lens, np.sum(k_lens), num_dat)

                    history['MRO_worst_values_regret'].append(new_worst_MRO)
                    history['MRO_worst_times_regret'].append(MRO_worst_time)

                    # cluster_SAA worst value (wrt prev stage sols)
                    _t0 = time.time()
                    new_worst_SAA = worst_case_value_box(SAA_x_prev, SAA_tau_prev, running_samples, uni_w, radius, lb, ub, a=a_const)
                    SAA_worst_time = time.time() - _t0

                    history['SAA_worst_values_regret'].append(new_worst_SAA)
                    history['SAA_worst_times_regret'].append(SAA_worst_time)

                    MRO_x_prev = MRO_x_current
                    MRO_tau_prev = MRO_tau_current
                    x_prev = x_current
                    tau_prev = tau_current
                    SAA_x_prev = SAA_x_current
                    SAA_tau_prev = SAA_tau_current


            # New sample
            new_sample = dat[init_ind+num_dat]
            q_dict, k_dict, weight_update_time = online_cluster_update(K, new_sample, q_dict, k_dict, num_dat, t, fixed_time, m, Q, rmse_mult, cluster_interval=cluster_interval)
            if t >= fixed_time:
                new_k_dict, cluster_time = fixed_cluster(new_k_dict, new_sample, num_dat=num_dat, m=m)
            num_dat += 1
            # history['online_computation_times']['weight_update'].append(weight_update_time)
            # history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)

            if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list) :
                if t <= T or (t in t_list):
                    N_dist_cur = wasserstein(init_samples,running_samples)

                    history['regret_K'].append(w2_dist(k_dict, k_dict_prev, m) )
                    history['MRO_regret_K'].append(w2_dist(new_k_dict, new_k_dict_prev, m))
                    regret_bound = (np.sum(history['regret_K']) + N_dist_cur+ init_radius_val - radius)/(t+1)
                    MRO_regret_bound = (np.sum(history['MRO_regret_K']) + N_dist_cur+  init_radius_val - radius)/(t+1)
                    history["regret_bound"].append(regret_bound)
                    history["MRO_regret_bound"].append(MRO_regret_bound)
                    k_dict_prev = copy.deepcopy(k_dict)
                    new_k_dict_prev = copy.deepcopy(new_k_dict)

                    history['mean_val'].append(mean_val)
                    history['sig_val'].append(sig_val)
                    history['square_val'].append(square_val)
                    history['mean_val_MRO'].append(mean_val_mro)
                    history['sig_val_MRO'].append(sig_val_mro)
                    history['square_val_MRO'].append(square_val_mro)
                    history['mean_val_SAA'].append(mean_val_saa)
                    history['sig_val_SAA'].append(sig_val_saa)
                    history['square_val_SAA'].append(square_val_saa)
                    history['SAA_x'].append(SAA_x_current)
                    history['SAA_tau'].append(SAA_tau_current)
                    history['SAA_obj_values'].append(SAA_min_obj)
                    history['x'].append(x_current)
                    history['tau'].append(tau_current)
                    history['obj_values'].append(min_obj)
                    history['MRO_x'].append(MRO_x_current)
                    history['MRO_tau'].append(MRO_tau_current)
                    history['MRO_obj_values'].append(MRO_min_obj)
                    history['weights'].append(k_dict['w'].copy())
                    history['weights_q'].append(q_dict['w'].copy())
                    history['epsilon'].append(radius)


                    # print(f"Current allocation: {x_current}")
                    print(f"Current epsilon: {radius}")
                    # print(f"Weight sum: {np.sum(k_dict['w'])}")

            if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)  :
                if t <= T or (t in t_list):
                    ckpt_count += 1
                    if ckpt_count % checkpoint_every == 0:

                        MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = compute_cumulative_regret(
                        history, dateval, m)
                        SAA_e, SAA_s, SAA_ws = compute_saa_regret(history, dateval, m)

                        df = pd.DataFrame({
                        # 'x': history['x'],
                        'tau': np.array(history['tau']),
                        'obj_values': np.array(history['obj_values']),
                        # 'MRO_x': history['MRO_x'],
                        'MRO_tau':np.array(history['MRO_tau']),
                        'MRO_obj_values': np.array(history['MRO_obj_values']),
                        'SAA_tau': np.array(history['SAA_tau']),
                        'SAA_obj_values': np.array(history['SAA_obj_values']),
                        'epsilon': np.array(history['epsilon']),
                        'weights':  history['weights'],
                        'MRO_weights': history['MRO_weights'],
                        'SAA_weights': history['SAA_weights'],
                        # 'weights_q': history['weights_q'],
                        'online_time':  np.array(history['online_computation_times']['total_iteration']),
                        'MRO_time':  np.array(history['MRO_computation_times']['total_iteration']),
                        'SAA_time':  np.array(history['SAA_computation_times']['total_iteration']),
                        'MRO_mean_val': np.array(history['mean_val_MRO']),
                        'MRO_square_val': np.array(history['square_val_MRO']),
                        'MRO_sig_val': np.array(history['sig_val_MRO']),
                        'SAA_mean_val': np.array(history['mean_val_SAA']),
                        'SAA_square_val': np.array(history['square_val_SAA']),
                        'SAA_sig_val': np.array(history['sig_val_SAA']),
                        'mean_val': np.array(history['mean_val']),
                        'square_val': np.array(history['square_val']),
                        'sig_val': np.array(history['sig_val']),
                        "worst_values":np.array(history['worst_values']),
                        "MRO_worst_values":np.array(history['MRO_worst_values']),
                        "SAA_worst_values":np.array(history['SAA_worst_values']),
                        "worst_times":np.array(history['worst_times']),
                        "MRO_worst_times":np.array(history['MRO_worst_times']),
                        "SAA_worst_times":np.array(history['SAA_worst_times']),
                        "worst_values_regret":np.array(history['worst_values_regret']),
                        "MRO_worst_values_regret":np.array(history['MRO_worst_values_regret']),
                        "SAA_worst_values_regret":np.array(history['SAA_worst_values_regret']),
                        "worst_times_regret":np.array(history['worst_times_regret']),
                        "MRO_worst_times_regret":np.array(history['MRO_worst_times_regret']),
                        "SAA_worst_times_regret":np.array(history['SAA_worst_times_regret']),
                        't': np.array(history['t']),
                        'regret_bound': history["regret_bound"],
                        'MRO_regret_bound': history["MRO_regret_bound"]
                        })
                        colnames = ['MRO_eval', "MRO_satisfy",'O_eval',"O_satisfy", "O_worst_satisfy", "MRO_worst_satisfy", "SAA_eval", "SAA_satisfy", "SAA_worst_satisfy"]
                        colvals = [MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, SAA_e, SAA_s, SAA_ws]
                        for i in range(len(colnames)):
                            for j in range(2):
                                df[colnames[i]+str(j)] = np.array(colvals[i][j])
                        df.to_csv(foldername+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
                        # print(f"Weights: {q_dict['w'], np.sum(q_dict['w']) }")

        MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = compute_cumulative_regret(
                history, dateval, m)
        SAA_e, SAA_s, SAA_ws = compute_saa_regret(history, dateval, m)

        df = pd.DataFrame({
        # 'x': history['x'],
        # 'tau': np.array(history['tau'][1:]),
        'obj_values': np.array(history['obj_values']),
        # 'MRO_x': history['MRO_x'],
        # 'MRO_tau':np.array(history['MRO_tau'][1:]),
        'MRO_obj_values': np.array(history['MRO_obj_values']),
        'SAA_obj_values': np.array(history['SAA_obj_values']),
        'epsilon': np.array(history['epsilon']),
        'weights':  history['weights'],
        'MRO_weights': history['MRO_weights'],
        'SAA_weights': history['SAA_weights'],
        # 'weights_q': history['weights_q'],
        'online_time':  np.array(history['online_computation_times']['total_iteration']),
        'MRO_time':  np.array(history['MRO_computation_times']['total_iteration']),
        'SAA_time':  np.array(history['SAA_computation_times']['total_iteration']),
        'MRO_mean_val': np.array(history['mean_val_MRO']),
        'MRO_square_val': np.array(history['square_val_MRO']),
        'MRO_sig_val': np.array(history['sig_val_MRO']),
        'SAA_mean_val': np.array(history['mean_val_SAA']),
        'SAA_square_val': np.array(history['square_val_SAA']),
        'SAA_sig_val': np.array(history['sig_val_SAA']),
        'mean_val': np.array(history['mean_val']),
        'square_val': np.array(history['square_val']),
        'sig_val': np.array(history['sig_val']),
        "worst_values":np.array(history['worst_values']),
        "MRO_worst_values":np.array(history['MRO_worst_values']),
        "SAA_worst_values":np.array(history['SAA_worst_values']),
        "worst_times":np.array(history['worst_times']),
        "MRO_worst_times":np.array(history['MRO_worst_times']),
        "SAA_worst_times":np.array(history['SAA_worst_times']),
        "worst_values_regret":np.array(history['worst_values_regret']),
        "MRO_worst_values_regret":np.array(history['MRO_worst_values_regret']),
        "SAA_worst_values_regret":np.array(history['SAA_worst_values_regret']),
        "worst_times_regret":np.array(history['worst_times_regret']),
        "MRO_worst_times_regret":np.array(history['MRO_worst_times_regret']),
        "SAA_worst_times_regret":np.array(history['SAA_worst_times_regret']),
        't': np.array(history['t']),
        'regret_bound': history["regret_bound"],
                'MRO_regret_bound': history["MRO_regret_bound"]
        })
        colnames = ['MRO_eval', "MRO_satisfy",'O_eval',"O_satisfy", "O_worst_satisfy", "MRO_worst_satisfy", "SAA_eval", "SAA_satisfy", "SAA_worst_satisfy"]
        colvals = [MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws, SAA_e, SAA_s, SAA_ws]
        for i in range(len(colnames)):
            for j in range(2):
                df[colnames[i]+str(j)] = np.array(colvals[i][j])
        df.to_csv(foldername+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
        # df.to_csv('df.csv')

        return df
    except Exception as e:
        import traceback
        print(f"Exception in port_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


def task_dro_subgrad(r_input, T, N_init, data, lb, ub, eta_0, r_start, power,
                     list_inds, eps_init, alpha, m, train_size, test_size, init_ind,
                     interval, t_list, line_search, foldername, checkpoint_every,
                     solve_interval=0, t_solve_list=()):
    """port_experiments from port_box_DRO.py (full-data DRO, subgradient steps).

    ``solve_interval`` > 0 enables periodic exact re-anchor solves (same gate
    as the svm driver's ``_milp_warmstart``): whenever
    ``t % solve_interval == 0`` (t > 0) or ``t in t_solve_list``, the exact
    full-sample box SOCP replaces the DRO (x, tau) iterate -- solved with the
    direct Clarabel assembly (``BoxDROSocp``, same PORT_DRO_DIRECT gate and
    CVXPY fallback as ``task_dro_exact``) -- and its wall time is charged into
    that step's recorded total_iteration time.  On solver failure the current
    subgradient iterate is kept.  ``solve_interval`` 0/None keeps every code
    path identical to the legacy pure-subgradient behavior.
    """
    import os

    solve_interval = 0 if solve_interval is None else int(solve_interval)
    use_direct = False
    if solve_interval > 0:
        use_direct = os.environ.get('PORT_DRO_DIRECT', '1') != '0'
        if use_direct:
            try:
                from .direct_socp import BoxDROSocp
            except Exception as e:
                print(f"[task_dro_subgrad] direct SOCP unavailable ({e}); "
                      "falling back to CVXPY", flush=True)
                use_direct = False
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            data, train_size=train_size, test_size=test_size, random_state=r_start + r)

        init_eps = eps_init[epsnum]
        num_dat = N_init
        a_const = -1.0 / (1.0 - alpha)          # CVaR coefficient (a = -5 at alpha=0.8)
        # Direct solver for the periodic re-anchors, built once per task.
        direct_socp = BoxDROSocp(m, lb, ub, a=a_const) if use_direct else None

        # Online iterate; warm-starts across the increasing sample size.  All
        # evaluations below use this last iterate directly.
        DRO_x_current = np.ones(m) / m
        DRO_tau_current = 0.0

        history = {
            'DRO_x': [], 'DRO_tau': [], 'DRO_obj_values': [],
            'epsilon': [], 'DRO_computation_times': {'total_iteration': []},
            't': [],
        }

        ckpt_count = 0

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")
            radius = init_eps * (1.0 / (num_dat ** power))
            running_samples = dat[init_ind:(init_ind + num_dat)]
            w_unif = (1.0 / num_dat) * np.ones(num_dat)

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    # One Section-9.10 projected subgradient step (box inner solve
                    # of Section 9.5); the iterate persists across timesteps.
                    eta = eta_0 / np.sqrt(t + 1)
                    _t0 = time.perf_counter()
                    DRO_x_current, DRO_tau_current, DRO_min_obj = box_dro_subgrad_step(
                        DRO_x_current, DRO_tau_current, running_samples, w_unif,
                        radius, lb, ub, eta, a=a_const, line_search=line_search)
                    DRO_min_time = time.perf_counter() - _t0

                    # Periodic exact re-anchor: full-sample box SOCP replaces
                    # the iterate; on failure keep the subgradient iterate.
                    # Wall time is charged into this step's total_iteration.
                    resolve_time = 0.0
                    if solve_interval > 0 and ((t > 0 and t % solve_interval == 0)
                                               or (t in t_solve_list)):
                        _t0 = time.perf_counter()
                        if use_direct:
                            d_obj, d_x, d_tau, d_time, d_status = direct_socp.solve(
                                running_samples, w_unif, radius)
                            if d_status in ('optimal', 'optimal_inaccurate'):
                                DRO_x_current = d_x
                                DRO_tau_current = float(d_tau)
                            else:
                                print(f"[direct_socp] DRO_reanchor t={t} "
                                      f"status={d_status}; keeping iterate", flush=True)
                        else:
                            re_prob, re_x, re_tau, re_dat, re_eps, re_w = createproblem_box_DRO(
                                num_dat, m, lb, ub, a=a_const)
                            re_dat.value = running_samples
                            re_w.value = w_unif
                            re_eps.value = radius
                            if safe_solve(re_prob, name='DRO_reanchor', t=t,
                                          ignore_dpp=True, solver=cp.CLARABEL, verbose=False):
                                DRO_x_current = re_x.value
                                DRO_tau_current = float(re_tau.value)
                        resolve_time = time.perf_counter() - _t0

                    # Evaluate the DRO objective at the last iterate.
                    DRO_min_obj = worst_case_value_box(
                        DRO_x_current, DRO_tau_current, running_samples, w_unif,
                        radius, lb, ub, a=a_const)

                    history['DRO_computation_times']['total_iteration'].append(DRO_min_time + resolve_time)
                    history['DRO_x'].append(DRO_x_current)
                    history['DRO_tau'].append(DRO_tau_current)
                    history['DRO_obj_values'].append(DRO_min_obj)
                    history['epsilon'].append(radius)
                    history['t'].append(t)

            num_dat += 1

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    ckpt_count += 1
                    if ckpt_count % checkpoint_every == 0:
                        DRO_eval, DRO_satisfy = compute_cumulative_regret_dro_only(
                            history, dateval, m, a=a_const)
                        df = pd.DataFrame({
                            'DRO_tau': np.array(history['DRO_tau']),
                            'DRO_obj_values': np.array(history['DRO_obj_values']),
                            'epsilon': np.array(history['epsilon']),
                            'DRO_time': np.array(history['DRO_computation_times']['total_iteration']),
                            'DRO_eval1': DRO_eval[0], 'DRO_eval2': DRO_eval[1],
                            'DRO_satisfy1': DRO_satisfy[0], 'DRO_satisfy2': DRO_satisfy[1],
                            't': np.array(history['t']),
                        })
                        df.to_csv(foldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        DRO_eval, DRO_satisfy = compute_cumulative_regret_dro_only(history, dateval, m, a=a_const)
        df = pd.DataFrame({
            'DRO_tau': np.array(history['DRO_tau']),
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time': np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval1': DRO_eval[0], 'DRO_eval2': DRO_eval[1],
            'DRO_satisfy1': DRO_satisfy[0], 'DRO_satisfy2': DRO_satisfy[1],
            't': np.array(history['t']),
        })
        df.to_csv(foldername + 'DRO_new' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')
        return df
    except Exception as e:
        import traceback
        print(f"Exception in port_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


def task_dro_exact(r_input, T, N_init, data, lb, ub, r_start, power,
                   list_inds, eps_init, alpha, m, train_size, test_size, init_ind,
                   interval, interval_SAA, t_list, foldername, checkpoint_every):
    """port_experiments from port_box_DRO_orig.py (exact DRO SOCP + SAA at epsnum==0)."""
    import os

    # Direct (CVXPY-free) Clarabel assembly of the DRO SOCP and the SAA LP:
    # default ON when importable; PORT_DRO_DIRECT=0 forces the legacy CVXPY
    # path for both branches.
    use_direct = os.environ.get('PORT_DRO_DIRECT', '1') != '0'
    if use_direct:
        try:
            from .direct_socp import BoxDROSocp, SaaCvarLp
        except Exception as e:
            print(f"[task_dro_exact] direct SOCP unavailable ({e}); "
                  "falling back to CVXPY", flush=True)
            use_direct = False
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            data, train_size=train_size, test_size=test_size, random_state=r_start + r)

        init_eps = eps_init[epsnum]
        num_dat = N_init
        a_const = -1.0 / (1.0 - alpha)
        direct_socp = BoxDROSocp(m, lb, ub, a=a_const) if use_direct else None
        direct_saa = SaaCvarLp(m, a=a_const) if use_direct else None

        DRO_x_current = np.ones(m) / m
        DRO_tau_current = 0.0
        SA_x_current = np.ones(m) / m
        SA_tau_current = 0.0

        history = {
            'DRO_x': [], 'DRO_tau': [], 'DRO_obj_values': [],
            'epsilon': [], 'DRO_computation_times': {'total_iteration': []},
            'SA_computation_times': [], 'SA_obj_values': [], 'SA_x': [], 'SA_tau': [],
            't': [],
        }

        ckpt_count = 0

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")
            radius = init_eps * (1.0 / (num_dat ** power))
            running_samples = dat[init_ind:(init_ind + num_dat)]

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    # Exact box-support W1-DRO SOCP (rebuilt for the current n).
                    if use_direct:
                        # Direct Clarabel assembly (no CVXPY canonicalization);
                        # same problem, settings, and solve_time semantics.
                        d_obj, d_x, d_tau, d_time, d_status = direct_socp.solve(
                            running_samples, (1.0 / num_dat) * np.ones(num_dat), radius)
                        if d_status in ('optimal', 'optimal_inaccurate'):
                            DRO_x_current = d_x
                            DRO_tau_current = d_tau
                            DRO_min_obj = d_obj
                            DRO_min_time = d_time
                        else:
                            print(f"[direct_socp] DRO_box_SOCP t={t} "
                                  f"status={d_status}", flush=True)
                            DRO_min_obj = np.nan
                            DRO_min_time = np.nan
                    else:
                        DRO_problem, DRO_x, DRO_tau, DRO_data, DRO_eps, DRO_w = createproblem_box_DRO(
                            num_dat, m, lb, ub, a=a_const)
                        DRO_data.value = running_samples
                        DRO_w.value = (1.0 / num_dat) * np.ones(num_dat)
                        DRO_eps.value = radius
                        if safe_solve(DRO_problem, name='DRO_box_SOCP', t=t,
                                      ignore_dpp=True, solver=cp.CLARABEL, verbose=False):
                            DRO_x_current = DRO_x.value
                            DRO_tau_current = float(DRO_tau.value)
                            DRO_min_obj = DRO_problem.objective.value
                            DRO_min_time = DRO_problem.solver_stats.solve_time
                        else:
                            DRO_min_obj = np.nan
                            DRO_min_time = np.nan

            if t % interval_SAA == 0 or ((t - 1) % interval_SAA == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    # SAA doesn't depend on epsilon, so only solve it once
                    # (epsnum == 0) and skip the redundant solve for every
                    # other epsilon in the sweep.
                    if epsnum == 0:
                        if use_direct:
                            # Direct Clarabel assembly of the SAA LP (no CVXPY
                            # canonicalization); same problem, settings, and
                            # solve_time semantics as create_scenario+Clarabel.
                            s_obj, s_xv, s_tauv, s_time, s_status = direct_saa.solve(
                                running_samples, (1.0 / num_dat) * np.ones(num_dat))
                            if s_status in ('optimal', 'optimal_inaccurate'):
                                SA_x_current = s_xv
                                SA_tau_current = s_tauv
                                SA_obj_current = s_obj
                                SA_time = s_time
                            else:
                                print(f"[direct_socp] SAA t={t} "
                                      f"status={s_status}", flush=True)
                                SA_obj_current = np.nan
                                SA_time = np.nan
                        else:
                            s_prob, s_x, s_tau = create_scenario(running_samples, m, num_dat, a=a_const)
                            if safe_solve(s_prob, name='SAA', t=t, solver=cp.CLARABEL, verbose=False):
                                SA_x_current = s_x.value
                                SA_tau_current = float(s_tau.value)
                                SA_obj_current = s_prob.objective.value
                                SA_time = s_prob.solver_stats.solve_time
                            else:
                                SA_obj_current = np.nan
                                SA_time = np.nan
                    else:
                        SA_x_current = np.full(m, np.nan)
                        SA_tau_current = np.nan
                        SA_obj_current = np.nan
                        SA_time = np.nan

                    history['DRO_computation_times']['total_iteration'].append(DRO_min_time)
                    history['DRO_x'].append(DRO_x_current)
                    history['DRO_tau'].append(DRO_tau_current)
                    history['DRO_obj_values'].append(DRO_min_obj)
                    history['epsilon'].append(radius)
                    history['t'].append(t)
                    history['SA_computation_times'].append(SA_time)
                    history['SA_x'].append(SA_x_current)
                    history['SA_tau'].append(SA_tau_current)
                    history['SA_obj_values'].append(SA_obj_current)

            num_dat += 1

            if t % interval_SAA == 0 or ((t - 1) % interval_SAA == 0) or (t in t_list):
                if t <= T or (t in t_list):
                    ckpt_count += 1
                    if ckpt_count % checkpoint_every == 0:
                        DRO_eval, DRO_satisfy, SA_eval, SA_satisfy = compute_cumulative_regret_dro(
                            history, dateval, m, a=a_const)
                        df = pd.DataFrame({
                            'DRO_tau': np.array(history['DRO_tau']),
                            'DRO_obj_values': np.array(history['DRO_obj_values']),
                            'epsilon': np.array(history['epsilon']),
                            'DRO_time': np.array(history['DRO_computation_times']['total_iteration']),
                            'DRO_eval1': DRO_eval[0], 'DRO_eval2': DRO_eval[1],
                            'DRO_satisfy1': DRO_satisfy[0], 'DRO_satisfy2': DRO_satisfy[1],
                            'SA_eval1': SA_eval[0], 'SA_eval2': SA_eval[1],
                            'SA_satisfy1': SA_satisfy[0], 'SA_satisfy2': SA_satisfy[1],
                            'SA_obj_values': np.array(history['SA_obj_values']),
                            'SA_time': np.array(history['SA_computation_times']),
                            't': np.array(history['t']),
                        })
                        df.to_csv(foldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        DRO_eval, DRO_satisfy, SA_eval, SA_satisfy = compute_cumulative_regret_dro(
            history, dateval, m, a=a_const)
        df = pd.DataFrame({
            'DRO_tau': np.array(history['DRO_tau']),
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time': np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval1': DRO_eval[0], 'DRO_eval2': DRO_eval[1],
            'DRO_satisfy1': DRO_satisfy[0], 'DRO_satisfy2': DRO_satisfy[1],
            'SA_eval1': SA_eval[0], 'SA_eval2': SA_eval[1],
            'SA_satisfy1': SA_satisfy[0], 'SA_satisfy2': SA_satisfy[1],
            'SA_obj_values': np.array(history['SA_obj_values']),
            'SA_time': np.array(history['SA_computation_times']),
            't': np.array(history['t']),
        })
        df.to_csv(foldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')
        return df
    except Exception as e:
        import traceback
        print(f"Exception in port_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
        traceback.print_exc(file=output_stream)
        try:
            return df
        except NameError:
            return None


MOSEK_PARAMS = {
    'MSK_IPAR_NUM_THREADS': 4,
    'MSK_DPAR_OPTIMIZER_MAX_TIME': 3600.0,
}


def run_true_saa(data, m, N_true, a, r, r_start, test_size, train_size):
    """Solve SAA CVaR portfolio on the first N_true samples for seed r.

    Uses the same train/test split as port_box_DRO_orig.py so that dateval
    is identical to the one used during online experiments.

    Returns a dict with keys:
        r, insample_obj, outsample_eval0, outsample_eval1,
        solve_time, converged
    """
    dat, dateval = train_test_split(
        data, train_size=train_size, test_size=test_size,
        random_state=r_start + r)

    # Take the first N_true points in the same streaming order as experiments.
    dat_true = dat[:N_true]

    t0 = time.time()
    prob, x_var, tau_var = create_scenario(dat_true, m, N_true, a=a)
    prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
               mosek_params=MOSEK_PARAMS)
    solve_time = time.time() - t0

    if x_var.value is None:
        print(f"  r={r}: solver failed (status: {prob.status})")
        return {
            'r': r,
            'insample_obj': np.nan,
            'outsample_eval0': np.nan,
            'outsample_eval1': np.nan,
            'solve_time': solve_time,
            'converged': False,
        }

    x = x_var.value
    tau = float(tau_var.value)
    insample = float(prob.objective.value)
    # Two disjoint 200-sample windows — same windows compute_cumulative_regret uses.
    oos0 = _expected_cost(dateval[0:200,   :m], x, tau, a=a)
    oos1 = _expected_cost(dateval[200:400, :m], x, tau, a=a)

    print(f"  r={r}: in-sample={insample:.6f}  oos0={oos0:.6f}  oos1={oos1:.6f}"
          f"  time={solve_time:.1f}s")
    return {
        'r': r,
        'insample_obj': insample,
        'outsample_eval0': oos0,
        'outsample_eval1': oos1,
        'solve_time': solve_time,
        'converged': True,
    }
