import argparse
import os
import sys

import cvxpy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import time
import itertools
import copy

from utils import (
    createproblem_portLP,
    dro_subgrad_step,
    fixed_cluster,
    get_n_processes,
    safe_solve,
    save_run_metadata,
    w2_dist,
    wasserstein,
    worst_case_value,
)
from utils import calc_cluster_val_online as calc_cluster_val
from utils import compute_cumulative_regret_online as compute_cumulative_regret
from utils import online_cluster_init_online as online_cluster_init
from utils import online_cluster_update_online as online_cluster_update


output_stream = sys.stdout


def port_experiments(r_input,K,T,N_init,synthetic_returns,eta_0, r_start, newfoldername):
    try:
        r,epsnum = list_inds[r_input]
        np.random.seed(r_start+r)
        dat, dateval = train_test_split(
             synthetic_returns[:, :m], train_size=19000, test_size=1000, random_state=r_start+r)
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
        init_radius_val = init_eps*(1/(num_dat**(1/40)))

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


        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            radius = init_eps*(1/(num_dat**(1/40)))
            running_samples = dat[init_ind:(init_ind+num_dat)]

            # solve online MRO problem
            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):
                online_dat = k_dict['d'][:num_dat]      # at most K micro-clusters
                online_w = k_dict['w'][:num_dat]

                grad_time = 0.0
                if t == 0:
                    # One LP warm-start (cardinality dropped) to seed (x, tau).
                    lp_problem, lp_x, lp_s, lp_tau, lp_lam, lp_dat, lp_eps, lp_w = \
                        createproblem_portLP(online_dat.shape[0], m)
                    lp_dat.value = online_dat
                    lp_eps.value = radius
                    lp_w.value = online_w
                    if safe_solve(lp_problem, name='lp_problem(online,t=0)', t=t,
                                  solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        x_current = lp_x.value
                        tau_current = lp_tau.value
                        min_obj = lp_problem.objective.value
                        min_time = lp_problem.solver_stats.solve_time
                    else:
                        # Warm-start failed; keep the default initial iterate.
                        min_obj = np.nan
                        min_time = np.nan
                else:
                    # Closed-form worst-case value + Danskin (sub)gradient, then
                    # one projected step (no inner CVXPY solve; strong duality).
                    eta = eta_0 / np.sqrt(t + 1)
                    grad_start = time.time()
                    x_current, tau_current, min_obj = dro_subgrad_step(
                        x_current, tau_current, online_dat, online_w, radius, eta,
                        a=a_const, line_search=line_search,
                    )
                    grad_time = time.time() - grad_start
                    min_time = grad_time


                # Store timing information
                history['online_computation_times']['min_problem'].append(min_time)
                history['online_computation_times']['gradient_step'].append(grad_time)
                history['online_computation_times']['total_iteration'].append(min_time+weight_update_time)
                history['online_computation_times']['weight_update'].append(weight_update_time)
                history['t'].append(t)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):
                # solve MRO problem with new clusters
                if t <= fixed_time:
                    start_time = time.time()
                    cur_K = np.minimum(K,num_dat)
                    if new_k_dict is not None and (num_dat > (interval+N_init)) and new_k_dict['d'].shape[0] == cur_K:
                        kmeans = KMeans(n_clusters=cur_K, init=new_k_dict['d'],n_init=1).fit(running_samples)
                    else:
                        kmeans = KMeans(n_clusters=cur_K,init="k-means++", n_init=1).fit(running_samples)
                    new_centers = kmeans.cluster_centers_
                    wk = np.bincount(kmeans.labels_) / num_dat
                    cluster_time = time.time()-start_time  if K < num_dat else 0.0
                    new_k_dict = {}
                    new_k_dict['K'] = cur_K
                    new_k_dict['data'] = {}
                    new_k_dict['a'] = new_centers
                    new_k_dict['d'] = new_centers
                    new_k_dict['w'] = wk
                    for k in range(K):
                        new_k_dict['data'][k] = running_samples[kmeans.labels_==k]
                    new_k_dict['d'] = new_centers


                cur_K_mro = new_k_dict['d'].shape[0]

                MRO_grad_time = 0.0
                if t == 0:
                    # One LP warm-start on clustered data to seed MRO (x, tau).
                    lp_problem, lp_x, lp_s, lp_tau, lp_lam, lp_dat, lp_eps, lp_w = \
                        createproblem_portLP(cur_K_mro, m)
                    lp_dat.value = new_k_dict['d']
                    lp_eps.value = radius
                    lp_w.value = new_k_dict['w']
                    if safe_solve(lp_problem, name='lp_problem(MRO,t=0)', t=t,
                                  solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        MRO_x_current = lp_x.value
                        MRO_tau_current = lp_tau.value
                        MRO_min_obj = lp_problem.objective.value
                        MRO_min_time = lp_problem.solver_stats.solve_time
                    else:
                        MRO_min_obj = np.nan
                        MRO_min_time = np.nan
                else:
                    # Closed-form worst-case value + Danskin (sub)gradient, then
                    # one projected step (no inner CVXPY solve; strong duality).
                    eta = eta_0 / np.sqrt(t + 1)
                    grad_start = time.time()
                    MRO_x_current, MRO_tau_current, MRO_min_obj = dro_subgrad_step(
                        MRO_x_current, MRO_tau_current, new_k_dict['d'], new_k_dict['w'],
                        radius, eta, a=a_const, line_search=line_search,
                    )
                    MRO_grad_time = time.time() - grad_start
                    MRO_min_time = MRO_grad_time

                mean_val_mro, square_val_mro, sig_val_mro = calc_cluster_val(K, new_k_dict,num_dat,MRO_x_current,running_samples)

                history['MRO_computation_times']['min_problem'].append(MRO_min_time)
                history['MRO_computation_times']['gradient_step'].append(MRO_grad_time)
                history['MRO_computation_times']['total_iteration'].append(MRO_min_time+cluster_time)
                history['MRO_computation_times']['clustering'].append(cluster_time)
                history['MRO_weights'].append(new_k_dict['w'])


            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):
                # compute online MRO worst value (wrt non clustered data)
                uni_w = (1/num_dat)*np.ones(num_dat)
                t0 = time.time()
                new_worst = worst_case_value(x_current, tau_current, running_samples, uni_w, radius)
                worst_time = time.time() - t0

                history['worst_values'].append(new_worst)
                history['worst_times'].append(worst_time)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):
                t0 = time.time()
                new_worst_MRO = worst_case_value(MRO_x_current, MRO_tau_current, running_samples, uni_w, radius)
                MRO_worst_time = time.time() - t0

                mean_val, square_val, sig_val = calc_cluster_val(K, k_dict,num_dat,x_current,running_samples)

                history['MRO_worst_values'].append(new_worst_MRO)
                history['MRO_worst_times'].append(MRO_worst_time)


            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):
                # compute online worst value (wrt prev stage sols
                t0 = time.time()
                new_worst = worst_case_value(x_prev, tau_prev, running_samples, uni_w, radius)
                worst_time = time.time() - t0

                history['worst_values_regret'].append(new_worst)
                history['worst_times_regret'].append(worst_time)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):
                t0 = time.time()
                new_worst_MRO = worst_case_value(MRO_x_prev, MRO_tau_prev, running_samples, uni_w, radius)
                MRO_worst_time = time.time() - t0

                history['MRO_worst_values_regret'].append(new_worst_MRO)
                history['MRO_worst_times_regret'].append(MRO_worst_time)

                MRO_x_prev = MRO_x_current
                MRO_tau_prev = MRO_tau_current
                x_prev = x_current
                tau_prev = tau_current


            # New sample
            new_sample = dat[init_ind+num_dat]
            q_dict, k_dict, weight_update_time = online_cluster_update(K, new_sample, q_dict, k_dict, num_dat, t, fixed_time, m, Q, rmse_mult)
            if t >= fixed_time:
                new_k_dict, cluster_time = fixed_cluster(new_k_dict, new_sample, num_dat=num_dat, m=m)
            num_dat += 1
            # history['online_computation_times']['weight_update'].append(weight_update_time)
            # history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):
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

            if (t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)) and (t <= 2001 or (t in t_list)):

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

            # Plot regret analysis
            # plot_regret_analysis(
            #       cumulative_regret,
            #       instantaneous_regret,theo,MRO_cum_regret,MRO_regret
            #   )

            #   # After all other plots
            # plot_computation_times(history)

            # plot_eval(eval, MRO_eval, DRO_eval, SA_eval, history)

            # plot_computation_times_iter(history)

        return df
    except Exception as e:
        import traceback
        print(f"Exception in port_experiments (r_input={locals().get('r_input', None)}): {e}", file=output_stream)
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
    parser.add_argument('--T', type=int, default=3001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=30)
    parser.add_argument('--Q', type=int, default=500)
    parser.add_argument('--fixed_time', type=int, default=1500)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--N_init', type=int, default=50)
    parser.add_argument('--rmse_mult', type=float, default=1.25)
    parser.add_argument('--r_start', type=int, default=0)
    parser.add_argument('--line_search', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Armijo backtracking line search inside gradient_step (default: on; pass --no-line_search to disable).')
    parser.add_argument('--eta_0', type=float, default=0.01)


    arguments = parser.parse_args()
    foldername = arguments.foldername
    K = arguments.K
    R = arguments.R
    m = arguments.m
    Q = arguments.Q
    T = arguments.T
    eta_0 = arguments.eta_0
    r_start = arguments.r_start
    fixed_time = arguments.fixed_time
    interval = arguments.interval
    N_init = arguments.N_init
    rmse_mult = arguments.rmse_mult
    line_search = arguments.line_search
    K_arr = [15,25,30]
    K = K_arr[idx]
    newfoldername = foldername + 'K'+str(K)+'_R'+str(R)+'_T'+str(T-1)+'/'
    os.makedirs(newfoldername, exist_ok=True)
    print(newfoldername)
    # Resolve the synthetic data relative to this script so sbatch cwd does not matter.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datname = os.path.join(script_dir, 'synthetic_200_1.csv')
    synthetic_returns = pd.read_csv(datname
                                    ).to_numpy()[:, 1:][:,:m]
    init_ind = 0
    njobs = get_n_processes(100)
    #eps_init = [0.006,0.005,0.004,0.0035,0.003,0.0025,0.002,0.0015,0.001]
    #eps_init = [0.0085,0.008,0.007,0.006,0.005,0.0045,0.004,0.0035]
    eps_init = [0.03,0.02,0.015,0.009,0.008,0.007,0.006,0.005]
    if T >= 5000:
        eps_init = [0.0035,0.003,0.0025,0.002]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R),np.arange(M)))
    # dat, dateval = train_test_split(
    #     synthetic_returns[:, :m], train_size=48000, test_size=12000, random_state=50)
    t_list = [4,5,9,10,14,15,19,20,1249,1250,1499,1500,1749,1750,1999,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    newdatname = foldername +'T'+str(T-1)+'R'+str(R)+'/'

    # Persist run metadata before any computation so it is on disk even if
    # the parallel sweep later crashes mid-experiment.
    save_run_metadata(
        {
            'filename': os.path.basename(__file__),
            'K': K, 'T': T, 'R': R, 'm': m, 'Q': Q,
            'fixed_time': fixed_time, 'interval': interval, 'N_init': N_init,
            'rmse_mult': rmse_mult,
            'r_start': r_start, 'line_search': line_search,
            'eta_0': eta_0,
            'epsilon_values': [float(e) for e in eps_init],
            'num_epsilon_values': len(eps_init),
            'num_random_seeds': R,
            'total_test_combinations': len(eps_init) * R,
        },
        [newfoldername, (newdatname, f'metadata_K{K}.json')],
    )

    results = Parallel(n_jobs=njobs)(delayed(port_experiments)(
        r_input,K,T,N_init,synthetic_returns, eta_0,r_start, newfoldername) for r_input in range(len(list_inds)))

    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r,epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))],ignore_index=True)
        findfs[r].to_csv(newfoldername + 'df_' + str(r+r_start) +'.csv')

    for r in range(R):
        findfs[r] = findfs[r].drop(columns=['weights','MRO_weights'])
        findfs[r].to_csv(newdatname + 'df_' + 'K'+str(K)+'R'+ str(r+r_start) +'.csv')

    print("DONE")
