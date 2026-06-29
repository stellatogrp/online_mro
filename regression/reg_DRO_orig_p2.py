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
import itertools

from utils_p1 import (
    createproblem_hingeMIO_kappa,
    create_scenario_hinge,
    generate_classification_data,
    get_n_processes,
    MOSEK_PARAMS,
    save_run_metadata,
    worst_case_hinge_kappa,
)
from utils_p1 import compute_cumulative_regret_dro_hinge as compute_cumulative_regret


output_stream = sys.stdout


def reg_experiments(r_input, T, N_init, synthetic_data, power, kappa, p, r_start):
    """Full-data DRO sparse-SVM vs. SAA — finite label transport cost kappa.

    Transport cost: c((x,y),(x',y')) = ||x - x'||_2 + kappa * |y - y'|.
    The DRO problem no longer has a closed form; instead the Wasserstein dual
    is reformulated as an MI-SOCP (Corollary 3.12 of the writeup):

        min_{beta, z, lambda, s}  lambda*rho + (1/N) sum_i s_i
        s.t.  s_i >= 1 - y_i beta^T x_i
              s_i >= 1 + y_i beta^T x_i - 2*kappa*lambda
              s_i >= 0,  ||beta||_2 <= lambda,  lambda >= 0
              ||beta||_0 <= k  (big-M + binary z).

    kappa -> inf recovers the p=2 hinge DRO (labels fixed, penalty delta||beta||_2).
    """
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            synthetic_data[:, :m + 1], train_size=19000, test_size=1000, random_state=r_start + r)

        init_eps = eps_init[epsnum]
        num_dat = N_init

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

        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            radius = init_eps * (1 / (num_dat**power))
            running_samples = dat[init_ind:(init_ind + num_dat)]

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= 2001 or (t in t_list):
                    DRO_problem, DRO_beta, DRO_z, DRO_lmbda, DRO_s, DRO_data, DRO_eps, DRO_kappa, DRO_w = \
                        createproblem_hingeMIO_kappa(num_dat, m, k, p=p)
                    DRO_data.value = running_samples
                    DRO_w.value = (1 / num_dat) * np.ones(num_dat)
                    DRO_eps.value = radius
                    DRO_kappa.value = kappa
                    DRO_problem.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
                                      mosek_params=MOSEK_PARAMS)
                    DRO_x_current = DRO_beta.value
                    DRO_min_obj = DRO_problem.objective.value
                    DRO_min_time = DRO_problem.solver_stats.solve_time

            if t % interval_SAA == 0 or ((t - 1) % interval_SAA == 0) or (t in t_list):
                if t <= 2001 or (t in t_list):
                    s_prob, s_x, s_z = create_scenario_hinge(running_samples, m, num_dat, k)
                    s_prob.solve(solver=cp.MOSEK, ignore_dpp=True, verbose=False,
                                 mosek_params=MOSEK_PARAMS)
                    SA_x_current = s_x.value
                    SA_obj_current = s_prob.objective.value
                    SA_time = s_prob.solver_stats.solve_time

                    history['DRO_computation_times']['total_iteration'].append(DRO_min_time)
                    history['DRO_x'].append(DRO_x_current)
                    history['DRO_obj_values'].append(DRO_min_obj)
                    history['epsilon'].append(radius)
                    history['t'].append(t)

                    history['SA_computation_times'].append(SA_time)
                    history['SA_x'].append(SA_x_current)
                    history['SA_obj_values'].append(SA_obj_current)

            num_dat += 1

            if t % interval_SAA == 0 or ((t - 1) % interval_SAA == 0) or (t in t_list):
                if t <= 2001 or (t in t_list):

                    DRO_eval, DRO_satisfy, SA_eval, SA_satisfy = compute_cumulative_regret(
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
                    df.to_csv(foldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

        DRO_eval, DRO_satisfy, SA_eval, SA_satisfy = compute_cumulative_regret(
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
        df.to_csv(foldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv')

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    parser.add_argument('--T', type=int, default=2001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k_true', type=int, default=10)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--interval_SAA', type=int, default=100)
    parser.add_argument('--N_init', type=int, default=50)
    parser.add_argument('--r_start', type=int, default=0)
    parser.add_argument('--noise', type=float, default=3.0)
    parser.add_argument('--power', type=float, default=0.5)
    parser.add_argument('--kappa', type=float, default=1.0,
                        help='Label transport cost. kappa -> inf recovers infinite label cost (p=2 hinge).')
    parser.add_argument('--p', type=int, default=2)

    arguments = parser.parse_args()
    foldername = arguments.foldername
    R = arguments.R
    m = arguments.m
    k = arguments.k
    k_true = arguments.k_true
    T = arguments.T
    noise_std = arguments.noise
    r_start = arguments.r_start
    interval = arguments.interval
    interval_SAA = arguments.interval_SAA
    N_init = arguments.N_init
    power = arguments.power
    kappa = arguments.kappa
    p = arguments.p

    foldername = foldername + 'R' + str(R) + '_T' + str(T - 1) + '/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)

    synthetic_data, beta_true = generate_classification_data(
        n_total=20000, m=m, k_true=k_true, noise_std=noise_std, seed=12345)

    init_ind = 0
    njobs = get_n_processes(100)
    if T >= 10000:
        eps_init = [0.5]
    else:
        eps_init = [1.5, 1.0, 0.7, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R), np.arange(M)))
    t_list = [4, 5, 9, 10, 14, 15, 19, 20, 29, 30, 59, 60, 1249, 1250, 1499, 1500, 1749, 1750, 1999, 2000]
    newdatname = foldername + 'T' + str(T - 1) + 'R' + str(R) + '/'

    save_run_metadata(
        {
            'filename': os.path.basename(__file__),
            'T': T, 'R': R, 'm': m, 'k': k, 'k_true': k_true,
            'noise_std': noise_std,
            'interval': interval, 'interval_SAA': interval_SAA, 'N_init': N_init,
            'r_start': r_start,
            'power': power,
            'kappa': kappa,
            'p': p,
            'epsilon_values': [float(e) for e in eps_init],
            'num_epsilon_values': len(eps_init),
            'num_random_seeds': R,
            'total_test_combinations': len(eps_init) * R,
            'beta_true': [float(b) for b in beta_true],
        },
        [foldername, newdatname],
    )

    results = Parallel(n_jobs=njobs)(delayed(reg_experiments)(
        r_input, T, N_init, synthetic_data, power, kappa, p, r_start) for r_input in range(len(list_inds)))

    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r, epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))], ignore_index=True)
        findfs[r].to_csv(foldername + 'DRO_df_' + str(r + r_start) + '.csv')

    for r in range(R):
        findfs[r].to_csv(newdatname + 'df_' + 'K' + str(0) + 'R' + str(r + r_start) + '.csv')

    print("DONE")
