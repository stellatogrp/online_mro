import argparse
import os
import sys

# Ensure local package imports work when run from SLURM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import itertools

from utils_p1 import (
    generate_classification_data,
    get_n_processes,
    save_run_metadata,
    HingeGurobiCG,
)
from utils_p1 import compute_cumulative_regret_dro_hinge as compute_cumulative_regret


output_stream = sys.stdout


def reg_experiments(r_input, T, N_init, synthetic_data, r_start):
    """Full-data DRO sparse-SVM vs. SAA SVM (p = 1) -- Gurobi *constraint generation*.

    Same problem as ``reg_DRO_orig_p1_gurobi.py`` but instead of rebuilding the
    best-subset MILP from scratch at every interval, a single persistent Gurobi
    model is kept alive per method (DRO and SAA) and **grown incrementally**: the
    (beta, z) best-subset structure is built once and each newly streamed sample
    contributes a single hinge epigraph constraint ``s_i >= 1 - y_i beta^T x_i``
    (``HingeGurobiCG.ensure_rows``).  Re-solving only rewrites the objective
    coefficients (the sample weights and the radius delta) and Gurobi warm-starts
    from the retained incumbent.  Because ``running_samples`` is a cumulative
    prefix, every previously added hinge row stays valid, so this reproduces the
    from-scratch solution while avoiding the repeated model build.
    """
    try:
        r, epsnum = list_inds[r_input]
        np.random.seed(r_start + r)
        dat, dateval = train_test_split(
            synthetic_data[:, :m + 1], train_size=19000, test_size=1000, random_state=r_start + r)

        init_eps = eps_init[epsnum]
        num_dat = N_init

        # Pre-seed iterates so a solver failure at the first interval still
        # leaves them defined for the history append.
        DRO_x_current = np.zeros(m)
        SA_x_current = np.zeros(m)

        # Persistent constraint-generation models (one per method): the hinge
        # rows accumulate across timesteps, the objective is rewritten per solve.
        dro_cg = HingeGurobiCG(m, k)
        saa_cg = HingeGurobiCG(m, k)

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

            # radius represents delta (order-1); RWPI picks delta_n ~ 1/sqrt(n)
            # so the induced ell_1 penalty is statistically principled.
            radius = init_eps * (1 / np.sqrt(num_dat))
            running_samples = dat[init_ind:(init_ind + num_dat)]
            unif_w = (1 / num_dat) * np.ones(num_dat)

            if t % interval == 0 or ((t - 1) % interval == 0) or (t in t_list):
                if t <= 2001 or (t in t_list):
                    # DRO sparse-SVM: append the newly arrived hinge rows, then
                    # re-solve the warm-started model with the current (w, delta).
                    dro_cg.ensure_rows(running_samples)
                    DRO_x_current, DRO_min_obj, DRO_min_time = dro_cg.solve(unif_w, delta=radius)

            if t % interval_SAA == 0 or ((t - 1) % interval_SAA == 0) or (t in t_list):
                if t <= 2001 or (t in t_list):
                    # non-robust SAA best subset = same model with delta = 0
                    saa_cg.ensure_rows(running_samples)
                    SA_x_current, SA_obj_current, SA_time = saa_cg.solve(unif_w, delta=0.0)

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
    parser.add_argument('--m', type=int, default=50)        # covariate dimension d
    parser.add_argument('--k', type=int, default=10)         # cardinality budget
    parser.add_argument('--k_true', type=int, default=10)    # true sparsity
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--interval_SAA', type=int, default=100)
    parser.add_argument('--N_init', type=int, default=50)
    parser.add_argument('--r_start', type=int, default=0)
    parser.add_argument('--noise', type=float, default=3.0)

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

    foldername = foldername + 'R' + str(R) + '_T' + str(T - 1) + '/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)

    # Synthetic classification data: one fixed dataset; each seed draws its own
    # train/test split below.
    synthetic_data, beta_true = generate_classification_data(
        n_total=20000, m=m, k_true=k_true, noise_std=noise_std, seed=12345)

    init_ind = 0
    njobs = get_n_processes(100)
    # eps_init values are the delta prefactors; radius = delta = init_eps / sqrt(n).
    if T >= 10000:
        eps_init = [0.5]
    else:
        eps_init = [1.5, 1.0, 0.7, 0.5, 0.3, 0.1]
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
            'epsilon_values': [float(e) for e in eps_init],
            'num_epsilon_values': len(eps_init),
            'num_random_seeds': R,
            'total_test_combinations': len(eps_init) * R,
            'beta_true': [float(b) for b in beta_true],
            'solver': 'gurobi-constraint-generation',
        },
        [foldername, newdatname],
    )

    results = Parallel(n_jobs=njobs)(delayed(reg_experiments)(
        r_input, T, N_init, synthetic_data, r_start) for r_input in range(len(list_inds)))

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
