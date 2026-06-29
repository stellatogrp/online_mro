import argparse
import os
import sys
import time

# Ensure local package imports work when run from SLURM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cvxpy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import itertools

from utils import createproblem_portLP, createproblem_worstcase_p2, get_n_processes, gradient_step, safe_solve, save_run_metadata
from utils import compute_cumulative_regret_dro as compute_cumulative_regret
from utils import create_scenario_dro as create_scenario


output_stream = sys.stdout


def port_experiments(r_input,T,N_init,synthetic_returns,eta_0,r_start):
    try:
        r,epsnum = list_inds[r_input]
        np.random.seed(r_start+r)
        dat, dateval = train_test_split(
            synthetic_returns[:, :m], train_size=19000, test_size=1000, random_state=r_start+r)
        # dat_indices = np.random.choice(48000,48000,replace=False)
        # dat = dat[dat_indices]

        init_eps = eps_init[epsnum]
        num_dat = N_init

        # Saddle-point scheme for the DRO solve: maintain (x, tau) iterates updated
        # by one projected-subgradient step per interval using Danskin gradients
        # from the inner worst-case dual. Step size eta_t = eta_0 / sqrt(t+1).
        a_const = -5
        D_x = np.sqrt(2)
        R = np.linalg.norm(dateval, axis=1).mean()
        L_x = abs(a_const) * R
        # eta_0 = D_x / L_x
        DRO_x_current = np.ones(m) / m
        DRO_tau_current = 0.0
        # Pre-seed SAA iterate so a solver failure doesn't leave it unbound.
        SA_x_current = np.ones(m) / m
        SA_tau_current = 0.0

        # History for analysis
        history = {
            'x': [],
            'tau': [],
            'obj_values': [],
            'MRO_x': [],
            'MRO_tau': [],
            'MRO_obj_values': [],
            'DRO_x': [],
            'DRO_tau': [],
            'DRO_obj_values': [],
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
            'DRO_computation_times':{
            'total_iteration':[]
            },
            'distances':[],
            'mean_val':[],
            'square_val': [],
            'sig_val': [],
            'mean_val_MRO':[],
            'square_val_MRO': [],
            'sig_val_MRO': [],
            'SA_computation_times':[],
            'SA_obj_values':[],
            'SA_x': [],
            'SA_tau':[],
            "satisfy":[],
            "MRO_satisfy":[],
            "DRO_satisfy":[],
            "SA_eval":[],
            "SA_satisfy":[],
            't':[]
        }


        for t in range(T):
            print(f"\nTimestep {t+1}/{T}")

            radius = init_eps*(1/(num_dat**(1/(40))))
            running_samples = dat[init_ind:(init_ind+num_dat)]

            if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list) :
                if t <= 2001 or (t in t_list):
                # solve DRO problem
                    if t == 0:
                        # One LP warm-start (cardinality dropped) to seed (x, tau).
                        DRO_problem, DRO_x, DRO_s, DRO_tau, DRO_lmbda, DRO_data, DRO_eps, DRO_w = createproblem_portLP(num_dat,m)
                        DRO_data.value = running_samples
                        DRO_w.value = (1/num_dat)*np.ones(num_dat)
                        DRO_eps.value = radius
                        if safe_solve(DRO_problem, name='DRO_problem(t=0)', t=t,
                                      ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                            DRO_x_current = DRO_x.value
                            DRO_tau_current = DRO_tau.value
                            DRO_min_obj = DRO_problem.objective.value
                            DRO_min_time = DRO_problem.solver_stats.solve_time
                        else:
                            DRO_min_obj = np.nan
                            DRO_min_time = np.nan
                    else:
                        # Solve worst-case dual at current iterate, then one gradient step.
                        DRO_wc_problem, DRO_p_var, DRO_z_var, DRO_x_star, DRO_tau_star, \
                            DRO_wc_data, DRO_wc_eps, DRO_wc_w = createproblem_worstcase_p2(num_dat, m)
                        DRO_wc_data.value = running_samples
                        DRO_wc_eps.value = radius**2
                        DRO_wc_w.value = (1/num_dat)*np.ones(num_dat)
                        DRO_x_star.value = DRO_x_current
                        DRO_tau_star.value = DRO_tau_current
                        if safe_solve(DRO_wc_problem, name='DRO_wc_problem_p2', t=t,
                                      ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                            p_opt = DRO_p_var.value
                            z_opt = DRO_z_var.value
                            eta = eta_0 / np.sqrt(t + 1)
                            F_curr_dro = DRO_wc_problem.objective.value

                            def _inner_eval_dro(x_val, tau_val):
                                DRO_x_star.value = x_val
                                DRO_tau_star.value = tau_val
                                if not safe_solve(DRO_wc_problem, name='DRO_wc_problem_p2(inner_eval)', t=t,
                                                  ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                                    return np.inf
                                return DRO_wc_problem.objective.value

                            grad_start = time.time()
                            DRO_x_current, DRO_tau_current = gradient_step(
                                DRO_x_current, DRO_tau_current, p_opt, z_opt, eta, a=a_const,
                                line_search=line_search,
                                inner_eval=_inner_eval_dro,
                                F_curr=F_curr_dro,
                            )
                            grad_time = time.time() - grad_start
                            DRO_min_obj = F_curr_dro
                            # Worst-case update time = inner dual solve + subgradient
                            # step (including any line-search re-solves).
                            DRO_min_time = DRO_wc_problem.solver_stats.solve_time + grad_time
                        else:
                            DRO_min_obj = np.nan
                            DRO_min_time = np.nan


            if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0) or (t in t_list)  :
                if t <= 2001 or (t in t_list):
                    s_prob, s_x, s_tau = create_scenario(running_samples,m,num_dat)
                    if safe_solve(s_prob, name='SAA', t=t,
                                  ignore_dpp=True, solver=cp.CLARABEL, verbose=False, time_limit=1500.0):
                        SA_x_current = s_x.value
                        SA_tau_current = s_tau.value
                        SA_obj_current = s_prob.objective.value
                        SA_time = s_prob.solver_stats.solve_time
                    else:
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

            # New sample
            new_sample = dat[init_ind+num_dat]
            # q_dict, k_dict, weight_update_time = online_cluster_update(K,new_sample, q_dict, k_dict,num_dat, t, fixed_time)
            num_dat += 1
            # history['online_computation_times']['weight_update'].append(weight_update_time)
            # history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)


            if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0) or (t in t_list)  :
                if t <= 2001 or (t in t_list):

                    DRO_eval, DRO_satisfy,SA_eval, SA_satisfy = compute_cumulative_regret(
                    history, dateval, m)

                    df = pd.DataFrame({
                    # 'DRO_x': history['DRO_x'],
                    'DRO_tau': np.array(history['DRO_tau']),
                    'DRO_obj_values': np.array(history['DRO_obj_values']),
                    'epsilon': np.array(history['epsilon']),
                    'DRO_time':  np.array(history['DRO_computation_times']['total_iteration']),
                    'DRO_eval1': DRO_eval[0],
                    'DRO_eval2': DRO_eval[1],
                    # 'DRO_eval3': DRO_eval[2],
                    # 'DRO_eval4': DRO_eval[3],
                    "DRO_satisfy1": DRO_satisfy[0],
                    "DRO_satisfy2": DRO_satisfy[1],
                    # "DRO_satisfy3": DRO_satisfy[2],
                    # "DRO_satisfy4": DRO_satisfy[3],
                    'SA_eval1' : SA_eval[0],
                    'SA_eval2' : SA_eval[1],
                    # 'SA_eval3' : SA_eval[2],
                    # 'SA_eval4' : SA_eval[3],
                    'SA_satisfy1': SA_satisfy[0],
                    'SA_satisfy2': SA_satisfy[1],
                    # 'SA_satisfy3': SA_satisfy[2],
                    # 'SA_satisfy4': SA_satisfy[3],
                    'SA_obj_values': np.array(history['SA_obj_values']),
                    'SA_time':np.array(history['SA_computation_times']),
                    # 'SA_x': history['SA_x'],
                    'SA_tau': np.array(history['SA_tau']),
                    't':np.array(history['t'])
                    })
                    df.to_csv(foldername+'DRO'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
                    # print(f"Weights: {q_dict['w'], np.sum(q_dict['w']) }")


        DRO_eval, DRO_satisfy,SA_eval, SA_satisfy = compute_cumulative_regret(
                history, dateval, m)

        df = pd.DataFrame({
                # 'DRO_x': history['DRO_x'],
                'DRO_tau': np.array(history['DRO_tau']),
                'DRO_obj_values': np.array(history['DRO_obj_values']),
                'epsilon': np.array(history['epsilon']),
                'DRO_time':  np.array(history['DRO_computation_times']['total_iteration']),
                'DRO_eval1': DRO_eval[0],
                'DRO_eval2': DRO_eval[1],
                # 'DRO_eval3': DRO_eval[2],
                # 'DRO_eval4': DRO_eval[3],
                "DRO_satisfy1": DRO_satisfy[0],
                "DRO_satisfy2": DRO_satisfy[1],
                # "DRO_satisfy3": DRO_satisfy[2],
                # "DRO_satisfy4": DRO_satisfy[3],
                'SA_eval1' : SA_eval[0],
                'SA_eval2' : SA_eval[1],
                # 'SA_eval3' : SA_eval[2],
                # 'SA_eval4' : SA_eval[3],
                'SA_satisfy1': SA_satisfy[0],
                'SA_satisfy2': SA_satisfy[1],
                # 'SA_satisfy3': SA_satisfy[2],
                # 'SA_satisfy4': SA_satisfy[3],
                'SA_obj_values': np.array(history['SA_obj_values']),
                'SA_time':np.array(history['SA_computation_times']),
                # 'SA_x': history['SA_x'],
                'SA_tau': np.array(history['SA_tau']),
                't':np.array(history['t'])
                })
        df.to_csv(foldername+'DRO_new'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    parser.add_argument('--T', type=int, default=3001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=30)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--interval_SAA', type=int, default=500)

    parser.add_argument('--N_init', type=int, default=50)
    parser.add_argument('--r_start', type=int, default=0)
    parser.add_argument('--eta_0', type=float, default=0.01)
    parser.add_argument('--line_search', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Armijo backtracking line search inside gradient_step (default: on; pass --no-line_search to disable).')


    arguments = parser.parse_args()
    foldername = arguments.foldername
    R = arguments.R
    m = arguments.m
    T = arguments.T
    r_start = arguments.r_start

    interval = arguments.interval
    interval_SAA = arguments.interval_SAA
    N_init = arguments.N_init
    line_search = arguments.line_search
    eta_0 = arguments.eta_0
    K_arr = [5,15]
    foldername = foldername +'R'+str(R)+'_T'+str(T-1)+'/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datname = os.path.join(script_dir, 'synthetic_200_1.csv')
    synthetic_returns = pd.read_csv(datname
                                    ).to_numpy()[:, 1:][:,:m]

    init_ind = 0
    njobs = get_n_processes(100)
    if T >= 10000:
        eps_init = [0.003]
    else:
        eps_init = [0.005,0.004,0.003,0.002,0.001]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R),np.arange(M)))
    # mults = np.concatenate((5*np.ones(51),4*np.ones(50),3*np.ones(100),2*np.ones(100),1*np.ones(1000)))
    # mults = np.concatenate((8*np.ones(51),4*np.ones(50),2.5*np.ones(50),2*np.ones(50),1.7*np.ones(50),1.3*np.ones(50),1*np.ones(1000)))

    # dat, dateval = train_test_split(
    #     synthetic_returns[:, :m], train_size=48000, test_size=12000, random_state=50)
    t_list = [4,5,9,10,14,15,19,20,1249,1250,1499,1500,1749,1750,1999,2000]
    newdatname = foldername +'T'+str(T-1)+'R'+str(R)+'/'

    # Persist run metadata before any computation so it is on disk even if
    # the parallel sweep later crashes mid-experiment.
    save_run_metadata(
        {
            'filename': os.path.basename(__file__),
            'T': T, 'R': R, 'm': m,
            'interval': interval, 'interval_SAA': interval_SAA, 'N_init': N_init,
            'r_start': r_start, 'line_search': line_search,
            'eta_0': eta_0,
            'epsilon_values': [float(e) for e in eps_init],
            'num_epsilon_values': len(eps_init),
            'num_random_seeds': R,
            'total_test_combinations': len(eps_init) * R,
        },
        [foldername, newdatname],
    )

    results = Parallel(n_jobs=njobs)(delayed(port_experiments)(
        r_input,T,N_init,synthetic_returns,eta_0,r_start) for r_input in range(len(list_inds)))

    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r,epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))],ignore_index=True)
        findfs[r].to_csv(foldername + 'DRO_new_df_' + str(r+r_start) +'.csv')

    for r in range(R):
        # findfs[r] = findfs[r].drop(columns=["DRO_x","SA_x"])
        findfs[r].to_csv(newdatname + 'df_' + 'K'+str(0)+'R'+ str(r+r_start) +'.csv')

    print("DONE")