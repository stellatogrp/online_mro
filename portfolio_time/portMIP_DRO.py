import argparse
import os
import sys

import cvxpy as cp
import joblib
import mosek
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import itertools

output_stream = sys.stdout


def get_n_processes(max_n=np.inf):
    """Get number of processes from current cps number
    Parameters
    ----------
    max_n: int
        Maximum number of processes.
    Returns
    -------
    float
        Number of processes to use.
    """

    try:
        # Check number of cpus if we are on a SLURM server
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc


def createproblem_portMIP(N, m):
    """Create the problem in cvxpy, minimize CVaR
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable(m, boolean=True)
    tau = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # + cp.quad_over_lin(a*x, 4*lam)
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*dat@x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*x, 2) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    constraints += [x - z <= 0, cp.sum(z) <= 5]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w
    

def create_scenario(dat,m,num_dat):
    tau = cp.Variable()
    x = cp.Variable(m)
    z = cp.Variable(m, boolean=True)
    objective = cp.sum(tau + 5*cp.maximum(-dat@x - tau,0))/num_dat
    constraints = []
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [x - z <= 0, cp.sum(z) <= 5]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau
    

def compute_cumulative_regret(history,dateval):
    """
    Compute cumulative regret by comparing online decisions against optimal DRO solution in hindsight.
    At each time t, use the same samples that were available to the online policy.
    
    Args:
        history (dict): History of online decisions and parameters
        dro_params (DROParameters): Problem parameters
        online_samples (np.array): Array of observed samples
        num_eval_samples (int): Number of samples to use for SAA evaluation
        seed (int): Random seed for reproducibility
    """
    def evaluate_expected_cost(d_eval, x, tau):
        return np.mean(
            np.maximum(-5*d_eval@x - 4*tau, tau)) 
    
    DRO_e = []
    DRO_s = []
    SA_e = []
    SA_s = []
    T = len(history['t'])
    # Generate evaluation samples from true distribution for cost computation
    for j in range(1):
        DRO_eval_values = np.zeros(T)
        SA_eval_values = np.zeros(T)
        eval_samples = dateval[(j*1000):(j+1)*1000,:m]
        
        # For each timestep t
        for t in range(T):            
            # Compute out of sample values
            optimal_cost = evaluate_expected_cost(eval_samples, history['DRO_x'][t],history['DRO_tau'][t])
            SA_cost = evaluate_expected_cost(eval_samples, history['SA_x'][t],history['SA_tau'][t])
            DRO_eval_values[t] = optimal_cost
            SA_eval_values[t] = SA_cost

        DRO_satisfy = np.array(history['DRO_obj_values'] >= DRO_eval_values).astype(float)
        SA_satisfy = np.array(history['SA_obj_values'] >= SA_eval_values).astype(float)

        DRO_e.append(DRO_eval_values)
        DRO_s.append(DRO_satisfy)
        SA_e.append(SA_eval_values)
        SA_s.append(SA_satisfy)

    return DRO_e, DRO_s, SA_e, SA_s


def port_experiments(r_input,T,N_init,synthetic_returns,r_start):
    r,epsnum = list_inds[r_input]
    np.random.seed(r_start+r)
    dat, dateval = train_test_split(
         synthetic_returns[:, :m], train_size=57000, test_size=3000, random_state=r_start+r)
    # dat_indices = np.random.choice(48000,48000,replace=False) 
    # dat = dat[dat_indices]

    init_eps = eps_init[epsnum]
    num_dat = N_init

    # History for analysis
    history = {
        'DRO_x': [],
        'DRO_tau': [],
        'DRO_obj_values': [],
        'epsilon': [],
        'DRO_computation_times':{
        'total_iteration':[]
        },
        'distances':[],
        'SA_computation_times':[],
        'SA_obj_values':[],
        'SA_x': [],
        'SA_tau':[],
        "DRO_satisfy":[],
        "SA_eval":[],
        "SA_satisfy":[], 
        't':[]
    }


    for t in range(T):
        print(f"\nTimestep {t+1}/{T}")
        
        radius = init_eps*(1/(num_dat**(1/(2*m))))
        running_samples = dat[init_ind:(init_ind+num_dat)]
    
        if t % interval == 0 or ((t-1) % interval == 0)  :
        # solve DRO problem 
            DRO_problem, DRO_x, DRO_s, DRO_tau, DRO_lmbda, DRO_data, DRO_eps, DRO_w = createproblem_portMIP(num_dat,m)
            DRO_data.value = running_samples
            DRO_w.value = (1/num_dat)*np.ones(num_dat)
            DRO_eps.value = radius
            DRO_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                mosek.dparam.optimizer_max_time:  2000.0})
            DRO_x_current = DRO_x.value
            DRO_tau_current = DRO_tau.value
            DRO_min_obj = DRO_problem.objective.value
            DRO_min_time = DRO_problem.solver_stats.solve_time



        if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0)  :
            s_prob, s_x, s_tau = create_scenario(running_samples,m,num_dat)
            s_prob.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                    mosek.dparam.optimizer_max_time:  2000.0})
            SA_x_current = s_x.value
            SA_tau_current = s_tau.value
            SA_obj_current = s_prob.objective.value
            SA_time = s_prob.solver_stats.solve_time

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
        num_dat += 1
    

        if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0) :

            DRO_eval, DRO_satisfy,SA_eval, SA_satisfy = compute_cumulative_regret(
            history,dateval)
            
            df = pd.DataFrame({
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time':  np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval': DRO_eval[0],
            "DRO_satisfy": DRO_satisfy[0],
            'SA_eval' : SA_eval[0],
            'SA_satisfy': SA_satisfy[0],
            'SA_obj_values': np.array(history['SA_obj_values']),
            'SA_time':np.array(history['SA_computation_times']),
            't':np.array(history['t'])
            })
            # df.to_csv(foldername+'DRO'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')

    DRO_eval, DRO_satisfy,SA_eval, SA_satisfy = compute_cumulative_regret(
            history,dateval)
            
    df = pd.DataFrame({
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time':  np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval': DRO_eval[0],
            "DRO_satisfy1": DRO_satisfy[0],
            'SA_eval1' : SA_eval[0],
            'SA_satisfy1': SA_satisfy[0],
            'SA_obj_values': np.array(history['SA_obj_values']),
            'SA_time':np.array(history['SA_computation_times']),
            't':np.array(history['t'])
            })
    # df.to_csv(foldername+'DRO'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
  
    return df
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    parser.add_argument('--T', type=int, default=3001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=30)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--interval_SAA', type=int, default=100)

    parser.add_argument('--N_init', type=int, default=50)
    parser.add_argument('--r_start', type=int, default=0)


    arguments = parser.parse_args()
    foldername = arguments.foldername
    R = arguments.R
    m = arguments.m
    T = arguments.T
    r_start = arguments.r_start

    interval = arguments.interval
    interval_SAA = arguments.interval_SAA
    N_init = arguments.N_init
    foldername = foldername +'R'+str(R)+'_T'+str(T-1)+'/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)
    datname = 'portfolio_time/synthetic.csv'
    synthetic_returns = pd.read_csv(datname
                                    ).to_numpy()[:, 1:]
    init_ind = 0
    njobs = get_n_processes(100)
    if T >= 10000:
        eps_init = [0.00]
    else:
        eps_init = [0.005,0.004,0.003,0.002,0.0015,0.001]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R),np.arange(M)))
    
    results = Parallel(n_jobs=njobs)(delayed(port_experiments)(
        r_input,T,N_init,synthetic_returns,r_start) for r_input in range(len(list_inds)))
    
    # save dataframes
    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r,epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    newdatname = 'portfolio_exp/T'+str(T-1)+'R'+str(R)+'/'
    os.makedirs(newdatname, exist_ok=True)

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))],ignore_index=True)
        findfs[r].to_csv(newdatname + 'df_' + 'K'+str(0)+'R'+ str(r+r_start) +'.csv')
        
    print("DONE")