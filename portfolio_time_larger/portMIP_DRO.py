import argparse
import os
import sys

import cvxpy as cp
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from sklearn.metrics import mean_squared_error
import math
import itertools
import gurobipy
import mosek

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
    constraints += [cp.norm(a*x, 1) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    constraints += [x - z <= 0, cp.sum(z) <= 30]
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
    constraints += [x - z <= 0, cp.sum(z) <= 30]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau
    

def find_min_pairwise_distance(data):
    distances = distance.cdist(data, data)
    np.fill_diagonal(distances, np.inf)  # set diagonal to infinity to ignore self-distances
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    return min_indices

def online_cluster_init(K,Q,data):
    start_time = time.time()
    k_dict = {}
    q_dict = {}
    init_num = data.shape[0]
    cur_Q =np.minimum(Q,init_num)
    q_dict['cur_Q'] = cur_Q
    qmeans = KMeans(n_clusters=q_dict['cur_Q']).fit(data)
    q_dict['a'] = np.zeros((Q+1,m))
    q_dict['d'] = np.zeros((Q+1,m))
    q_dict['w'] = np.zeros(Q+1)
    q_dict['rmse'] = np.zeros(Q+1)
    q_dict['a'][:cur_Q,:] = qmeans.cluster_centers_
    q_dict['d'][:cur_Q,:] = qmeans.cluster_centers_
    q_dict['w'][:cur_Q] = np.bincount(qmeans.labels_) / init_num
    q_dict['rmse'][:cur_Q] = np.zeros(q_dict['cur_Q'])
    total_time = time.time() - start_time
    q_dict['data'] = {}
    for q in range(q_dict['cur_Q']):
        cluster_data = data[qmeans.labels_ == q]
        q_dict['data'][q] = cluster_data
        centroid_array = np.tile(q_dict['d'][q], (len(cluster_data), 1))
        rmse = np.sqrt(mean_squared_error(cluster_data, centroid_array))
        if rmse <= 1e-6:
            rmse = 0.001
        q_dict['rmse'][q] = rmse
    k_dict = {}
    k_dict['a'] = np.zeros((K,m))
    k_dict['w'] = np.zeros(K)
    k_dict['d'] = np.zeros((K,m))
    k_dict['data'] = {}
    k_dict, t_time = cluster_k(K,q_dict, k_dict, init=True)
    return q_dict, k_dict, total_time + t_time

def cluster_k(K,q_dict, k_dict, init=False):
    start_time = time.time()
    cur_K = np.minimum(K,q_dict['cur_Q'])
    cur_Q = q_dict['cur_Q']
    k_dict['K'] = cur_K
    if init:
        kmeans = KMeans(n_clusters=cur_K, init='k-means++', n_init=1).fit(q_dict['a'][:cur_Q,:])
    else:
        kmeans = KMeans(n_clusters=cur_K, init=k_dict['a'], n_init=1).fit(q_dict['a'][:cur_Q,:])
    k_dict['a'] = kmeans.cluster_centers_
    # k_dict['w'] = np.zeros(cur_K)
    # k_dict['d'] = np.zeros((cur_K,m))
    # k_dict['data'] = {}
    for k in range(cur_K):
        k_dict[k]= np.where(kmeans.labels_ == k)[0]
        d_cur = q_dict['d'][:cur_Q,:][kmeans.labels_ == k]
        w_cur = q_dict['w'][:cur_Q][kmeans.labels_ == k]
        k_dict['w'][k] = np.sum(w_cur)
        w_cur_norm = w_cur/(k_dict['w'][k])
        k_dict['d'][k] = np.sum(d_cur*w_cur_norm[:,np.newaxis],axis=0)
    total_time = time.time() - start_time
    for k in range(cur_K):
        k_dict['data'][k] = np.vstack([q_dict['data'][q] for q in k_dict[k]])
    return k_dict, total_time

def online_cluster_update(K,new_dat, q_dict, k_dict,num_dat, t, fix_time):
    new_dat = np.reshape(new_dat,(1,m))
    if t >= fix_time:
        k_dict, total_time = fixed_cluster(k_dict,new_dat,num_dat)
        return q_dict, k_dict, total_time
    cur_Q = q_dict['cur_Q']
    start_time = time.time()
    dists = cdist(new_dat,q_dict['a'][:cur_Q,:])
    min_dist = np.min(dists)
    min_ind = np.argmin(dists)
    if min_dist <= 2*q_dict['rmse'][min_ind]:
        q_dict['d'][min_ind] = (q_dict['d'][min_ind]*q_dict['w'][min_ind]*num_dat + new_dat)/(q_dict['w'][min_ind]*num_dat + 1)
        q_dict['rmse'][min_ind] = np.sqrt((q_dict['rmse'][min_ind]**2*q_dict['w'][min_ind]*num_dat + np.linalg.norm(new_dat - q_dict['d'][min_ind],2)**2)/(q_dict['w'][min_ind]*num_dat + 1))
        w_q_temp = q_dict['w'][:cur_Q]*num_dat/(num_dat+1)
        increased_w = (q_dict['w'][min_ind]*num_dat + 1)/(num_dat+1)
        q_dict['w'][:cur_Q] = w_q_temp
        q_dict['w'][min_ind] = increased_w
        for k in range(K):
            if min_ind in k_dict[k]:
                k_dict['d'][k] = (k_dict['d'][k]*k_dict['w'][k]*num_dat + new_dat)/(k_dict['w'][k]*num_dat + 1)
                k_dict['w'][k] = (k_dict['w'][k]*num_dat + 1)/(num_dat + 1)
            else:
                k_dict['w'][k] = (k_dict['w'][k]*num_dat)/(num_dat + 1)
        total_time = time.time() - start_time
        q_dict['data'][min_ind] = np.vstack([q_dict['data'][min_ind],new_dat])
        for k in range(K):
            if min_ind in k_dict[k]:
                k_dict['data'][k] = np.vstack([k_dict['data'][k],new_dat])
    else:
        start_time = time.time()
        cur_Q = q_dict['cur_Q'] + 1
        q_dict['cur_Q'] = cur_Q
        q_dict['a'][cur_Q-1] = new_dat
        q_dict['d'][cur_Q-1] = new_dat
        q_dict['rmse'][cur_Q-1] = 2*np.min(q_dict['rmse'])
        q_dict['w'][:cur_Q-1] = (q_dict['w'][:cur_Q-1]*num_dat)/(num_dat+1)
        q_dict['w'][cur_Q-1] = 1/(num_dat+1)
        total_time = time.time() - start_time
        q_dict['data'][cur_Q-1] = new_dat
        if cur_Q > Q:
            start_time = time.time()
            q_dict['cur_Q'] = Q
            min_pair = find_min_pairwise_distance(q_dict['a'])
            merged_weight = np.sum(q_dict['w'][min_pair[0]]+q_dict['w'][min_pair[1]])
            merged_center = (q_dict['rmse'][min_pair[0]]*q_dict['w'][min_pair[0]] + q_dict['rmse'][min_pair[1]]*q_dict['w'][min_pair[1]])/merged_weight
            merged_centroid = (q_dict['d'][min_pair[0]]*q_dict['w'][min_pair[0]] + q_dict['d'][min_pair[1]]*q_dict['w'][min_pair[1]])/merged_weight
            merged_rmse = np.sqrt((q_dict['rmse'][min_pair[0]]**2*q_dict['w'][min_pair[0]] + q_dict['rmse'][min_pair[1]]**2*q_dict['w'][min_pair[1]])/merged_weight + (q_dict['w'][min_pair[0]]*np.linalg.norm( q_dict['d'][min_pair[0]]- merged_centroid)**2 + q_dict['w'][min_pair[1]]*np.linalg.norm(q_dict['d'][min_pair[1]]- merged_centroid)**2)/(merged_weight ))
            q_dict['a'][min_pair[0]] = merged_center
            q_dict['d'][min_pair[0]] = merged_centroid
            q_dict['w'][min_pair[0]] = merged_weight
            q_dict['rmse'][min_pair[0]] = merged_rmse
            q_dict['a'][min_pair[1]] = q_dict['a'][Q]
            q_dict['d'][min_pair[1]] = q_dict['d'][Q]
            q_dict['w'][min_pair[1]] = q_dict['w'][Q]
            q_dict['rmse'][min_pair[1]] = q_dict['rmse'][Q]
            total_time += time.time() - start_time
            merged_data = np.vstack([q_dict['data'][q] for q in min_pair])
            q_dict['data'][min_pair[0]] = merged_data
            q_dict['data'][min_pair[1]] = q_dict['data'][Q]
        k_dict, time_temp = cluster_k(K,q_dict,k_dict)
        total_time += time_temp
    return q_dict, k_dict, total_time

            
def fixed_cluster(k_dict, new_dat,num_dat):
    new_dat = np.reshape(new_dat,(1,m))
    start_time = time.time()
    dists = cdist(new_dat,k_dict['a'])
    min_ind = np.argmin(dists)
    k_dict['d'][min_ind] = (k_dict['d'][min_ind]*k_dict['w'][min_ind]*num_dat + new_dat)/(k_dict['w'][min_ind]*num_dat + 1)
    w_k_temp = k_dict['w']*num_dat/(num_dat+1)
    increased_w = (k_dict['w'][min_ind]*num_dat + 1)/(num_dat+1)
    k_dict['w'] = w_k_temp
    k_dict['w'][min_ind] = increased_w
    total_time = time.time() - start_time
    k_dict['data'][min_ind] = np.vstack([k_dict['data'][min_ind],new_dat])
    return k_dict, total_time


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
    for j in range(4):
        DRO_eval_values = np.zeros(T)
        SA_eval_values = np.zeros(T)
        eval_samples = dateval[(j*3000):(j+1)*3000,:m]
    # For each timestep t
        for t in range(T):            
            # Compute instantaneous regret at time t using true distribution
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

def plot_regret_analysis(cumulative_regret, regret, theo, MRO_cumulative_regret, MRO_regret):
    """Plot regret analysis results with LaTeX formatting and log scales."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    
    # Create figure with 2x2 subplots

    T = len(cumulative_regret)
    t_range = np.arange(T)
    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(t_range, cumulative_regret, 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, MRO_cumulative_regret, 'r-', linewidth=2, label = "reclustering")
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Cumulative Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'regret_analysis_cumulative.pdf', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(t_range, regret, 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, MRO_regret, 'r-', linewidth=2, label = "reclustering")
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Instantaneous Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'regret_analysis_inst.pdf', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1,1, figsize=(9, 4), dpi=300)
    ax.plot(t_range, cumulative_regret, 'b-', linewidth=2, label = "actual cumulative regret")
    ax.plot(t_range, theo, 'r-', linewidth=2, label = "theoretical regret")
    # axins = zoomed_inset_axes(ax, 6, loc="lower right")
    # axins.set_xlim(3700, 4000)
    # axins.set_ylim(7, 10)
    # axins.plot(t_range, cumulative_regret, 'b-',linewidth=2)
    # axins.set_xticks(ticks=[])
    # axins.set_yticks(ticks=[])
    # mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
    ax.set_xlabel(r'Time step $(t)$')
    ax.set_ylabel(r'Cumulative Regret')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'regret_analysis_comp.pdf', bbox_inches='tight', dpi=300)


def plot_eval(eval, MRO_eval, DRO_eval,SA_eval, history):
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 16
    })
    T = len(eval)
    t_range = np.arange(T)
    plt.figure(figsize=(7, 4), dpi=300)
    plt.plot(t_range, eval, 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, MRO_eval, 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, SA_eval, 'g-', linewidth=2, label = "SAA")
    plt.plot(t_range, DRO_eval, color ='black', linewidth=2, label = "DRO")
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Evaluation value (out of sample)')
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'eval_analysis.pdf', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(7, 4), dpi=300)
    plt.plot(t_range, history['obj_values'], 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, history['MRO_obj_values'], 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, history['SA_obj_values'], 'g-', linewidth=2, label = "SAA")
    plt.plot(t_range, history['DRO_obj_values'], color ='black', linewidth=2, label = "DRO")
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Objective value (in sample)')
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'obj_analysis.pdf', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(7, 4), dpi=300)
    plt.plot(t_range, history['obj_values'], 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, history['MRO_obj_values'], 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, history['DRO_obj_values'], color ='black', linewidth=2, label = "DRO")
    plt.plot(t_range, history['SA_obj_values'], 'g-', linewidth=2, label = "SAA")
    plt.plot(t_range, eval,  'b', linewidth=2, linestyle='-.')
    plt.plot(t_range, MRO_eval, 'r', linewidth=2, linestyle='-.')
    plt.plot(t_range, DRO_eval, color ='black', linestyle='-.')
    plt.plot(t_range, SA_eval,'g', linestyle='-.')
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Objective value and evaluation value')
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'obj_eval_analysis.pdf', bbox_inches='tight', dpi=300)



def plot_results(history):
    """Plot results with LaTeX formatting."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    
    # Create figure with higher DPI
    plt.figure(figsize=(11, 4), dpi=300)

    # Plot 2: Epsilon Evolution
    plt.subplot(121)
    plt.plot(history['epsilon'], 'r-', linewidth=2)
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'$\epsilon$')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Ball Weights
    plt.subplot(122)
    plt.plot(np.array(history['weights']), linewidth=2)
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Ball Weights')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    plt.savefig(foldername+'radius.pdf', bbox_inches='tight', dpi=300)


def plot_computation_times(history):
    """Plot computation time analysis with LaTeX formatting."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    
    # Create figure
    plt.figure(figsize=(15, 3), dpi=300)
    
    # Prepare data for boxplot
    data = [
        history['online_computation_times']['total_iteration'], history['MRO_computation_times']['total_iteration'],history['DRO_computation_times']['total_iteration'] 
    ]
    # np.save("online",history['online_computation_times']['total_iteration'])
    # np.save("mro",history['MRO_computation_times']['total_iteration'])
    # np.save("dro",history['DRO_computation_times']['total_iteration'])

    # Create boxplot
    bp = plt.boxplot(data, labels=[

        r'online clustering', r'reclustering', r'DRO' 
    ])
    
    # Customize boxplot colors
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='blue')
    plt.setp(bp['caps'], color='blue')
    plt.setp(bp['medians'], color='red')

    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'Compuation time')
    plt.yscale("log")
    plt.savefig(foldername+'time.pdf', bbox_inches='tight', dpi=300)

def plot_computation_times_iter(history):
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    t_range = np.arange(len( history['online_computation_times']['total_iteration']))
    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(t_range, history['online_computation_times']['total_iteration'], 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, history['MRO_computation_times']['total_iteration'], 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, history['DRO_computation_times']['total_iteration'], color ='black', linewidth=2, label = "DRO")
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Compuation time')
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.savefig(foldername+'time_iters.pdf', bbox_inches='tight', dpi=300)

def calc_cluster_val(K,k_dict, num_dat,x):
    mean_val = 0
    square_val = 0
    sig_val = 0
    for k in range(K):
        centroid = k_dict['d'][k]
        for dat in k_dict['data'][k]:
            cur_val = np.linalg.norm(dat-centroid,2)
            mean_val += cur_val
            square_val += cur_val**2
            sig_val = np.maximum(sig_val,(dat-centroid)@x)
    return mean_val/num_dat, square_val/num_dat, sig_val

def port_experiments(r_input,T,N_init,dateval,r_start):
    r,epsnum = list_inds[r_input]
    np.random.seed(r_start+r)
    datname = '/scratch/gpfs/iywang/mro_mpc/synthetic/synthetic_200_2.csv'
    dat = pd.read_csv(datname
                                    ).to_numpy()[:, 1:][:, :m]
    dat_indices = np.random.choice(20000,20000,replace=False) 
    dat = dat[dat_indices]

    init_eps = eps_init[epsnum]
    num_dat = N_init

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
        
        radius = init_eps*(1/(num_dat**(1/(30))))
        running_samples = dat[init_ind:(init_ind+num_dat)]
    
        if t % interval == 0 or ((t-1) % interval == 0) :
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



        if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0) :
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
        # q_dict, k_dict, weight_update_time = online_cluster_update(K,new_sample, q_dict, k_dict,num_dat, t, fixed_time)
        num_dat += 1
        # history['online_computation_times']['weight_update'].append(weight_update_time)
        # history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)
    

        if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0) :

            DRO_eval, DRO_satisfy,SA_eval, SA_satisfy = compute_cumulative_regret(
            history,dateval)
            
            df = pd.DataFrame({
            'DRO_x': history['DRO_x'],
            'DRO_tau': np.array(history['DRO_tau']),
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time':  np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval1': DRO_eval[0],
            'DRO_eval2': DRO_eval[1],
            'DRO_eval3': DRO_eval[2],
            'DRO_eval4': DRO_eval[3],
            "DRO_satisfy1": DRO_satisfy[0],
            "DRO_satisfy2": DRO_satisfy[1],
            "DRO_satisfy3": DRO_satisfy[2],
            "DRO_satisfy4": DRO_satisfy[3],
            'SA_eval1' : SA_eval[0],
            'SA_eval2' : SA_eval[1],
            'SA_eval3' : SA_eval[2],
            'SA_eval4' : SA_eval[3],
            'SA_satisfy1': SA_satisfy[0],
            'SA_satisfy2': SA_satisfy[1],
            'SA_satisfy3': SA_satisfy[2],
            'SA_satisfy4': SA_satisfy[3],
            'SA_obj_values': np.array(history['SA_obj_values']),
            'SA_time':np.array(history['SA_computation_times']),
            'SA_x': history['SA_x'],
            'SA_tau': np.array(history['SA_tau']),
            't':np.array(history['t'])
            })
            df.to_csv(foldername+'DRO'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
            # print(f"Weights: {q_dict['w'], np.sum(q_dict['w']) }")
        

    DRO_eval, DRO_satisfy,SA_eval, SA_satisfy = compute_cumulative_regret(
            history,dateval)
            
    df = pd.DataFrame({
            'DRO_x': history['DRO_x'],
            'DRO_tau': np.array(history['DRO_tau']),
            'DRO_obj_values': np.array(history['DRO_obj_values']),
            'epsilon': np.array(history['epsilon']),
            'DRO_time':  np.array(history['DRO_computation_times']['total_iteration']),
            'DRO_eval1': DRO_eval[0],
            'DRO_eval2': DRO_eval[1],
            'DRO_eval3': DRO_eval[2],
            'DRO_eval4': DRO_eval[3],
            "DRO_satisfy1": DRO_satisfy[0],
            "DRO_satisfy2": DRO_satisfy[1],
            "DRO_satisfy3": DRO_satisfy[2],
            "DRO_satisfy4": DRO_satisfy[3],
            'SA_eval1' : SA_eval[0],
            'SA_eval2' : SA_eval[1],
            'SA_eval3' : SA_eval[2],
            'SA_eval4' : SA_eval[3],
            'SA_satisfy1': SA_satisfy[0],
            'SA_satisfy2': SA_satisfy[1],
            'SA_satisfy3': SA_satisfy[2],
            'SA_satisfy4': SA_satisfy[3],
            'SA_obj_values': np.array(history['SA_obj_values']),
            'SA_time':np.array(history['SA_computation_times']),
            'SA_x': history['SA_x'],
            'SA_tau': np.array(history['SA_tau']),
            't':np.array(history['t'])
            })
    df.to_csv(foldername+'DRO'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
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
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    parser.add_argument('--T', type=int, default=3001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=190)
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
    K_arr = [5,15]
    foldername = foldername +'R'+str(R)+'_T'+str(T-1)+'/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)
    datname = '/scratch/gpfs/iywang/mro_mpc/synthetic/synthetic_200_1.csv'
    dateval = pd.read_csv(datname
                                    ).to_numpy()[:, 1:][:, :m]
    init_ind = 0
    njobs = get_n_processes(100)
    if T >= 10000:
        eps_init = [0.003]
    else:
        eps_init = [0.007,0.006,0.005,0.004,0.003,0.002,0.0015,0.001,0.0005]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R),np.arange(M)))
    
    # dat, dateval = train_test_split(
    #     synthetic_returns[:, :m], train_size=48000, test_size=12000, random_state=50)
    results = Parallel(n_jobs=njobs)(delayed(port_experiments)(
        r_input,T,N_init,dateval,r_start) for r_input in range(len(list_inds)))
    
    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r,epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))],ignore_index=True)
        findfs[r].to_csv(foldername + 'DRO_df_' + str(r+r_start) +'.csv')

    newdatname = '/scratch/gpfs/iywang/mro_mpc/portfolio_exp_larger/T'+str(T-1)+'R'+str(R)+'/'
    os.makedirs(newdatname, exist_ok=True)
    for r in range(R):
        findfs[r] = findfs[r].drop(columns=["DRO_x","SA_x"])
        findfs[r].to_csv(newdatname + 'df_' + 'K'+str(0)+'R'+ str(r+r_start) +'.csv')
    
    print("DONE")