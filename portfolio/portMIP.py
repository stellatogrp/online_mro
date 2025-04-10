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
import ot
import itertools
import copy

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
    constraints += [x - z <= 0, cp.sum(z) <= 8]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w

def worst_case(N,m,dat):
    """Problem to solve with fixed x
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    dat: array
        Data samples
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5
    tau = cp.Parameter()
    x = cp.Parameter(m)

    # VARIABLES #
    # weights, s_i, lambda, tau
    s = cp.Variable(N)
    lam = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # + cp.quad_over_lin(a*x, 4*lam)
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*dat@x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*x, 2) <= lam]
    constraints += [lam >= 0]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, s, lam, x, tau, eps, w
    

def wasserstein(samples_p, samples_q):
    """
    Compute the Wasserstein-1 distance between two multi-dimensional empirical distributions.

    Parameters:
        samples_p (np.array): Samples from distribution P, shape (N, D).
        samples_q (np.array): Samples from distribution Q, shape (M, D).

    Returns:
        float: The Wasserstein-1 distance.
    """
    # Ensure the input arrays are 2D
    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)

    # Number of samples in each distribution
    N = samples_p.shape[0]
    M = samples_q.shape[0]

    # Create uniform weights for the samples
    weights_p = np.ones(N) / N  # Uniform weights for P
    weights_q = np.ones(M) / M  # Uniform weights for Q

    # Compute the cost matrix (pairwise Euclidean distances)
    cost_matrix = ot.dist(samples_p, samples_q, metric='euclidean')

    # Compute the Wasserstein-1 distance
    w_distance = ot.emd2(weights_p, weights_q, cost_matrix)

    return w_distance

def w2_dist(k1,k2):
    """calculates the wasserstein distance between the two empirical distributions with up to K atoms"""
    K = k2['K']
    val = 0
    for k in range(K):
        val += np.abs(k1["w"][k] - k2["w"][k])*np.linalg.norm(k1["d"][k] - k2["d"][k])
    if k1['K']>K:
        dists = cdist(k1['d'][K].reshape((1,m)),k2['d'][:K])
        val += dists@np.abs(k2['w'][:K] - k1['w'][:K])
    return float(val)

    
def calc_rmse(dat,mean):
    """calculates the rmse of a cluster"""
    rmse = 0
    for d in dat:
        rmse += np.linalg.norm(d-mean,2)**2
    return rmse

def find_min_pairwise_distance(data):
    """find the minimum pairwise distance of entries of an array"""
    distances = distance.cdist(data, data)
    np.fill_diagonal(distances, np.inf)  # set diagonal to infinity to ignore self-distances
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    return min_indices

def online_cluster_init(K,Q,data):
    """Initialize the online clustering algorithm, with K macroclusters
    and Q microclusters. """
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
        rmse = np.sqrt(calc_rmse(cluster_data,np.reshape(q_dict['d'][q],(1,m))))
        if rmse <= 1e-6:
            rmse = 0.04
        q_dict['rmse'][q] = rmse
    k_dict = {}
    k_dict['a'] = np.zeros((K,m))
    k_dict['w'] = np.zeros(K)
    k_dict['d'] = np.zeros((K,m))
    k_dict['data'] = {}
    k_dict['K'] = np.minimum(K,init_num)
    k_dict, t_time = cluster_k(K,q_dict, k_dict, init=True)
    return q_dict, k_dict, total_time + t_time

def cluster_k(K,q_dict, k_dict, init=False):
    """Find K macroclusters using the Q microclusters """
    start_time = time.time()
    cur_K = np.minimum(K,q_dict['cur_Q'])
    cur_Q = q_dict['cur_Q']
    k_dict['K'] = cur_K
    if init or (cur_Q<=K):
        kmeans = KMeans(n_clusters=cur_K, init='k-means++', n_init=1).fit(q_dict['d'][:cur_Q,:])
    else:
        kmeans = KMeans(n_clusters=cur_K, init=k_dict['a'], n_init=1).fit(q_dict['d'][:cur_Q,:])
    k_dict['a'] = kmeans.cluster_centers_
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
    """Update the online clustering algorithm with the new datapoint. If t is greater than fix_time, update only the K macroclusters."""
    cur_K = k_dict['K']
    new_dat = np.reshape(new_dat,(1,m))
    # if greater than threshold, only update K macroclusters
    if t >= fix_time:
        k_dict, total_time = fixed_cluster(k_dict,new_dat,num_dat)
        return q_dict, k_dict, total_time
    cur_Q = q_dict['cur_Q']
    start_time = time.time()
    dists = cdist(new_dat,q_dict['d'][:cur_Q,:])
    min_dist = np.min(dists)
    min_ind = np.argmin(dists)
    # if the new datapoint is close to existing clusters, add it
    if min_dist <= 2*q_dict['rmse'][min_ind] and cur_K == K:
        q_dict['d'][min_ind] = (q_dict['d'][min_ind]*q_dict['w'][min_ind]*num_dat + new_dat)/(q_dict['w'][min_ind]*num_dat + 1)
        q_dict['rmse'][min_ind] = np.sqrt((q_dict['rmse'][min_ind]**2*q_dict['w'][min_ind]*num_dat + np.linalg.norm(new_dat - q_dict['d'][min_ind],2)**2)/(q_dict['w'][min_ind]*num_dat + 1))
        w_q_temp = q_dict['w'][:cur_Q]*num_dat/(num_dat+1)
        increased_w = (q_dict['w'][min_ind]*num_dat + 1)/(num_dat+1)
        q_dict['w'][:cur_Q] = w_q_temp
        q_dict['w'][min_ind] = increased_w
        for k in range(cur_K):
            if min_ind in k_dict[k]:
                k_dict['d'][k] = (k_dict['d'][k]*k_dict['w'][k]*num_dat + new_dat)/(k_dict['w'][k]*num_dat + 1)
                k_dict['w'][k] = (k_dict['w'][k]*num_dat + 1)/(num_dat + 1)
            else:
                k_dict['w'][k] = (k_dict['w'][k]*num_dat)/(num_dat + 1)
        total_time = time.time() - start_time
        q_dict['data'][min_ind] = np.vstack([q_dict['data'][min_ind],new_dat])
        for k in range(cur_K):
            if min_ind in k_dict[k]:
                k_dict['data'][k] = np.vstack([k_dict['data'][k],new_dat])
    else:
        # create a new microclsuter
        start_time = time.time()
        cur_Q = q_dict['cur_Q'] + 1
        q_dict['cur_Q'] = cur_Q
        q_dict['a'][cur_Q-1] = new_dat
        q_dict['d'][cur_Q-1] = new_dat
        q_dict['rmse'][cur_Q-1] = min_dist
        q_dict['w'][:cur_Q-1] = (q_dict['w'][:cur_Q-1]*num_dat)/(num_dat+1)
        q_dict['w'][cur_Q-1] = 1/(num_dat+1)
        total_time = time.time() - start_time
        q_dict['data'][cur_Q-1] = new_dat
        if cur_Q > Q:
            # merge existing microclusters
            start_time = time.time()
            q_dict['cur_Q'] = Q
            min_pair = find_min_pairwise_distance(q_dict['a'])
            merged_weight = np.sum(q_dict['w'][min_pair[0]]+q_dict['w'][min_pair[1]])
            merged_center = (q_dict['a'][min_pair[0]]*q_dict['w'][min_pair[0]] + q_dict['a'][min_pair[1]]*q_dict['w'][min_pair[1]])/merged_weight
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
        # redo K macroclusters
        k_dict, time_temp = cluster_k(K,q_dict,k_dict)
        total_time += time_temp
    return q_dict, k_dict, total_time

            
def fixed_cluster(k_dict, new_dat,num_dat):
    """Update K macroclusters only"""
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



def compute_eval(history,dateval):
    """
        Compute evaluation values
        At each time t, use the evaluation dataset
        Args:
            history (dict): History of online decisions and parameters
            dateval (np.array): Array of evalution samples
        """
    def evaluate_expected_cost(d_eval, x, tau):
        return np.mean(
            np.maximum(-5*d_eval@x - 4*tau, tau)) 
    
    MRO_e = []
    MRO_s = []
    online_e = []
    online_s = []
    online_ws = []
    MRO_ws = []

    T = len(history['t'])
    for j in range(1):
        eval_values = np.zeros(T)
        MRO_eval_values = np.zeros(T)
        eval_samples = dateval[(j*200):(j+1)*200,:m]
        for t in range(T):            
            # Compute out-of-sample costs
            online_cost = evaluate_expected_cost(eval_samples, history['x'][t],history['tau'][t])
            MRO_cost = evaluate_expected_cost(eval_samples, history['MRO_x'][t],history['MRO_tau'][t])
            eval_values[t] = online_cost
            MRO_eval_values[t] = MRO_cost

        MRO_satisfy = np.array(history['MRO_obj_values'] >= MRO_eval_values).astype(float)
        satisfy = np.array(history['obj_values'] >= eval_values).astype(float)
        worst_satisfy = np.array( np.array(history['obj_values']) + 5*np.array(history["sig_val"])>= eval_values).astype(float)
        MRO_worst_satisfy = np.array(np.array(history['MRO_obj_values']) + 5*np.array(history["sig_val_MRO"])>= MRO_eval_values).astype(float)

        MRO_e.append(MRO_eval_values)
        MRO_s.append(MRO_satisfy)
        online_e.append(eval_values)
        online_s.append(satisfy)
        online_ws.append(worst_satisfy)
        MRO_ws.append(MRO_worst_satisfy)
    
    return MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws

def calc_cluster_val(K,k_dict, num_dat,x,running_samples):
    """Compute clustering distances"""
    mean_val = 0
    square_val = 0
    sig_val = 0
    cur_K = np.minimum(K,num_dat)
    for k in range(cur_K):
        centroid = k_dict['d'][k]
        for dat in k_dict['data'][k]:
            cur_val = np.linalg.norm(dat-centroid,2)
            mean_val += cur_val
            square_val += cur_val**2
            sig_val += max(0,(dat-centroid)@x)
    cost_matrix = ot.dist(running_samples, k_dict['d'][:cur_K], metric='euclidean')
    w_distance = ot.emd2(np.ones(num_dat)/num_dat, k_dict['w'][:cur_K], cost_matrix)
    return w_distance, square_val/num_dat, sig_val/num_dat

def port_experiments(r_input,K,T,N_init,synthetic_returns,r_start):
    """One round of the experiment, with a certain seed
        Parameters
        ----------
        r_input: int
            The index of the experiment. controls the epsilon value
        r_start: int
            controls the seed
        K: int
            Number of clusters
        T: int
            Maximum time
        N_init:
            Number of datapoints to start with
        synthetic_return
            training + testing data
        Returns
        -------
        The dataframe with all key metrics
        """
    r,epsnum = list_inds[r_input]
    np.random.seed(r_start+r)
    dat, dateval = train_test_split(
         synthetic_returns[:, :m], train_size=19000, test_size=1000, random_state=r_start+r)
    init_eps = eps_init[epsnum]
    num_dat = N_init
    q_dict, k_dict,weight_update_time= online_cluster_init(K,Q,dat[init_ind:(init_ind+num_dat)])
    k_dict_prev = copy.deepcopy(k_dict)
    new_k_dict_prev = copy.deepcopy(k_dict)
    new_k_dict = None
    init_samples = dat[init_ind:(init_ind+N_init)]


    # Initialize solutions
    MRO_x_prev = np.zeros(m)
    MRO_tau_prev = 0
    tau_prev = 0
    x_prev = np.zeros(m)
    init_radius_val = init_eps*(1/(num_dat**(1/(40))))
    online_problem, online_x, online_s, online_tau, online_lmbda, data_train, eps_train, w_train = createproblem_portMIP(np.minimum(num_dat,K), m)

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
        
        radius = init_eps*(1/(num_dat**(1/(40))))
        running_samples = dat[init_ind:(init_ind+num_dat)]

        # solve online MRO problem
        if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list):
            if t <= 1001 or (t in t_list):
                if num_dat <= K or data_train.shape[0] < K:
                    cur_K = np.minimum(num_dat,K)
                    online_problem, online_x, online_s, online_tau, online_lmbda, data_train, eps_train, w_train = createproblem_portMIP(cur_K, m)
                data_train.value = k_dict['d'][:num_dat]
                eps_train.value = radius
                w_train.value = k_dict['w'][:num_dat]
                online_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                    mosek.dparam.optimizer_max_time:  1500.0})
                x_current = online_x.value
                tau_current = online_tau.value
                min_obj = online_problem.objective.value
                min_time = online_problem.solver_stats.solve_time
                

                # Store timing information
                history['online_computation_times']['min_problem'].append(min_time)
                history['online_computation_times']['total_iteration'].append(min_time+weight_update_time)
                history['online_computation_times']['weight_update'].append(weight_update_time)
                history['t'].append(t)

        if t % interval == 0 or ((t-1) % interval == 0)  or (t in t_list) :
            if t <= 1001 or (t in t_list):
                # solve MRO problem with new clusters
                if t <= fixed_time:
                    start_time = time.time()
                    cur_K = np.minimum(K,num_dat)
                    if new_k_dict is not None and (num_dat > (interval+N_init)):
                        kmeans = KMeans(n_clusters=cur_K, init=new_k_dict['d'],n_init=1).fit(running_samples)
                    else:
                        print("restart kmeans", cur_K, num_dat)
                        kmeans = KMeans(n_clusters=cur_K,init="k-means++", n_init=1).fit(running_samples)
                    new_centers = kmeans.cluster_centers_
                    wk = np.bincount(kmeans.labels_) / num_dat
                    cluster_time = time.time()-start_time
                    new_k_dict = {}
                    new_k_dict['K'] = cur_K
                    new_k_dict['data'] = {}
                    new_k_dict['a'] = new_centers
                    new_k_dict['d'] = new_centers
                    new_k_dict['w'] = wk
                    for k in range(K):
                        new_k_dict['data'][k] = running_samples[kmeans.labels_==k]
                    new_k_dict['d'] = new_centers
                    

                data_train.value = new_k_dict['d']
                w_train.value = new_k_dict['w']
                # eps_train.value = new_radius
                online_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                    mosek.dparam.optimizer_max_time:  1500.0})
                MRO_x_current = online_x.value
                MRO_tau_current = online_tau.value
                MRO_min_obj = online_problem.objective.value
                MRO_min_time = online_problem.solver_stats.solve_time
                mean_val_mro, square_val_mro, sig_val_mro = calc_cluster_val(K, new_k_dict,num_dat,MRO_x_current,running_samples)
            
                history['MRO_computation_times']['min_problem'].append(MRO_min_time)
                history['MRO_computation_times']['total_iteration'].append(MRO_min_time+cluster_time)
                history['MRO_computation_times']['clustering'].append(cluster_time)
                history['MRO_weights'].append(new_k_dict['w'])

    
        if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list) :
            if t <= 1001 or (t in t_list):
                # compute online MRO worst value (wrt non clustered data)

                new_problem, s_d, lam_d, x_d, tau_d, eps_d, w_d =  worst_case(num_dat,m,running_samples)
                w_d.value = (1/num_dat)*np.ones(num_dat)
                eps_d.value = radius
                x_d.value = x_current
                tau_d.value = tau_current
                new_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                    mosek.dparam.optimizer_max_time:  1500.0})
                new_worst = new_problem.objective.value
                worst_time = new_problem.solver_stats.solve_time
                
                history['worst_values'].append(new_worst)
                history['worst_times'].append(worst_time)
                
            if t % interval == 0 or ((t-1) % interval == 0)  or (t in t_list) :
                if t <= 1001 or (t in t_list):
                    x_d.value = MRO_x_current
                    tau_d.value = MRO_tau_current
                    new_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                        mosek.dparam.optimizer_max_time:  1500.0})
                    new_worst_MRO = new_problem.objective.value
                    MRO_worst_time = new_problem.solver_stats.solve_time

                    mean_val, square_val, sig_val = calc_cluster_val(K, k_dict,num_dat,x_current,running_samples)

                    history['MRO_worst_values'].append(new_worst_MRO)
                    history['MRO_worst_times'].append(MRO_worst_time)


        if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list) :
            if t <= 1001 or (t in t_list):
                # compute online worst value (wrt prev stage sols)
                x_d.value = x_prev
                tau_d.value = tau_prev
                new_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                    mosek.dparam.optimizer_max_time:  1500.0})
                new_worst = new_problem.objective.value
                worst_time = new_problem.solver_stats.solve_time
                
                history['worst_values_regret'].append(new_worst)
                history['worst_times_regret'].append(worst_time)
            
        if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)  :
            if t <= 1001 or (t in t_list):
                x_d.value = MRO_x_prev
                tau_d.value = MRO_tau_prev
                new_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                    mosek.dparam.optimizer_max_time:  1500.0})
                new_worst_MRO = new_problem.objective.value
                MRO_worst_time = new_problem.solver_stats.solve_time

                history['MRO_worst_values_regret'].append(new_worst_MRO)
                history['MRO_worst_times_regret'].append(MRO_worst_time)
                
                MRO_x_prev = MRO_x_current
                MRO_tau_prev = MRO_tau_current
                x_prev = x_current
                tau_prev = tau_current


        # New sample
        new_sample = dat[init_ind+num_dat]
        q_dict, k_dict, weight_update_time = online_cluster_update(K,new_sample, q_dict, k_dict,num_dat, t, fixed_time)
        if t >= fixed_time:
            new_k_dict, cluster_time = fixed_cluster(new_k_dict,new_sample,num_dat=num_dat)
        num_dat += 1

        if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list) :
            if t <= 1001 or (t in t_list):
                N_dist_cur = wasserstein(init_samples,running_samples)
                
                history['regret_K'].append(w2_dist(k_dict,k_dict_prev)+ 2*radius )
                history['MRO_regret_K'].append(w2_dist(new_k_dict,new_k_dict_prev)+ 2*radius)
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


        if t % interval == 0 or ((t-1) % interval == 0) or (t in t_list)  :
            if t <= 1001 or (t in t_list):

                MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = compute_eval(
                history,dateval)
                
                df = pd.DataFrame({
                # 'x': history['x'],
                'tau': np.array(history['tau']),
                'obj_values': np.array(history['obj_values']),
                # 'MRO_x': history['MRO_x'],
                'MRO_tau':np.array(history['MRO_tau']),
                'MRO_obj_values': np.array(history['MRO_obj_values']),
                'epsilon': np.array(history['epsilon']),
                # 'weights':  history['weights'],
                # 'MRO_weights': history['MRO_weights'],
                # 'weights_q': history['weights_q'],
                'online_time':  np.array(history['online_computation_times']['total_iteration']),
                'MRO_time':  np.array(history['MRO_computation_times']['total_iteration']),
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
                    for j in range(1):
                        df[colnames[i]+str(j)] = np.array(colvals[i][j])
                # df.to_csv(foldername+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
        
    MRO_e, MRO_s, online_e, online_s, online_ws, MRO_ws = compute_eval(
            history,dateval)
            
    df = pd.DataFrame({
    'obj_values': np.array(history['obj_values']),
    'MRO_obj_values': np.array(history['MRO_obj_values']),
    'epsilon': np.array(history['epsilon']),
    # 'weights':  history['weights'],
    # 'MRO_weights': history['MRO_weights'],
    'online_time':  np.array(history['online_computation_times']['total_iteration']),
    'MRO_time':  np.array(history['MRO_computation_times']['total_iteration']),
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
        for j in range(1):
            df[colnames[i]+str(j)] = np.array(colvals[i][j])
    # df.to_csv(foldername+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')

    return df
    
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="portfolio_exp/", metavar='N')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--T', type=int, default=2001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--Q', type=int, default=500)
    parser.add_argument('--fixed_time', type=int, default=1500)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--N_init', type=int, default=5)
    parser.add_argument('--r_start', type=int, default=0)

    arguments = parser.parse_args()
    foldername = arguments.foldername
    K = arguments.K
    R = arguments.R
    m = arguments.m
    Q = arguments.Q
    T = arguments.T
    r_start = arguments.r_start
    fixed_time = arguments.fixed_time
    interval = arguments.interval
    N_init = arguments.N_init
    datname = 'portfolio/synthetic_200_1.csv'
    synthetic_returns = pd.read_csv(datname
                                    ).to_numpy()[:, 1:][:,:m]
    init_ind = 0
    njobs = get_n_processes(100)
    eps_init = [0.0045,0.004,0.0035,0.003,0.0025,0.002,0.0015]
    if T >= 5000:
        eps_init = [0.0035,0.003,0.0025,0.002]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R),np.arange(M)))
    t_list = [4,5,9,10,14,15,19,20,1249,1250,1499,1500,1749,1750,1999,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    results = Parallel(n_jobs=njobs)(delayed(port_experiments)(
        r_input,K,T,N_init,synthetic_returns,r_start) for r_input in range(len(list_inds)))
    
    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r,epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    newdatname = foldername + '/T'+str(T-1)+'/'
    os.makedirs(newdatname, exist_ok=True)

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))],ignore_index=True)
        findfs[r].to_csv(newdatname + 'df_' + 'K'+str(K)+'R'+ str(r+r_start) +'.csv')

    print("DONE")