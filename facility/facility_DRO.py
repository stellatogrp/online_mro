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
import ot
import itertools
import gurobipy

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

def prob_facility(K, m, n,p,c,C):
    """Create the problem in cvxpy
    Parameters
    ----------
    K: int
        Number of data samples
    m: int
        Number of customers
    n: int
        Number of facilities
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    eps = cp.Parameter()
    d_train = cp.Parameter((K, m))
    wk = cp.Parameter(K)
    x = cp.Variable(n, boolean=True)
    X = cp.Variable((n, m))
    lmbda = cp.Variable(K)
    tau_obj = cp.Variable()
    tau = cp.Variable()
    s = cp.Variable(K)
    gam = cp.Variable((n, K*2*m))
    a = 5
    C_r = np.vstack([-np.eye(m), np.eye(m)])
    d_r = np.hstack([-np.zeros(m), np.ones(m)*10])

    objective = cp.Minimize(tau_obj)

    constraints = []
    for j in range(m):
        constraints += [cp.sum(X[:, j]) == 1]

    constraints += [cp.trace(C.T @ X) + c@x + 20*wk @ s <= tau_obj]
    for k in range(K):
        for i in range(n):
            constraints += [tau + -a*tau + lmbda[k]*eps - a*p[i]*x[i] + a*d_train[k]@X[i] +
                            gam[i, (k*2*m):((k+1)*2*m)]@(d_r - C_r@d_train[k]) <= s[k]]
            constraints += [cp.norm(C_r.T@gam[i, (k*2*m):((k+1)*2*m)
                                              ] - a*X[i], 1) <= lmbda[k]]
        constraints += [lmbda[k]*eps <= s[k]]
    constraints += [s >= tau]
    constraints += [X >= 0, lmbda >= 0, gam >= 0]

    problem = cp.Problem(objective, constraints)

    return problem, x, X, d_train, wk, eps, tau


def prob_facility_nosup(K, m, n,p,c,C):
    """Create the problem in cvxpy
    Parameters
    ----------
    K: int
        Number of data samples
    m: int
        Number of customers
    n: int
        Number of facilities
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    eps = cp.Parameter()
    d_train = cp.Parameter((K, m))
    wk = cp.Parameter(K)
    x = cp.Variable(n, boolean=True)
    X = cp.Variable((n, m))
    lmbda = cp.Variable(K)
    tau_obj = cp.Variable()
    tau = cp.Variable()
    s = cp.Variable(K)
    gam = cp.Variable((n, K*2*m))
    a = 5
    C_r = np.vstack([-np.eye(m), np.eye(m)])
    d_r = np.hstack([-np.zeros(m), np.ones(m)*10])

    objective = cp.Minimize(tau_obj)

    constraints = []
    for j in range(m):
        constraints += [cp.sum(X[:, j]) == 1]

    constraints += [cp.trace(C.T @ X) + c@x + 20*wk @ s <= tau_obj]
    constraints += [tau - a*tau + cp.vstack([lmbda*eps]*n).T - a*cp.vstack([cp.multiply(p,x)]*K) + a*d_train@X.T <= cp.vstack([s]*n).T]
    for i in range(n):
        constraints += [cp.vstack([cp.norm(-a*X[i], 1)]*K) <= cp.reshape(lmbda,(K,1))]
    constraints += [lmbda*eps + tau<= s]
    constraints += [X >= 0, lmbda >= 0, s >= 0]

    problem = cp.Problem(objective, constraints)
    return problem, x, X, d_train, wk, eps, tau


def worse_case(K, m, n,d_train,p,c,C):
    """Create the problem in cvxpy
    Parameters
    ----------
    K: int
        Number of data samples
    m: int
        Number of customers
    n: int
        Number of facilities
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    eps = cp.Parameter()
    wk = cp.Parameter(K)
    x = cp.Parameter(n)
    X = cp.Parameter((n, m))
    lmbda = cp.Variable(K)
    tau_obj = cp.Variable()
    tau = cp.Parameter()
    s = cp.Variable(K)
    gam = cp.Variable((n, K*2*m))
    a = 5
    C_r = np.vstack([-np.eye(m), np.eye(m)])
    d_r = np.hstack([-np.zeros(m), np.ones(m)*10])

    objective = cp.Minimize(tau_obj)

    constraints += [cp.trace(C.T @ X) + c@x + 20*wk @ s <= tau_obj]
    constraints += [tau - a*tau + cp.vstack([lmbda*eps]*n).T - a*cp.vstack([cp.multiply(p,x)]*K) + a*d_train@X.T <= cp.vstack([s]*n).T]
    for i in range(n):
        constraints += [cp.vstack([cp.norm(-a*X[i], 1)]*K) <= cp.reshape(lmbda,(K,1))]
    constraints += [lmbda*eps + tau<= s]
    constraints += [lmbda >= 0, s >= 0]
    problem = cp.Problem(objective, constraints)

    return problem, wk, eps, x, X, tau

def generate_facility_data(n=10, m=50):
    """Generate data for one problem instance
    Parameters
    ----------
    m: int
        Number of customers
    n: int
        Number of facilities
    Returns
    -------
    c: vector
        Opening cost of each facility
    C: array
        Shipment cost between customers and facilities
    p: vector
        Production capacity of each facility
    """
    # Cost for facility
    np.random.seed(1)
    c = np.random.uniform(3, 9, n)

    # Cost for shipment
    fac_loc = np.random.uniform(0, 2.5, size=(n, 2))
    cus_loc = np.random.uniform(0, 2.5, size=(m, 2))
    #  rho = 4

    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(fac_loc[i, :] - cus_loc[j, :])

    # Capacities for each facility
    p = np.random.randint(10, 40, n)

    # Past demands of customer (past uncertain data)
    return c, C, p


def generate_facility_demands(N, m,seed):
    """Generate uncertain demand
    Parameters:
     ----------
    N: int
        Number of data samples
    m: int
        Number of facilities
    R: int
        Number of sets of data samples
    Returns:
    -------
    d_train: vector
        Demand vector
    """
    np.random.seed(seed)
    dat = np.random.normal(2, 0.3, (N, m))
    dat2 = np.random.normal(3, 0.3, (N, m))
    dat3 = np.random.normal(4, 0.5, (N, m))
    dat = np.vstack([dat, dat2, dat3])
    dat = np.minimum(dat, 10)
    dat = np.maximum(dat, 0)
    indices = np.random.choice(3*N,3*N,replace=False) 
    dat = dat[indices]
    return dat


def evaluate_k(x, X, d, tau,p,c,C):
    """Evaluate stricter constraint satisfaction
    Parameters
    ----------
    p: vector
        Prices
    x: vector
        Decision variables of the optimization problem
    X: matrix
        Decision variables of the optimization problem
    d: matrix
        Validation data matrix
    Returns:
    -------
    boolean: indicator of constraint satisfaction
    """
    retval = 1
    maxval = np.zeros((np.shape(d)[0], np.shape(x)[0]))
    maxval2 = np.zeros((np.shape(d)[0], np.shape(x)[0]))
    for fac in range(np.shape(x)[0]):
        for ind in range(np.shape(d)[0]):
            maxval[ind, fac] = 20*np.maximum(tau + 5*np.maximum(-p[fac]*x[fac] + d[ind]@X[fac] - tau,0),0)
            maxval2[ind,fac] = -p[fac]*x[fac] + d[ind]@X[fac]
    if np.mean(np.max(maxval2, axis=1)) >= 0.00001:
        retval = 0
    return np.trace(C.T @ X) + c@x + np.mean(np.max(maxval, axis=1)), retval


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
    K = k2['K']
    val = 0
    for k in range(K):
        val += np.abs(k1["w"][k] - k2["w"][k])*np.linalg.norm(k1["d"][k] - k2["d"][k])
    if k1['K']>K:
        dists = cdist(k1['d'][K].reshape((1,m)),k2['d'][:K])
        val += dists@np.abs(k2['w'][:K] - k1['w'][:K])
    return val

def create_scenario(dat,m,num_dat):
    x = cp.Variable(n, boolean=True)
    X = cp.Variable((n, m))
    tau_obj = cp.Variable()
    tau = cp.Variable()
    objective = cp.Minimize(tau_obj)

    constraints = []
    for j in range(m):
        constraints += [cp.sum(X[:, j]) == 1]

    constraints += [cp.trace(C.T @ X) + c@x + 20*(cp.sum(cp.maximum(tau + 5*cp.maximum(cp.max(dat@X.T - cp.vstack([cp.multiply(x,p)]*(num_dat)) -tau,axis=1),0),0))/(num_dat)) <= tau_obj]

    constraints += [X >=0]

    problem = cp.Problem(objective, constraints)
    return problem, x, X, tau
    
def calc_rmse(dat,mean):
    rmse = 0
    for d in dat:
        rmse += np.linalg.norm(d-mean,2)**2
    return rmse

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
        rmse = np.sqrt(calc_rmse(cluster_data,np.reshape(q_dict['d'][q],(1,m))))
        if rmse <= 1e-6:
            rmse = 0.03
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
    start_time = time.time()
    cur_K = np.minimum(K,q_dict['cur_Q'])
    cur_Q = q_dict['cur_Q']
    k_dict['K'] = cur_K
    if init or (cur_Q<=K):
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
    cur_K = k_dict['K']
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


def compute_cumulative_regret(history,dateval,p,c,C):
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
    
    DRO_e = []
    DRO_s = []
    SA_e = []
    SA_s = []
    SA_satisfy_probs = []
    DRO_satisfy_probs = []

    T = len(history['t'])
    # Generate evaluation samples from true distribution for cost computation
    for j in range(4):
        SA_eval_values = np.zeros(T)
        DRO_eval_values = np.zeros(T)
        DRO_satisfy_prob = np.zeros(T)
        SA_satisfy_prob = np.zeros(T)
        eval_samples = dateval[(j*2000):(j+1)*2000,:m]
    # For each timestep t
        for t in range(T):            
            # Compute instantaneous regret at time t using true distribution
            DRO_cost,o1 = evaluate_k(history['DRO_x'][t],history['DRO_X'][t], eval_samples,history['DRO_tau'][t],p,c,C)
            SA_cost,m1 = evaluate_k(history['SA_x'][t],history['SA_X'][t], eval_samples,history['SA_tau'][t],p,c,C)
            SA_eval_values[t] = SA_cost
            DRO_eval_values[t] = DRO_cost
            SA_satisfy_prob[t] = m1
            DRO_satisfy_prob[t] = o1
            
        DRO_satisfy = np.array(history['DRO_obj_values'] >= DRO_eval_values).astype(float)
        SA_satisfy = np.array(history['SA_obj_values'] >= SA_eval_values).astype(float)

        DRO_e.append(DRO_eval_values)
        DRO_s.append(DRO_satisfy)
        SA_e.append(SA_eval_values)
        SA_s.append(SA_satisfy)
        DRO_satisfy_probs.append(DRO_satisfy_prob)
        SA_satisfy_probs.append(SA_satisfy_prob)
    
    return DRO_e, DRO_s, SA_e, SA_s, DRO_satisfy_probs, SA_satisfy_probs

def calc_cluster_val(K,k_dict, num_dat,X):
    mean_val = 0
    square_val = 0
    sig_val = 0
    cur_K = min(K,num_dat)
    for k in range(cur_K):
        centroid = k_dict['d'][k]
        for dat in k_dict['data'][k]:
            cur_val = np.linalg.norm(dat-centroid,2)
            mean_val += cur_val
            square_val += cur_val**2
            sig_val += max(np.max([(centroid - dat)@(X[i]) for i 
        in range(n)]),0)
    return mean_val/num_dat, square_val/num_dat, sig_val/num_dat

def facility_experiments(r_input,K,T,N_init,dateval,r_start,p,c,C):
    r,epsnum = list_inds[r_input]
    dat = generate_facility_demands(2000, m,seed = r_start+r)
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
        't':[],
        'DRO_obj_values_act':[],
        'SA_obj_values_act':[],
        'DRO_X':[],
        'SA_X':[]
    }

    for t in range(T):
        print(f"\nTimestep {t+1}/{T}")
        
        radius = init_eps*(1/(num_dat**(1/(10))))
        running_samples = dat[init_ind:(init_ind+num_dat)]

        if t % interval == 0 or ((t-1) % interval == 0) :
        # solve DRO problem 
            DRO_problem, DRO_x, DRO_X, DRO_data, DRO_w,DRO_eps,DRO_tau = prob_facility_nosup(num_dat, m, n,p,c,C)
            DRO_data.value = running_samples
            DRO_w.value = (1/num_dat)*np.ones(num_dat)
            DRO_eps.value = radius
            DRO_problem.solve( solver=cp.MOSEK, verbose=False, mosek_params={
                mosek.dparam.optimizer_max_time:  2000.0})
            DRO_x_current = DRO_x.value
            DRO_X_current = DRO_X.value
            DRO_tau_current = DRO_tau.value
            DRO_min_obj = DRO_problem.objective.value
            DRO_min_time = DRO_problem.solver_stats.solve_time

        if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0):
            s_prob, s_x, s_X, s_tau = create_scenario(running_samples,m,num_dat)
            s_prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={
                    mosek.dparam.optimizer_max_time:  2000.0})
            SA_x_current = s_x.value
            SA_X_current = s_X.value
            SA_tau_current = s_tau.value
            SA_obj_current = s_prob.objective.value
            SA_time = s_prob.solver_stats.solve_time

            history['DRO_computation_times']['total_iteration'].append(DRO_min_time)
            history['DRO_x'].append(DRO_x_current)
            history['DRO_X'].append(DRO_X_current)
            history['DRO_tau'].append(DRO_tau_current)
            history['DRO_obj_values'].append(DRO_min_obj)
            history['epsilon'].append(radius)
            history['t'].append(t)

            history['SA_computation_times'].append(SA_time)
            history['SA_x'].append(SA_x_current)
            history['SA_X'].append(SA_X_current)
            history['SA_tau'].append(SA_tau_current)
            history['SA_obj_values'].append(SA_obj_current)
            history['DRO_obj_values_act'].append(np.trace(C.T @ DRO_X_current) + c@DRO_x_current)
            history['SA_obj_values_act'].append(np.trace(C.T @ SA_X_current) + c@SA_x_current)


        # New sample
        new_sample = dat[init_ind+num_dat]
        # q_dict, k_dict, weight_update_time = online_cluster_update(K,new_sample, q_dict, k_dict,num_dat, t, fixed_time)
        num_dat += 1
        # history['online_computation_times']['weight_update'].append(weight_update_time)
        # history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)
    

        if t % interval_SAA == 0 or ((t-1) % interval_SAA == 0) or t<=200:

            DRO_eval, DRO_satisfy,SA_eval, SA_satisfy, DRO_satisfy_p, SA_satisfy_p = compute_cumulative_regret(
            history,dateval,p,c,C)
            
            df = pd.DataFrame({
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
            'SA_satisfy_p1': SA_satisfy_p[0],
            'SA_satisfy_p2': SA_satisfy_p[1],
            'SA_satisfy_p3': SA_satisfy_p[2],
            'SA_satisfy_p4': SA_satisfy_p[3],
            'DRO_satisfy_p1': DRO_satisfy_p[0],
            'DRO_satisfy_p2': DRO_satisfy_p[1],
            'DRO_satisfy_p3': DRO_satisfy_p[2],
            'DRO_satisfy_p4': DRO_satisfy_p[3],
            'SA_obj_values': np.array(history['SA_obj_values']),
            'SA_obj_values_act': history["SA_obj_values_act"],
            'DRO_obj_values_act': history["DRO_obj_values_act"],
            'SA_time':np.array(history['SA_computation_times']),
            'SA_tau': np.array(history['SA_tau']),
            't':np.array(history['t'])
            })
            df.to_csv(foldername+'DRO'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
            # print(f"Weights: {q_dict['w'], np.sum(q_dict['w']) }")
        

    DRO_eval, DRO_satisfy,SA_eval, SA_satisfy, DRO_satisfy_p, SA_satisfy_p = compute_cumulative_regret(
    history,dateval,p,c,C)
    
    df = pd.DataFrame({
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
    'SA_satisfy_p1': SA_satisfy_p[0],
    'SA_satisfy_p2': SA_satisfy_p[1],
    'SA_satisfy_p3': SA_satisfy_p[2],
    'SA_satisfy_p4': SA_satisfy_p[3],
    'DRO_satisfy_p1': DRO_satisfy_p[0],
    'DRO_satisfy_p2': DRO_satisfy_p[1],
    'DRO_satisfy_p3': DRO_satisfy_p[2],
    'DRO_satisfy_p4': DRO_satisfy_p[3],
    'SA_obj_values': np.array(history['SA_obj_values']),
    'SA_obj_values_act': history["SA_obj_values_act"],
    'DRO_obj_values_act': history["DRO_obj_values_act"],
    'SA_time':np.array(history['SA_computation_times']),
    'SA_tau': np.array(history['SA_tau']),
    't':np.array(history['t'])
    })
    df.to_csv(foldername+'DRO'+str(epsnum)+'_R'+str(r+r_start)+'_df.csv')
    # df.to_csv('df.csv')
    return df
    
if __name__ == '__main__':

    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--T', type=int, default=2001)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--m', type=int, default=30)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--Q', type=int, default=500)
    parser.add_argument('--fixed_time', type=int, default=1500)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--N_init', type=int, default=5)
    parser.add_argument('--r_start', type=int, default=0)
    parser.add_argument('--interval_SAA', type=int, default=100)

  

    arguments = parser.parse_args()
    foldername = arguments.foldername
    K = arguments.K
    R = arguments.R
    m = arguments.m
    n = arguments.n
    Q = arguments.Q
    T = arguments.T
    r_start = arguments.r_start
    fixed_time = arguments.fixed_time
    interval = arguments.interval
    interval_SAA = arguments.interval_SAA
    N_init = arguments.N_init
    K_arr = [5,10,15]
    K = K_arr[idx]
    foldername = foldername + 'R'+str(R)+'_T'+str(T-1)+'/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)

    c, C, p = generate_facility_data(n, m)
    dateval = generate_facility_demands(3000, m,seed = 1)
    
    init_ind = 0
    njobs = get_n_processes(100)
    #eps_init = [0.006,0.005,0.004,0.0035,0.003,0.0025,0.002,0.0015,0.001]
    # eps_init = [0.007,0.006,0.005,0.0015]
    eps_init = [0.02,0.015,0.01,0.008,0.007,0.006,0.005,0.003,0.001]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R),np.arange(M)))

    results = Parallel(n_jobs=njobs)(delayed(facility_experiments)(
        r_input,K,T,N_init,dateval,r_start,p,c,C) for r_input in range(len(list_inds)))
    
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

    newdatname = '/scratch/gpfs/iywang/mro_mpc/facility_exp/T'+str(T-1)+'R'+str(R)+'/'
    os.makedirs(newdatname, exist_ok=True)
    for r in range(R):
        # findfs[r] = findfs[r].drop(columns=['weights','MRO_weights'])
        findfs[r].to_csv(newdatname + 'df_' + 'K'+str(0)+'R'+ str(r+r_start) +'.csv')
    
    print("DONE")