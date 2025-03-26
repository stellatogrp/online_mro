import argparse
import os
import sys

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

output_stream = sys.stdout

        
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
    parser.add_argument('--interval_online', type=int, default=100)
    parser.add_argument('--N_init', type=int, default=50)

    arguments = parser.parse_args()
    foldername = arguments.foldername
    K = arguments.K
    R = arguments.R
    m = arguments.m
    Q = arguments.Q
    T = arguments.T
    fixed_time = arguments.fixed_time
    interval = arguments.interval
    interval_online = arguments.interval_online
    N_init = arguments.N_init
    K_arr = [5,8,10,15]
    K = K_arr[idx]
    foldername = foldername + 'K'+str(K)+'_R'+str(R)+'_T'+str(T-1)+'/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)
    datname = '/scratch/gpfs/iywang/mro_mpc/portfolio_time/synthetic.csv'
    synthetic_returns = pd.read_csv(datname
                                    ).to_numpy()[:, 1:]
    init_ind = 0
    eps_init = [0.007,0.006,0.005,0.004,0.003,0.002]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R),np.arange(M)))

    dfs = {}
    newdatname = '/scratch/gpfs/iywang/mro_mpc/portfolio_exp/T'+str(T-1)+'R'+str(R)+'/'
    os.makedirs(newdatname, exist_ok=True)
    for r in range(R):
        datname = foldername + 'df_' + str(r) +'.csv'
        dfs[r] = pd.read_csv(datname)
        dfs[r] = dfs[r].drop(columns=["DRO_x","MRO_x","x",'weights_q','weights'])
        dfs[r].to_csv(newdatname + 'df_' + 'K'+str(K)+'R'+ str(r) +'.csv')

    dfs_list = [dfs[i] for i in range(R)]
    sum_df = dfs[0].copy()
    for df in dfs_list[1:]:
        sum_df = sum_df.add(df, fill_value=0)
    sum_df = sum_df/R
    sum_df.to_csv(newdatname+'df_'+ 'K'+str(K)+'.csv')
    print("DONE")
    