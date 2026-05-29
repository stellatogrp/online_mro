import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import gamma
import itertools
import time
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import pandas as pd

def plot_eval_all(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,ylim = [0.008,0.022], legend = True,val2 = 3):
    j1,j4,j3 = j
    # Set up LaTeX rendering
    df = df[K].copy()
    df1 = df1[0].copy()
    fontsize= 10
    # df = df[K]
    # df = quantiles[K][50].copy()
    # df1 = quantiles[0][50].copy()
    quantiles = quantiles[K].copy()
    quantiles1 = quantiles1[0].copy()
    q1,q2 = q
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"],
    #     "font.size": 10,
    #     "axes.labelsize": 10,
    #     "axes.titlesize": 11,
    #     "legend.fontsize": 11
    # })
    t_range = np.array(df['t'])[(0*end_ind):(1)*end_ind:2]
    fig, (ax2,ax3,ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)
    
    # # online and reclustering
    # ax1.plot(t_range, df['O_eval0'][(j1*end_ind)+1:(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)

    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval0'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['O_eval0'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    # ax1.plot(t_range, df['MRO_eval0'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval0'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_eval0'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # DRO and SAA
    # ax1.plot(t_range, df1['SA_eval1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=1.5)
    # ax1.plot(t_range, df1['DRO_eval1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=1.5)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='g')
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='black')

    # ax1.set_ylim(ylim)
    # # plt.legend()
    # ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_title(r'Out-of-sample expected value')
    # ax1.grid(True, alpha=0.3)

    ax1.plot(t_range, df['online_time'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)
    
    # ax1.plot(t_range, np.array(df['MRO_time'][(j2*end_ind):(j2+1)*end_ind:2])+np.array(df['MRO_worst_times'][(j1*end_ind):(j1+1)*end_ind:2]), 'r-', linewidth=1, label = "reclustering",marker="D",ms=1.5)

    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')

    # reclustering worst
    ax1.plot(t_range, df['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2], 'r', linewidth=1, label = r"reclustering", marker="D",ms=1.5)
    
    # ax1.plot(t_range, quantiles[50]['MRO_time'][(j4*end_ind)+0:(j4+1)*end_ind:2], 'r:', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=1.5)
    
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')
    
    # DRO and SAA
    ax1.plot(t_range, df1['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=1.5)
    ax1.plot(t_range, df1['SA_time'][(j3*end_ind):(j3+1)*end_ind:2], color ='g', linewidth=1, label = "SAA",marker="o",ms=1.5)
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'g')

    ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_xscale("log")
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim([1e-4,1e3])
    ax1.set_yscale("log")


    # online and reclustering
    lines1, = ax2.plot(t_range, df['obj_values'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')

    lines2, = ax2.plot(t_range, np.array(df['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]), 'r', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # reclustering worst
    # ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')
    # lines5, = ax2.plot(t_range, df['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=1.5)

    # DRO and SAA
    lines3, = ax2.plot(t_range, df1['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=1.5)
    lines4, = ax2.plot(t_range, df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=1.5)
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'g')

    ax2.set_xlabel(r'Time step $(t)$')
    ax2.set_xscale("log")
    ax2.set_title(r'In-sample objective value')
    # ax2.set_ylim(ylim)
    ax2.grid(True, alpha=0.3)

    # online and reclustering
    ax3.plot(t_range, df['O_satisfy0'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)

    ax3.plot(t_range, df['MRO_satisfy0'][(j4*end_ind):(j4+1)*end_ind:2], 'r',linestyle='-', linewidth=1, label = "reclustering",marker="D",ms=1.5)
    # reclustering worst
    # ax3.plot(t_range, df['MRO_worst_satisfy1'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$",marker="^",ms=1.5)
    # DRO and SAA
    ax3.plot(t_range, df1['SA_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA",marker="o",ms=1.5)
    ax3.plot(t_range, df1['DRO_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=1.5)
    ax3.set_xlabel(r'Time step $(t)$')
    # ax3.set_xscale("log")

    ax3.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax3.grid(True, alpha=0.3)
    
    # Create a shared legend beneath the plots
    # lines = [lines1,lines2, lines3, lines4]
    lines = [lines1,lines2]
    labels = [line.get_label() for line in lines]
    if legend:
        legend = fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.05)  # Adjust the bottom margin to fit the legend
    plt.savefig(folderout + 'obj_analysis'+str(K)+'.pdf', bbox_inches='tight', dpi=300)


def plot_certificates(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,ylim = [0.008,0.022], legend = True,val2 = 3):
    j1,j4,j3 = j
    # Set up LaTeX rendering
    df = df[K].copy()
    fontsize= 10
    # df = df[K]
    # df = quantiles[K][50].copy()
    # df1 = quantiles[0][50].copy()
    quantiles = quantiles[K].copy()
    q1,q2 = q
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"],
    #     "font.size": 10,
    #     "axes.labelsize": 10,
    #     "axes.titlesize": 11,
    #     "legend.fontsize": 11
    # })
    t_range = np.array(df['t'])[(0*end_ind):(1)*end_ind:2]
    fig, (ax2,ax3,ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)
    
    # # online and reclustering
    # ax1.plot(t_range, df['O_eval0'][(j1*end_ind)+0:(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)

    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval0'][(j1*end_ind)+0:(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['O_eval0'][(j1*end_ind)+0:(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    # ax1.plot(t_range, df['MRO_eval0'][(j2*end_ind)+0:(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval0'][(j2*end_ind)+0:(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_eval0'][(j2*end_ind)+0:(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # # DRO and SAA
    # ax1.plot(t_range, df1['SA_eval1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=1.5)
    # ax1.plot(t_range, df1['DRO_eval1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=1.5)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='g')
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='black')

    # ax1.set_ylim(ylim)
    # # plt.legend()
    # ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_title(r'Out-of-sample expected value')
    # ax1.grid(True, alpha=0.3)

    ax1.plot(t_range, df['online_time'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)
    
    # ax1.plot(t_range, np.array(df['MRO_time'][(j2*end_ind):(j2+1)*end_ind:2])+np.array(df['MRO_worst_times'][(j1*end_ind):(j1+1)*end_ind:2]), 'r-', linewidth=1, label = "reclustering",marker="D",ms=1.5)

    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')

    # reclustering worst
    ax1.plot(t_range, df['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2], 'r', linewidth=1, label = r"reclustering", marker="D",ms=1.5)
    
    # ax1.plot(t_range, quantiles[50]['MRO_time'][(j4*end_ind)+0:(j4+1)*end_ind:2], 'r:', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=1.5)
    
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')
    
    # DRO and SAA
    ax1.plot(t_range, df1['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=1.5)
    ax1.plot(t_range, df1['SA_time'][(j3*end_ind):(j3+1)*end_ind:2], color ='g', linewidth=1, label = "SAA",marker="o",ms=1.5)
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'g')

    ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_xscale("log")
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim([1e-4,1e3])
    ax1.set_yscale("log")


    # online and reclustering
    lines1, = ax2.plot(t_range, np.array(df['obj_values'][(j1*end_ind):(j1+1)*end_ind:2])+5*np.array(df['sig_val'][(j1*end_ind):(j1+1)*end_ind:2]), 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)
    
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q1]['sig_val'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q2]['sig_val'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')

    lines2, = ax2.plot(t_range, np.array(df['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2])+5*np.array(df['MRO_sig_val'][(j4*end_ind):(j4+1)*end_ind:2]), 'r', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q1]['MRO_sig_val'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q2]['MRO_sig_val'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # reclustering worst
    # ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')
    # lines5, = ax2.plot(t_range, df['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=1.5)

    # DRO and SAA
    lines3, = ax2.plot(t_range, df1['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=1.5)
    lines4, = ax2.plot(t_range, df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=1.5)
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'g')

    ax2.set_xlabel(r'Time step $(t)$')
    # ax2.set_xscale("log")
    ax2.set_title(r'Certificate')
    ax2.set_ylim(ylim)
    ax2.grid(True, alpha=0.3)

    # online and reclustering
    ax3.plot(t_range, df['O_worst_satisfy1'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)

    ax3.plot(t_range, df['MRO_worst_satisfy1'][(j4*end_ind):(j4+1)*end_ind:2], 'r',linestyle='-', linewidth=1, label = "reclustering",marker="D",ms=1.5)
    # reclustering worst
    # ax3.plot(t_range, df['MRO_worst_satisfy1'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$",marker="^",ms=1.5)
    # DRO and SAA
    ax3.plot(t_range, df1['SA_satisfy2'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA",marker="o",ms=1.5)
    ax3.plot(t_range, df1['DRO_satisfy2'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=1.5)
    ax3.set_xlabel(r'Time step $(t)$')
    # ax3.set_xscale("log")

    ax3.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax3.grid(True, alpha=0.3)
    
    # Create a shared legend beneath the plots
    lines = [lines1,lines2, lines3, lines4]
    labels = [line.get_label() for line in lines]
    if legend:
        legend = fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.05)  # Adjust the bottom margin to fit the legend
    plt.savefig(folderout + 'obj_analysis'+str(K)+'.pdf', bbox_inches='tight', dpi=300)

def plot_eval(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,legend = True):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
    df1 = df1[0]
    fontsize= 10
    quantiles = quantiles[K]
    quantiles1 = quantiles1[0]
    q1,q2 = q
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": 11,
        "legend.fontsize": fontsize
    })
    t_range = np.array(df['t'])[(0*end_ind):(1)*end_ind:2] +1
    plt.figure(figsize=(4.3, 2.1), dpi=300)
    plt.plot(t_range, df['O_eval1'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering" , marker="v",ms=1.5)

    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['O_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    plt.plot(t_range, df['MRO_eval1'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval1'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_eval1'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    plt.plot(t_range, df1['DRO_eval2'][(j3*end_ind):(j3+1)*end_ind:2], 'black', linewidth=1, label = "DRO", marker="s",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_eval2'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_eval2'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='black')

    plt.plot(t_range, df1['SA_eval2'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_eval2'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_eval2'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='g')
    plt.xscale("log")
    plt.ylim([0.008,0.04])
    if legend:
        plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.title(f'Out-of-sample expected value, $K$ = {K}')
    plt.grid(True, alpha=alpha)
    plt.savefig(folderout+f'eval_analysis{K}.pdf', bbox_inches='tight', dpi=300)

def plot_satisfy(df, df1=None,end_ind=61,j=(0,0,0),K=5):
    # Set up LaTeX rendering
    j1,j2,j3 = j
    df = df[K]
    # df = quantiles[K][50].copy()
    # df1 = quantiles[0][50].copy()
    quantiles = quantiles[K].copy()
    fontsize= 10
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "legend.fontsize": 8
    })
    t_range = df['t'][(0*end_ind)+1:(1)*end_ind:2]
    plt.figure(figsize=(4, 2), dpi=300)

    # online and reclustering
    plt.plot(t_range, df['O_satisfy1'][(j1*end_ind)+1:(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)
    plt.plot(t_range, df['MRO_satisfy1'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering",marker="D",ms=1.5)
    
    #reclustering worst
    plt.plot(t_range, df['MRO_worst_satisfy1'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r:', linewidth=1, label = r"reclustering $\hat{H}^K_t$",marker="^",ms=1.5)

    # DRO and SAA
    plt.plot(t_range, df1['SA_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA",marker="o",ms=1.5)
    plt.plot(t_range, df1['DRO_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=1.5)
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Confidence $(1-\hat{\beta}_t)$')
    plt.grid(True, alpha=0.3)
    plt.savefig(folderout + 'prob_analysis.pdf', bbox_inches='tight', dpi=300)

def plot_regret(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,ylim=[0.008,0.022]):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
    # df = quantiles[K][50].copy()
    # df1 = quantiles[0][50].copy()
    quantiles = quantiles[K].copy()
    fontsize= 10
    q1,q2 = q
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": 11,
        "legend.fontsize": 7.1
    })
    t_range = np.array(df['t'])[(0*end_ind)+1:(1)*end_ind:2]
    plt.figure(figsize=(4.3, 2.1), dpi=300)

    # online and reclustering regret

    plt.plot(t_range, 5*df['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]+ np.array([5*np.sum(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'b:', linewidth=1, label = "online clustering UB", marker="o",ms=1.5)
    
    plt.plot(t_range, 5*np.array(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2]), 'b--', label = r"online clustering $\Phi^K_t$",linewidth = 0.5)

    plt.plot(t_range, np.array([np.sum((np.array(df['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)



    plt.plot(t_range, 5*df['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2]+ np.array([5*np.sum(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]) , 'r:', linewidth=1, label = "reclustering UB", marker="D",ms=1.5)

    plt.plot(t_range, 5*np.array(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2]), 'r--',label = r"reclustering $\Phi^K_t$" , linewidth = 0.5)

    plt.plot(t_range, np.array([np.sum((np.array(df['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'r-', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    

    plt.fill_between(np.array(t_range),y1=[np.sum((np.array(quantiles[q1]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],y2=[np.sum((np.array(quantiles[q2]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],alpha=alpha, color = 'b')

    

    plt.fill_between(np.array(t_range),y1=[np.sum((np.array(quantiles[q1]['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],y2=[np.sum((np.array(quantiles[q2]['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],alpha=alpha, color = 'r')


    # theoretical bounds
   
    plt.fill_between(np.array(t_range),y1=np.array(5*quantiles[q1]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+ np.array([5*np.sum(quantiles[q1]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in  range(1,int((end_ind)/2)+1)]) ,y2=np.array(5*quantiles[q2]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+np.array([5*np.sum(quantiles[q2]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]) ,alpha=alpha, color = 'b')

    
    plt.fill_between(np.array(t_range),y1=np.array(5*quantiles[q1]['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2])+ np.array([5*np.sum(quantiles[q1]['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in  range(1,int((end_ind)/2)+1)]) ,y2=np.array(5*quantiles[q2]['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2])+np.array([5*np.sum(quantiles[q2]['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]) ,alpha=alpha, color = 'r')


    plt.legend(ncol = 2)
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Dynamic regret, $\varepsilon_t = 0.0025(t+5)^{-1/40}$')
    plt.ylim(ylim)
    plt.yscale('log')
    plt.grid(True, alpha=alpha)
    plt.savefig(folderout + 'regret_analysis.pdf', bbox_inches='tight', dpi=300)

def plot_bounds(df, quantiles, df1=None, quantiles1=None, end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
    # df = quantiles[K][50].copy()
    # df1 = quantiles[0][50].copy()
    quantiles = quantiles[K].copy()
    fontsize= 10
    q1,q2 = q
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": 11,
        "legend.fontsize": 8
    })
    t_range = np.array(df['t'])[(0*end_ind):(1)*end_ind:2]

    plt.figure(figsize=(4.3, 2), dpi=300)

    # DRO upper
    plt.plot(t_range, np.array(df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2])+ 5*np.array(df['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]), color ='purple', linewidth=1, label = r"DRO $H_t + \underline{\psi}^K_t$", marker="s",ms=1.5)

    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2])+5*np.array(quantiles[q1]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]) ,y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2])+5*np.array(quantiles[q2]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]),alpha=alpha, color = 'purple')

    # reclustering upper
    plt.plot(t_range, df['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2] + 5*df['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2], 'r', linewidth=1, label = r"reclustering $H^K_t + \underline{\psi}^K_t$", marker="D",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float)+ 5*np.array(quantiles[q1]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q2]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')


    # DRO
    plt.plot(t_range, df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = r"DRO $H_t$", marker="s",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    
    # reclustering 
    plt.plot(t_range, df['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2], 'orange', linewidth=1, label = r"reclustering $H^K_t$", marker="v",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='orange')

    # # reclustering worst
    # plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # plt.plot(t_range, df['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=1.5)
    

    
    plt.legend( ncol = 2)
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Certificates, $\varepsilon_t = 0.0025(t+5)^{-1/40}$')
    # plt.ylim([0.003,0.033])
    plt.grid(True, alpha=alpha)
    plt.savefig(folderout + 'bounds_analysis.pdf', bbox_inches='tight', dpi=300)


T=2001
R = 10
K_list = [0,15,25,50]
eps_init = [0.004,0.003,0.002,0.001,0.0005,0.0001,0.00001]
eps_dro =  [0.0035,0.00325,0.003,0.0025]
M = len(eps_init)
quant_list = [25,75]

# foldername = '/Users/irina.wang/Desktop/Princeton/Project2/mro_mpc/port_new/results_new_p2/2/T'+str(T-1)+'R'+str(R)+'/'

foldername = '/scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/port_new/results/orig/p1/1/T'+str(T-1)+'R'+str(R)+'/'

# folderout = '/Users/irina.wang/Desktop/Princeton/Project2/mro_mpc/port_new/results_new_p2_plots/2/T'+str(T-1)+'R'+str(R)+'/'

folderout = '/scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/port_new/plots_new/orig/1/T'+str(T-1)+'R'+str(R)+'/'

os.makedirs(folderout, exist_ok=True)

# folderout2 = '/Users/irina.wang/Desktop/Princeton/Project2/mro_mpc/port_new/results_new_p1_plots/2/T'+str(T-1)+'R'+str(R)+'/'
# folderout2 = '/scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/port_new/results_new_p1_plots/2/T'+str(T-1)+'R'+str(R)+'/'

# os.makedirs(folderout2, exist_ok=True)

# setup MRO dfs
def setup_dfs(folderout = folderout, K_list = K_list, init = False):
    if init:
        quantiles = {}
        for K in K_list:
            dfs_list = []
            for r in range(R):
                csv_path = foldername + 'df_' + 'K'+str(K)+'R'+ str(r) +'.csv'
                if not os.path.exists(csv_path):
                    continue
                newdf = pd.read_csv(csv_path)
                dfs_list.append(newdf)
            if not dfs_list:
                print(f"Skipping K={K}: no input CSV files found in {foldername}")
                continue
            df1 = dfs_list[0]
            quantiles[K] = {}
            # Only aggregate numeric columns; non-numeric ones (serialized
            # arrays like x / MRO_x / weights_q) are copied through unchanged.
            numeric_cols = df1.select_dtypes(include=[np.number]).columns
            # Replications have ragged lengths (incomplete runs). pd.concat
            # aligns by row index and pads with NaN; quantile/mean(axis=1)
            # then skip NaN, so each row is aggregated over whichever
            # replications reached that row. Output length = max replication.
            max_len = max(len(d) for d in dfs_list)
            out_index = pd.RangeIndex(max_len)
            combined_by_col = {
                col: pd.concat([d[col] for d in dfs_list], axis=1)
                for col in numeric_cols
            }
            for quant in quant_list:
                quantiles[K][quant] = pd.DataFrame(index=out_index, columns=df1.columns)
                for col in df1.columns:
                    if col in numeric_cols:
                        quantiles[K][quant][col] = combined_by_col[col].quantile(quant/100.0, axis=1)
                    else:
                        quantiles[K][quant][col] = df1[col].reindex(out_index).values
                quantiles[K][quant].to_csv(folderout+'quantiles_'+ str(quant)+'K'+str(K)+'.csv')
            sum_df = pd.DataFrame(index=out_index, columns=df1.columns)
            for col in df1.columns:
                if col in numeric_cols:
                    sum_df[col] = combined_by_col[col].mean(axis=1)
                else:
                    sum_df[col] = df1[col].reindex(out_index).values
            sum_df.to_csv(folderout+'df_'+ 'K'+str(K)+'.csv')
    df = {}
    quantiles = {}
    for K in K_list:
        df_path = folderout+'df_' + 'K'+str(K)+'.csv'
        if not os.path.exists(df_path):
            print(f"Skipping K={K}: missing aggregated CSV {df_path}")
            continue
        df[K] = pd.read_csv(df_path)
        quantiles[K] = {}
        for quant in quant_list:
            q_path = folderout+'quantiles_'+ str(quant)+'K'+str(K)+'.csv'
            if not os.path.exists(q_path):
                print(f"Missing quantile CSV for K={K}, quant={quant}: {q_path}")
                continue
            quantiles[K][quant] = pd.read_csv(q_path)
    return df, quantiles


def infer_end_ind(df_dict, K=None, t_col='t'):
    if not df_dict:
        raise ValueError("No dataframes available to infer end_ind")

    if K is None or K not in df_dict:
        K = next(iter(df_dict))

    dfk = df_dict[K]
    if t_col not in dfk.columns or dfk.empty:
        return len(dfk)

    t = pd.to_numeric(dfk[t_col], errors='coerce').to_numpy()
    first_t = t[0]
    repeated = np.where(t[1:] == first_t)[0]
    if repeated.size > 0:
        return int(repeated[0] + 1)

    return int(len(dfk))

df, quantiles = setup_dfs(folderout = folderout, K_list = [0,15,25], init = False)
# df1,quantiles1 = setup_dfs(folderout = folderout,init = False)

end_ind = infer_end_ind(df, K=25)

plot_eval_all(df,quantiles,df,quantiles,j=(0,0,0),K=15,q=(25,75),ylim=[0.004,0.02],legend = True,val2=2.3, end_ind=end_ind)

plot_eval(df,quantiles,df,quantiles,j=(0,0,0),K=15,q=(25,75),end_ind=end_ind,legend = True)