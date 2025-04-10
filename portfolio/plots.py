import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import os

output_stream = sys.stdout

def plot_certificates(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,ylim = [0.008,0.022], legend = True,val2 = 3):
    j1,j4,j3 = j
    # Set up LaTeX rendering
    df = df[K].copy()
    fontsize= 10
    quantiles = quantiles[K].copy()
    q1,q2 = q
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 11
    })
    t_range = np.array(df['t'])[(0*end_ind):(1)*end_ind:2]
    fig, (ax2,ax3,ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)

    ax1.plot(t_range, df['online_time'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)
   
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    
    # reclustering worst
    ax1.plot(t_range, df['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2], 'r', linewidth=1, label = r"reclustering", marker="D",ms=1.5)
    
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
    ax3.plot(t_range, df['O_worst_satisfy0'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)

    ax3.plot(t_range, df['MRO_worst_satisfy0'][(j4*end_ind):(j4+1)*end_ind:2], 'r',linestyle='-', linewidth=1, label = "reclustering",marker="D",ms=1.5)

    # DRO and SAA
    ax3.plot(t_range, df1['SA_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA",marker="o",ms=1.5)
    ax3.plot(t_range, df1['DRO_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=1.5)
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
    plt.savefig('obj_analysis'+str(K)+'.pdf', bbox_inches='tight', dpi=300)



def plot_regret(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
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
    plt.plot(t_range, 5*df['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]+ np.array([5*np.sum(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)]), 'b:', linewidth=1, label = "online clustering UB", marker="o",ms=1.5)
    
    plt.plot(t_range, 5*np.array(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2]), 'b--', label = r"online clustering $\Phi^K_t$",linewidth = 0.5)

    plt.plot(t_range, np.array([np.sum((np.array(df['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)]), 'b-', linewidth=1, label = "online clustering", marker="v",ms=1.5)

    plt.plot(t_range, 5*df['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2]+ np.array([5*np.sum(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)]) , 'r:', linewidth=1, label = "reclustering UB", marker="D",ms=1.5)

    plt.plot(t_range, 5*np.array(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2]), 'r--',label = r"reclustering $\Phi^K_t$" , linewidth = 0.5)

    plt.plot(t_range, np.array([np.sum((np.array(df['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)]), 'r-', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    

    plt.fill_between(np.array(t_range),y1=[np.sum((np.array(quantiles[q1]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)],y2=[np.sum((np.array(quantiles[q2]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)],alpha=alpha, color = 'b')

    

    plt.fill_between(np.array(t_range),y1=[np.sum((np.array(quantiles[q1]['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)],y2=[np.sum((np.array(quantiles[q2]['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)],alpha=alpha, color = 'r')


    plt.fill_between(np.array(t_range),y1=np.array(5*quantiles[q1]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+ np.array([5*np.sum(quantiles[q1]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in  range(1,int((end_ind-1)/2)+1)]) ,y2=np.array(5*quantiles[q2]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+np.array([5*np.sum(quantiles[q2]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)]) ,alpha=alpha, color = 'b')

    
    plt.fill_between(np.array(t_range),y1=np.array(5*quantiles[q1]['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2])+ np.array([5*np.sum(quantiles[q1]['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in  range(1,int((end_ind-1)/2)+1)]) ,y2=np.array(5*quantiles[q2]['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2])+np.array([5*np.sum(quantiles[q2]['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind-1)/2)+1)]) ,alpha=alpha, color = 'r')


    plt.legend(ncol = 2)
    plt.xlabel(r'Time $(T)$')
    plt.title(r'Dynamic regret, $\varepsilon_t = 0.0025(t+5)^{-1/40}$')
    # plt.ylim([0.008,0.022])
    plt.yscale('log')
    plt.grid(True, alpha=alpha)
    plt.savefig('regret_analysis.pdf', bbox_inches='tight', dpi=300)

def plot_bounds(df, quantiles, df1=None, quantiles1=None, end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
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
    
    plt.legend( ncol = 2)
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Certificates, $\varepsilon_t = 0.0025(t+5)^{-1/40}$')
    # plt.ylim([0.003,0.033])
    plt.grid(True, alpha=alpha)
    plt.savefig('bounds_analysis.pdf', bbox_inches='tight', dpi=300)


def plot_dists(df, quantiles, end_ind=61, j = (0,0), q = (25,75),K=5, alpha=0.1):
    j1,j2 = j
    df = df[K]
    fontsize= 11
    quantiles = quantiles[K]
    q1,q2 = q
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 9
        
    })
    t_range = np.array(df['t'])[(0*end_ind)+1:(1)*end_ind:2]
    plt.figure(figsize=(3, 2), dpi=300)
    plt.plot(t_range, df['mean_val'][(j1*end_ind)+1:(j1+1)*end_ind:2], 'b:', linewidth=1, label = r"$d^K_{t,1}$")
    plt.plot(t_range, df['MRO_mean_val'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r:', linewidth=1)

    plt.plot(t_range, 5*np.array(df['sig_val'][(j1*end_ind)+1:(j1+1)*end_ind:2]), 'b-.', linewidth=1, label = r"$\Phi^K_t$")
    plt.plot(t_range, 5*np.array(df['MRO_sig_val'][(j2*end_ind)+1:(j2+1)*end_ind:2]), 'r-.', linewidth=1)

    plt.plot(t_range, df['square_val'][(j1*end_ind)+1:(j1+1)*end_ind:2], 'b-', linewidth=1, label = r"$(D^K_{t,2})^2$")
    plt.plot(t_range, df['MRO_square_val'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r', linewidth=1)

    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['mean_val'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['mean_val'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_mean_val'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_mean_val'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['square_val'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['square_val'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_square_val'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_square_val'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')

    plt.fill_between(np.array(t_range),y1=5*np.array(quantiles[q1]['sig_val'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),y2=5*np.array(quantiles[q2]['sig_val'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    plt.fill_between(np.array(t_range),y1=5*np.array(quantiles[q1]['MRO_sig_val'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),y2=5*np.array(quantiles[q2]['MRO_sig_val'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')

    plt.legend(loc = 'center right',  bbox_to_anchor=(1, 0.7))
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Clustering distances')
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.savefig('dist_conv.pdf', bbox_inches='tight', dpi=300)


def plot_eval(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,legend = True):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
    fontsize= 10
    quantiles = quantiles[K]
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
    plt.plot(t_range, df['O_eval0'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering" , marker="v",ms=1.5)

    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval0'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['O_eval0'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    plt.plot(t_range, df['MRO_eval0'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering", marker="D",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval0'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_eval0'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    plt.plot(t_range, df1['DRO_eval1'][(j3*end_ind):(j3+1)*end_ind:2], 'black', linewidth=1, label = "DRO", marker="s",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_eval1'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_eval1'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='black')

    plt.plot(t_range, df1['SA_eval1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=1.5)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_eval1'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_eval1'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='g')
    plt.xscale("log")
    if legend:
        plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.title(f'Out-of-sample expected value, $K$ = {K}')
    plt.grid(True, alpha=alpha)
    plt.savefig('eval_analysis'+str(K)+'.pdf', bbox_inches='tight', dpi=300)


def plot_eval_T(df, quantiles, end_ind=61,j=(0,0), q = (40,60),K=5, alpha=0.1):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
    fontsize= 10
    quantiles = quantiles[K]
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
    t_range = np.array(df['t'])[(0*end_ind):(1)*end_ind:1]
    plt.figure(figsize=(4.3, 2.1), dpi=300)
    plt.plot(t_range, df['O_eval0'][(j1*end_ind):(j1+1)*end_ind:1], 'b-', linewidth=1, label = "online clustering")

    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval0'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),y2=np.array(quantiles[q2]['O_eval0'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),alpha=alpha, color = 'b')
    plt.plot(t_range, df['MRO_eval0'][(j2*end_ind):(j2+1)*end_ind:1], 'r-', linewidth=1, label = "reclustering")
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval0'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),y2=np.array(quantiles[q2]['MRO_eval0'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),alpha=alpha, color='r')

    plt.legend()
    plt.xscale("log")
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Out-of-sample expected value')
    plt.grid(True, alpha=alpha)
    plt.savefig('eval_analysis_T.pdf', bbox_inches='tight',format='pdf',dpi = 300)


def plot_dists(df, quantiles, end_ind=61, j = (0,0), q = (40,60),K=5, alpha=0.1,ylim = None):
    j1,j2 = j
    df = df[K]
    fontsize= 11
    quantiles = quantiles[K]
    q1,q2 = q
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9
        
    })
    t_range = np.array(df['t'])[(0*end_ind):(1)*end_ind:1]
    plt.figure(figsize=(4.3, 2.1), dpi=300)
    plt.plot(t_range, df['mean_val'][(j1*end_ind):(j1+1)*end_ind:1], 'b:', linewidth=1, label = r"$d^K_{t,1}$")
    plt.plot(t_range, df['MRO_mean_val'][(j2*end_ind):(j2+1)*end_ind:1], 'r:', linewidth=1)

    plt.plot(t_range, 5*np.array(df['sig_val'][(j1*end_ind):(j1+1)*end_ind:1]), 'b-.', linewidth=1, label = r"$\Phi^K_t$")
    plt.plot(t_range, 5*np.array(df['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:1]), 'r-.', linewidth=1)

    plt.plot(t_range, df['square_val'][(j1*end_ind):(j1+1)*end_ind:1], 'b-', linewidth=0.5, label = r"$(D^K_{t,2})^2$")
    plt.plot(t_range, df['MRO_square_val'][(j2*end_ind):(j2+1)*end_ind:1], 'r', linewidth=0.5)


    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['mean_val'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),y2=np.array(quantiles[q2]['mean_val'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),alpha=alpha, color = 'b')
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_mean_val'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),y2=np.array(quantiles[q2]['MRO_mean_val'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),alpha=alpha, color = 'r')
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['square_val'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),y2=np.array(quantiles[q2]['square_val'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),alpha=alpha, color = 'b')
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_square_val'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),y2=np.array(quantiles[q2]['MRO_square_val'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),alpha=alpha, color = 'r')

    plt.fill_between(np.array(t_range),y1=5*np.array(quantiles[q1]['sig_val'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),y2=5*np.array(quantiles[q2]['sig_val'][(j1*end_ind):(j1+1)*end_ind:1]).astype(float),alpha=alpha, color = 'b')
    plt.fill_between(np.array(t_range),y1=5*np.array(quantiles[q1]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),y2=5*np.array(quantiles[q2]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:1]).astype(float),alpha=alpha, color = 'r')

    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Clustering distances')
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    # plt.yscale("log")
    # plt.xlim([1e0,1e4])
    plt.ylim(ylim)
    plt.savefig('dist_conv.pdf', bbox_inches='tight', format='pdf',dpi = 300)


# setup MRO dfs
def setup_dfs(init, R, K_list, foldername, folderout):
    if init:
        quantiles = {}
        for K in K_list:
            dfs_list = []
            for r in range(R):
                newdf = pd.read_csv(foldername + 'df_' + 'K'+str(K)+'R'+ str(r) +'.csv')
                dfs_list.append(newdf)
            df1 = dfs_list[0]
            quantiles[K] = {}
            for quant in quant_list:
                quantiles[K][quant] = pd.DataFrame(index=df1.index, columns=df1.columns)
                # Calculate quantiles for each entry
                for i in range(len(df1.index)):
                    for j in range(len(df1.columns)):
                        values = [dfs_list[k].iloc[i, j] for k in range(len(dfs_list))]
                        quantiles[K][quant].iloc[i, j] = np.percentile(values, quant)
                quantiles[K][quant].to_csv(folderout+'quantiles_'+ str(quant)+'K'+str(K)+'.csv')
            sum_df = dfs_list[0].copy()
            for dfs in dfs_list[1:]:
                sum_df = sum_df.add(dfs, fill_value=0)
            sum_df = sum_df/(R)
            sum_df.to_csv(folderout+'df_'+ 'K'+str(K)+'.csv')
    df = {}
    quantiles = {}
    for K in K_list:
        df[K] = pd.read_csv(folderout+'df_' + 'K'+str(K)+'.csv')
        quantiles[K] = {}
        for quant in quant_list:
            quantiles[K][quant] = pd.read_csv(folderout+'quantiles_'+ str(quant)+'K'+str(K)+'.csv')
    return df, quantiles
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="portfolio_exp/", metavar='N')
    parser.add_argument('--R',type=int, default = 10)

    arguments = parser.parse_args()
    foldername = arguments.foldername
    R = arguments.R

        
    T=2001
    K_list = [0,15,25]
    quant_list = [25,75]
    foldername = foldername+str(T-1)+'/'
    folderout = foldername+str(T-1)+'/finalized_dfs/'
    os.makedirs(folderout, exist_ok=True)

    df, quantiles = setup_dfs(init = True,R = R, K_list = K_list, foldername = foldername, folderout= folderout)

    plot_certificates(df,quantiles,df[0],quantiles[0],j=(4,4,3),K=15,q=(25,75),ylim=[0.004,0.03],legend = True,val2=2.3,end_ind = 58)

    plot_eval(df,quantiles,df[0],quantiles[0],j=(4,4,3),K=15,q=(25,75),end_ind=58,legend = False)

    plot_eval(df,quantiles,df[0],quantiles[0],j=(4,4,3),K=25,q=(25,75),end_ind=58)

    plot_regret(df,quantiles,df[0],quantiles[0],j=(4,4,3),K=25,q=(25,75),end_ind = 58,ylim=[0.0005,1])

    plot_bounds(df,quantiles,df[0],quantiles[0],j=(4,4,3),K=25,q=(25,75),end_ind = 58)

    T=10001
    K_list = [25]
    foldername = foldername+str(T-1)+'/'
    folderout = foldername+str(T-1)+'/finalized_dfs/'
    os.makedirs(folderout, exist_ok=True)
    df, quantiles = setup_dfs(init = True,R = R, K_list = K_list, foldername = foldername, folderout= folderout)

    plot_eval_T(df,quantiles,j=(2,2),K=25,q=(25,75),end_ind = 30)

    plot_dists(df,quantiles,q = (25,75),j = (2,2), end_ind=30,K=25,ylim = None)

    print("DONE")
    
