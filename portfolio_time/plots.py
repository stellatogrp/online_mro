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

def plot_eval_all(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,ylim = [0.008,0.022], legend = True,val2 = 3,end_ind_dro=61):
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
    t_range_dro = np.array(df['t'])[(0*end_ind_dro):(1)*end_ind_dro:2]
    # # online and reclustering
    # ax1.plot(t_range, df['O_eval0'][(j1*end_ind)+1:(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)

    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval0'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['O_eval0'][(j1*end_ind)+1:(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    # ax1.plot(t_range, df['MRO_eval0'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering", marker="D",ms=0.8)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval0'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_eval0'][(j2*end_ind)+1:(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # DRO and SAA
    # ax1.plot(t_range, df1['SA_eval1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=0.8)
    # ax1.plot(t_range, df1['DRO_eval1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=0.8)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='g')
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='black')

    # ax1.set_ylim(ylim)
    # # plt.legend()
    # ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_title(r'Out-of-sample expected value')
    # ax1.grid(True, alpha=0.3)

    ax1.plot(t_range, df['online_time'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)
    
    # ax1.plot(t_range, np.array(df['MRO_time'][(j2*end_ind):(j2+1)*end_ind:2])+np.array(df['MRO_worst_times'][(j1*end_ind):(j1+1)*end_ind:2]), 'r-', linewidth=1, label = "reclustering",marker="D",ms=0.8)

    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')

    # reclustering worst
    ax1.plot(t_range, df['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2], 'r', linewidth=1, label = r"reclustering", marker="D",ms=0.8)
    
    # ax1.plot(t_range, quantiles[50]['MRO_time'][(j4*end_ind)+0:(j4+1)*end_ind:2], 'r:', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=0.8)
    
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')
    
    # DRO and SAA
    ax1.plot(t_range_dro, df1['DRO_time'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=0.8)
    ax1.plot(t_range_dro, df1['SA_time'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], color ='g', linewidth=1, label = "SAA",marker="o",ms=0.8)
    ax1.fill_between(np.array(t_range_dro),y1=np.array(quantiles1[q1]['DRO_time'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_time'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),alpha=alpha, color = 'black')
    ax1.fill_between(np.array(t_range_dro),y1=np.array(quantiles1[q1]['SA_time'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),y2=np.array(quantiles1[q2]['SA_time'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),alpha=alpha, color = 'g')
    if 'cluster_SAA_time' in df.columns:
        ax1.plot(t_range, df['cluster_SAA_time'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA",marker="x",ms=0.8)
        if 'cluster_SAA_time' in quantiles[q1].columns:
            ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['cluster_SAA_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['cluster_SAA_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'm')

    ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_xscale("log")
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim([1e-4,1e3])
    ax1.set_yscale("log")
    


    # online and reclustering
    lines1, = ax2.plot(t_range, df['obj_values'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')

    lines2, = ax2.plot(t_range, np.array(df['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]), 'r', linewidth=1, label = "reclustering", marker="D",ms=0.8)
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # reclustering worst
    # ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')
    # lines5, = ax2.plot(t_range, df['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=0.8)

    # DRO and SAA
    lines3, = ax2.plot(t_range_dro, df1['SA_obj_values'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=0.8)
    lines4, = ax2.plot(t_range_dro, df1['DRO_obj_values'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=0.8)
    if 'cluster_SAA_obj_values' in df.columns:
        lines_cluster, = ax2.plot(t_range, df['cluster_SAA_obj_values'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA", marker="x",ms=0.8)
    ax2.fill_between(np.array(t_range_dro),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),alpha=alpha, color = 'black')
    ax2.fill_between(np.array(t_range_dro),y1=np.array(quantiles1[q1]['SA_obj_values'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),y2=np.array(quantiles1[q2]['SA_obj_values'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),alpha=alpha, color = 'g')
    if 'cluster_SAA_obj_values' in quantiles[q1].columns:
        ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['cluster_SAA_obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['cluster_SAA_obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'm')

    ax2.set_xlabel(r'Time step $(t)$')
    ax2.set_xscale("log")
    ax2.set_title(r'In-sample objective value')
    # ax2.set_ylim(ylim)
    ax2.grid(True, alpha=0.3)

    # online and reclustering
    ax3.plot(t_range, df['O_satisfy0'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)

    ax3.plot(t_range, df['MRO_satisfy0'][(j4*end_ind):(j4+1)*end_ind:2], 'r',linestyle='-', linewidth=1, label = "reclustering",marker="D",ms=0.8)
    # reclustering worst
    # ax3.plot(t_range, df['MRO_worst_satisfy1'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$",marker="^",ms=0.8)
    # DRO and SAA
    ax3.plot(t_range_dro, df1['SA_satisfy1'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], 'g-', linewidth=1, label = "SAA",marker="o",ms=0.8)
    ax3.plot(t_range_dro, df1['DRO_satisfy1'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=0.8)
    if 'cluster_SAA_satisfy1' in df.columns:
        ax3.plot(t_range, df['cluster_SAA_satisfy1'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA",marker="x",ms=0.8)
    ax3.set_xlabel(r'Time step $(t)$')
    # ax3.set_xscale("log")

    ax3.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax3.grid(True, alpha=0.3)
    
    # Create a shared legend beneath the plots
    # lines = [lines1,lines2, lines3, lines4]
    lines = [lines1, lines2, lines3, lines4]
    try:
        lines.append(lines_cluster)
    except NameError:
        pass
    labels = [line.get_label() for line in lines]
    if legend:
        legend = fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.05)  # Adjust the bottom margin to fit the legend
    plt.savefig(folderout + 'obj_analysis'+str(K)+'.pdf', bbox_inches='tight', dpi=300)


def plot_eval_all_compare(
    df,  quantiles,  df1, quantiles1,
    df2, quantiles2, df3, quantiles3,
    end_ind=61,
    end_ind_grad=None,   # subgrad epsilon-block length; falls back to end_ind if None
    j=(0, 0, 0),         # full epsilon-indices: (online, reclustering, DRO/SAA)
    j_grad=(0, 0, 0),    # subgrad epsilon-indices: (online, reclustering, DRO)
    stride=2,            # subsample stride within an epsilon block (full)
    stride_grad=None,    # subgrad stride; falls back to stride if None
    q=(40, 60),
    K=5,
    alpha=0.1,
    ylim=[0.008, 0.022],
    legend=True,
    val2=3,
):
    """3-panel comparison plot overlaying two experiment sets.

    Mirrors ``plot_eval_all`` for the *full* set (df/df1, solid lines, suffix
    "full") and overlays a second *subgrad* set (df2/df3, dashed lines, suffix
    "subgrad") on the same axes.  Same colors per method, same markers.

    The two experiment sets are allowed to have different per-epsilon block
    lengths -- pass ``end_ind`` for the full set and ``end_ind_grad`` for the
    subgrad set.  Each set is sliced with its own block length and its own
    ``t``-axis taken from the corresponding dataframe.

    Method -> data source:
      online clustering : df[K]   / df2[K]
      reclustering      : df[K]   / df2[K]
      DRO               : df1[0]  / df3[0]
      SAA               : df1[0]  ONLY  (no subgrad overlay)
    """
    if end_ind_grad is None:
        end_ind_grad = end_ind
    if stride_grad is None:
        stride_grad = stride
    j1, j4, j3 = j
    j1g, j4g, j3g = j_grad
    df  = df[K].copy()
    df1 = df1[0].copy()
    df2 = df2[K].copy()
    df3 = df3[0].copy()
    quantiles  = quantiles[K].copy()
    quantiles1 = quantiles1[0].copy()
    quantiles2 = quantiles2[K].copy()
    quantiles3 = quantiles3[0].copy()
    q1, q2 = q

    t_range      = np.array(df['t'])[(0*end_ind):(1)*end_ind:stride]
    t_range_grad = np.array(df3['t'])[(0*end_ind_grad):(1)*end_ind_grad:stride_grad]
    fig, (ax2, ax3, ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)

    # ---- helpers to keep the per-method block compact ----------------------
    def _band(ax, qa, qb, col, ji, ei, color, tr, step):
        ax.fill_between(
            np.array(tr),
            y1=np.array(qa[col][(ji*ei):(ji+1)*ei:step]).astype(float),
            y2=np.array(qb[col][(ji*ei):(ji+1)*ei:step]).astype(float),
            alpha=alpha, color=color,
        )

    # ============================================================
    # ax1: computation time
    # ============================================================
    # full
    ax1.plot(t_range, df['online_time'][(j1*end_ind):(j1+1)*end_ind:stride], 'b-',
             linewidth=1, label="online clustering", marker="v", ms=0.8)
    _band(ax1, quantiles[q1], quantiles[q2], 'online_time', j1, end_ind, 'b', t_range, stride)
    ax1.plot(t_range, df['MRO_time'][(j4*end_ind):(j4+1)*end_ind:stride], 'r-',
             linewidth=1, label="Reclustering full", marker="D", ms=0.8)
    _band(ax1, quantiles[q1], quantiles[q2], 'MRO_time', j4, end_ind, 'r', t_range, stride)
    ax1.plot(t_range, df1['DRO_time'][(j3*end_ind):(j3+1)*end_ind:stride], 'k-',
             linewidth=1, label="DRO", marker="s", ms=0.8)
    _band(ax1, quantiles1[q1], quantiles1[q2], 'DRO_time', j3, end_ind, 'black', t_range, stride)
    ax1.plot(t_range, df1['SA_time'][(j3*end_ind):(j3+1)*end_ind:stride], 'g-',
             linewidth=1, label="SAA", marker="o", ms=0.8)
    _band(ax1, quantiles1[q1], quantiles1[q2], 'SA_time', j3, end_ind, 'g', t_range, stride)
    # subgrad overlay (no SAA) -- uses end_ind_grad and t_range_grad,
    # and a lighter shade of each full color (b -> cornflowerblue,
    # r -> salmon, k -> gray) so full vs subgrad is distinguishable by both
    # linestyle and hue.
    # ax1.plot(t_range_grad, df2['online_time'][(j1g*end_ind_grad):(j1g+1)*end_ind_grad:stride_grad],
    #          color='cornflowerblue', linestyle='--',
    #          linewidth=1, label="online clustering subgrad", marker="v", ms=0.8)
    # _band(ax1, quantiles2[q1], quantiles2[q2], 'online_time', j1g, end_ind_grad, 'cornflowerblue', t_range_grad, stride_grad)
    # ax1.plot(t_range_grad, df2['MRO_time'][(j4g*end_ind_grad):(j4g+1)*end_ind_grad:stride_grad],
    #          color='salmon', linestyle='--',
    #          linewidth=1, label="reclustering subgrad", marker="D", ms=0.8)
    # _band(ax1, quantiles2[q1], quantiles2[q2], 'MRO_time', j4g, end_ind_grad, 'salmon', t_range_grad, stride_grad)
    ax1.plot(t_range_grad, df3['DRO_time'][(j3g*end_ind_grad):(j3g+1)*end_ind_grad:stride_grad],
             color='gray', linestyle='--',
             linewidth=1, label="DRO subgrad", marker="s", ms=0.8)
    _band(ax1, quantiles3[q1], quantiles3[q2], 'DRO_time', j3g, end_ind_grad, 'gray', t_range_grad, stride_grad)

    if 'cluster_SAA_time' in df.columns:
        ax1.plot(t_range, df['cluster_SAA_time'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA",marker="x",ms=0.8)
        if 'cluster_SAA_time' in quantiles[q1].columns:
            ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['cluster_SAA_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['cluster_SAA_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'm')

    ax1.set_xlabel(r'Time step $(t)$')
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    # ============================================================
    # ax2: in-sample objective value  (line handles captured for legend)
    # ============================================================
    # full
    line_online_full,    = ax2.plot(t_range, df['obj_values'][(j1*end_ind):(j1+1)*end_ind:stride], 'b-',
                                    linewidth=1, label="online clustering", marker="v", ms=0.8)
    _band(ax2, quantiles[q1], quantiles[q2], 'obj_values', j1, end_ind, 'b', t_range, stride)
    line_recluster_full, = ax2.plot(t_range, np.array(df['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:stride]), 'r-',
                                    linewidth=1, label="reclustering", marker="D", ms=0.8)
    _band(ax2, quantiles[q1], quantiles[q2], 'MRO_obj_values', j4, end_ind, 'r', t_range, stride)
    line_DRO_full,       = ax2.plot(t_range, df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:stride], 'k-',
                                    linewidth=1, label="DRO", marker="s", ms=0.8)
    _band(ax2, quantiles1[q1], quantiles1[q2], 'DRO_obj_values', j3, end_ind, 'black', t_range, stride)
    line_SAA,            = ax2.plot(t_range, df1['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:stride], 'g-',
                                    linewidth=1, label="SAA", marker="o", ms=0.8)
    _band(ax2, quantiles1[q1], quantiles1[q2], 'SA_obj_values', j3, end_ind, 'g', t_range, stride)
    # subgrad overlay -- lighter shades, dashed; uses end_ind_grad / t_range_grad.
    # line_online_grad,    = ax2.plot(t_range_grad, df2['obj_values'][(j1g*end_ind_grad):(j1g+1)*end_ind_grad:stride_grad],
    #                                 color='cornflowerblue', linestyle='--',
    #                                 linewidth=1, label="online clustering subgrad", marker="v", ms=0.8)
    # _band(ax2, quantiles2[q1], quantiles2[q2], 'obj_values', j1g, end_ind_grad, 'cornflowerblue', t_range_grad, stride_grad)
    # line_recluster_grad, = ax2.plot(t_range_grad, np.array(df2['MRO_obj_values'][(j4g*end_ind_grad):(j4g+1)*end_ind_grad:stride_grad]),
    #                                 color='salmon', linestyle='--',
    #                                 linewidth=1, label="reclustering subgrad", marker="D", ms=0.8)
    # _band(ax2, quantiles2[q1], quantiles2[q2], 'MRO_obj_values', j4g, end_ind_grad, 'salmon', t_range_grad, stride_grad)
    line_DRO_grad,       = ax2.plot(t_range_grad, df3['DRO_obj_values'][(j3g*end_ind_grad):(j3g+1)*end_ind_grad:stride_grad],
                                    color='gray', linestyle='--',
                                    linewidth=1, label="DRO subgrad", marker="s", ms=0.8)
    _band(ax2, quantiles3[q1], quantiles3[q2], 'DRO_obj_values', j3g, end_ind_grad, 'gray', t_range_grad, stride_grad)

    if 'cluster_SAA_obj_values' in df.columns:
        lines_cluster, = ax2.plot(t_range, df['cluster_SAA_obj_values'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA", marker="x",ms=0.8)
    if 'cluster_SAA_obj_values' in quantiles[q1].columns:
        ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['cluster_SAA_obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['cluster_SAA_obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'm')

    ax2.set_xlabel(r'Time step $(t)$')
    ax2.set_title(r'In-sample objective value')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")


    # ============================================================
    # ax3: confidence
    # ============================================================
    # full
    ax3.plot(t_range, df['O_satisfy0'][(j1*end_ind):(j1+1)*end_ind:stride], 'b-',
             linewidth=1, label="online clustering", marker="v", ms=0.8)
    ax3.plot(t_range, df['MRO_satisfy0'][(j4*end_ind):(j4+1)*end_ind:stride], 'r', linestyle='-',
             linewidth=1, label="reclustering", marker="D", ms=0.8)
    ax3.plot(t_range, df1['DRO_satisfy1'][(j3*end_ind):(j3+1)*end_ind:stride], 'k-',
             linewidth=1, label="DRO", marker="s", ms=0.8)
    ax3.plot(t_range, df1['SA_satisfy1'][(j3*end_ind):(j3+1)*end_ind:stride], 'g-',
             linewidth=1, label="SAA", marker="o", ms=0.8)
    # subgrad -- lighter shades, dashed; uses end_ind_grad / t_range_grad.
    # ax3.plot(t_range_grad, df2['O_satisfy0'][(j1g*end_ind_grad):(j1g+1)*end_ind_grad:stride_grad],
    #          color='cornflowerblue', linestyle='--',
    #          linewidth=1, label="online clustering subgrad", marker="v", ms=0.8)
    # ax3.plot(t_range_grad, df2['MRO_satisfy0'][(j4g*end_ind_grad):(j4g+1)*end_ind_grad:stride_grad],
    #          color='salmon', linestyle='--',
    #          linewidth=1, label="reclustering subgrad", marker="D", ms=0.8)
    ax3.plot(t_range_grad, df3['DRO_satisfy1'][(j3g*end_ind_grad):(j3g+1)*end_ind_grad:stride_grad],
             color='gray', linestyle='--',
             linewidth=1, label="DRO subgrad", marker="s", ms=0.8)
    if 'cluster_SAA_satisfy1' in df.columns:
        ax3.plot(t_range, df['cluster_SAA_satisfy1'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA",marker="x",ms=0.8)

    ax3.set_xlabel(r'Time step $(t)$')
    ax3.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax3.grid(True, alpha=0.3)
    # ax3.set_xscale("log")

    # ---- shared legend (7 handles, full block + SAA + subgrad block) ----------
    lines = [
         line_online_full,line_recluster_full, line_DRO_full,
         line_DRO_grad,line_SAA, lines_cluster
    ]
    labels = [ln.get_label() for ln in lines]
    if legend:
        fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6)
    plt.tight_layout()
    plt.savefig(folderout + 'obj_analysis_compare' + str(K) + '.pdf',
                bbox_inches='tight', dpi=300)


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
    # ax1.plot(t_range, df['O_eval0'][(j1*end_ind)+0:(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)

    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval0'][(j1*end_ind)+0:(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['O_eval0'][(j1*end_ind)+0:(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    # ax1.plot(t_range, df['MRO_eval0'][(j2*end_ind)+0:(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering", marker="D",ms=0.8)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval0'][(j2*end_ind)+0:(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_eval0'][(j2*end_ind)+0:(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # # DRO and SAA
    # ax1.plot(t_range, df1['SA_eval1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=0.8)
    # ax1.plot(t_range, df1['DRO_eval1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=0.8)
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='g')
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_eval1'][(j3*end_ind)+0:(j3+1)*end_ind:2]).astype(float),alpha=alpha, color='black')

    # ax1.set_ylim(ylim)
    # # plt.legend()
    # ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_title(r'Out-of-sample expected value')
    # ax1.grid(True, alpha=0.3)

    ax1.plot(t_range, df['online_time'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)
    
    # ax1.plot(t_range, np.array(df['MRO_time'][(j2*end_ind):(j2+1)*end_ind:2])+np.array(df['MRO_worst_times'][(j1*end_ind):(j1+1)*end_ind:2]), 'r-', linewidth=1, label = "reclustering",marker="D",ms=0.8)

    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['online_time'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    
    # ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')

    # reclustering worst
    ax1.plot(t_range, df['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2], 'r', linewidth=1, label = r"reclustering", marker="D",ms=0.8)
    
    # ax1.plot(t_range, quantiles[50]['MRO_time'][(j4*end_ind)+0:(j4+1)*end_ind:2], 'r:', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=0.8)
    
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_time'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color = 'r')
    
    # DRO and SAA
    ax1.plot(t_range, df1['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=0.8)
    ax1.plot(t_range, df1['SA_time'][(j3*end_ind):(j3+1)*end_ind:2], color ='g', linewidth=1, label = "SAA",marker="o",ms=0.8)
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    ax1.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_time'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'g')

    ax1.set_xlabel(r'Time step $(t)$')
    # ax1.set_xscale("log")
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim([1e-4,1e3])
    ax1.set_yscale("log")


    # online and reclustering
    lines1, = ax2.plot(t_range, np.array(df['obj_values'][(j1*end_ind):(j1+1)*end_ind:2])+5*np.array(df['sig_val'][(j1*end_ind):(j1+1)*end_ind:2]), 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)
    
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q1]['sig_val'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['obj_values'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q2]['sig_val'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')

    lines2, = ax2.plot(t_range, np.array(df['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2])+5*np.array(df['MRO_sig_val'][(j4*end_ind):(j4+1)*end_ind:2]), 'r', linewidth=1, label = "reclustering", marker="D",ms=0.8)
    
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q1]['MRO_sig_val'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q2]['MRO_sig_val'][(j4*end_ind):(j4+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # reclustering worst
    # ax2.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')
    # lines5, = ax2.plot(t_range, df['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=0.8)

    # DRO and SAA
    lines3, = ax2.plot(t_range, df1['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=0.8)
    lines4, = ax2.plot(t_range, df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO", marker="s",ms=0.8)
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    ax2.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['SA_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'g')

    ax2.set_xlabel(r'Time step $(t)$')
    # ax2.set_xscale("log")
    ax2.set_title(r'Certificate')
    ax2.set_ylim(ylim)
    ax2.grid(True, alpha=0.3)

    # online and reclustering
    ax3.plot(t_range, df['O_worst_satisfy1'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)

    ax3.plot(t_range, df['MRO_worst_satisfy1'][(j4*end_ind):(j4+1)*end_ind:2], 'r',linestyle='-', linewidth=1, label = "reclustering",marker="D",ms=0.8)
    # reclustering worst
    # ax3.plot(t_range, df['MRO_worst_satisfy1'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$",marker="^",ms=0.8)
    # DRO and SAA
    ax3.plot(t_range, df1['SA_satisfy2'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA",marker="o",ms=0.8)
    ax3.plot(t_range, df1['DRO_satisfy2'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=0.8)
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

def plot_eval(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,legend = True,end_ind_dro = 61):
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
    t_range_dro = np.array(df['t'])[(0*end_ind_dro):(1)*end_ind_dro:2] +1
    plt.figure(figsize=(4.3, 2.1), dpi=300)
    plt.plot(t_range, df['O_eval1'][(j1*end_ind):(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering" , marker="v",ms=0.8)

    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['O_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['O_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'b')
    plt.plot(t_range, df['MRO_eval1'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering", marker="D",ms=0.8)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_eval1'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_eval1'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    plt.plot(t_range_dro, df1['DRO_eval2'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], 'black', linewidth=1, label = "DRO", marker="s",ms=0.8)
    plt.fill_between(np.array(t_range_dro),y1=np.array(quantiles1[q1]['DRO_eval2'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_eval2'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),alpha=alpha, color='black')

    plt.plot(t_range_dro, df1['SA_eval2'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2], 'g-', linewidth=1, label = "SAA", marker="o",ms=0.8)
    plt.fill_between(np.array(t_range_dro),y1=np.array(quantiles1[q1]['SA_eval2'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),y2=np.array(quantiles1[q2]['SA_eval2'][(j3*end_ind_dro):(j3+1)*end_ind_dro:2]).astype(float),alpha=alpha, color='g')

    if 'cluster_SAA_eval1' in df.columns:
        plt.plot(t_range, df['cluster_SAA_eval1'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA",marker="x",ms=0.8)
        if 'cluster_SAA_eval1' in quantiles[q1].columns:
            plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['cluster_SAA_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['cluster_SAA_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'm')

    plt.xscale("log")
    # plt.ylim([0.008,0.04])
    if legend:
        plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.title(f'Out-of-sample expected value, $K$ = {K}')
    plt.grid(True, alpha=alpha)
    plt.savefig(folderout+f'eval_analysis{K}.pdf', bbox_inches='tight', dpi=300)

def plot_eval_compare(
    df,  quantiles,  df1, quantiles1,
    df2, quantiles2, df3, quantiles3,
    end_ind=61,
    end_ind_grad=None,   # subgrad epsilon-block length; falls back to end_ind if None
    j=(0, 0, 0),         # full epsilon-indices: (online, reclustering, DRO/SAA)
    j_grad=(0, 0, 0),    # subgrad epsilon-indices: (online, reclustering, DRO)
    stride=2,            # subsample stride within an epsilon block (full)
    stride_grad=None,    # subgrad stride; falls back to stride if None
    q=(40, 60),
    K=5,
    alpha=0.1,
    legend=True,
):
    """Out-of-sample expected value -- full vs subgrad overlay.

    Mirrors ``plot_eval`` for the full set (solid lines, suffix "full") and
    overlays a second subgrad set (dashed lines + lighter colors, suffix "subgrad").
    SAA is not overlaid.

    Method -> data source:
      online clustering : df[K]   / df2[K]
      reclustering      : df[K]   / df2[K]
      DRO               : df1[0]  / df3[0]
      SAA               : df1[0]  ONLY  (no subgrad overlay)
    """
    if end_ind_grad is None:
        end_ind_grad = end_ind
    if stride_grad is None:
        stride_grad = stride
    j1, j2, j3 = j
    j1g, j2g, j3g = j_grad
    df  = df[K]
    df1 = df1[0]
    df2 = df2[K]
    df3 = df3[0]
    quantiles  = quantiles[K]
    quantiles1 = quantiles1[0]
    quantiles2 = quantiles2[K]
    quantiles3 = quantiles3[0]
    q1, q2 = q
    fontsize = 10
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": 11,
        "legend.fontsize": fontsize,
    })

    t_range      = np.array(df['t'])[(0*end_ind):(1)*end_ind:stride] + 1
    t_range_grad = np.array(df2['t'])[(0*end_ind_grad):(1)*end_ind_grad:stride_grad] + 1
    plt.figure(figsize=(4.3, 2.1), dpi=300)

    def _band(qa, qb, col, ji, ei, color, tr, step):
        plt.fill_between(
            np.array(tr),
            y1=np.array(qa[col][(ji*ei):(ji+1)*ei:step]).astype(float),
            y2=np.array(qb[col][(ji*ei):(ji+1)*ei:step]).astype(float),
            alpha=alpha, color=color,
        )

    # full
    plt.plot(t_range, df['O_eval1'][(j1*end_ind):(j1+1)*end_ind:stride], 'b-',
             linewidth=1, label="online clustering", marker="v", ms=0.8)
    _band(quantiles[q1], quantiles[q2], 'O_eval1', j1, end_ind, 'b', t_range, stride)
    plt.plot(t_range, df['MRO_eval1'][(j2*end_ind):(j2+1)*end_ind:stride], 'r-',
             linewidth=1, label="reclustering", marker="D", ms=0.8)
    _band(quantiles[q1], quantiles[q2], 'MRO_eval1', j2, end_ind, 'r', t_range, stride)
    plt.plot(t_range, df1['DRO_eval2'][(j3*end_ind):(j3+1)*end_ind:stride], color='black', linestyle='-',
             linewidth=1, label="DRO", marker="s", ms=0.8)
    _band(quantiles1[q1], quantiles1[q2], 'DRO_eval2', j3, end_ind, 'black', t_range, stride)
    plt.plot(t_range, df1['SA_eval2'][(j3*end_ind):(j3+1)*end_ind:stride], 'g-',
             linewidth=1, label="SAA", marker="o", ms=0.8)
    _band(quantiles1[q1], quantiles1[q2], 'SA_eval2', j3, end_ind, 'g', t_range, stride)
    # subgrad overlay (no SAA) -- lighter shades, dashed.
    # plt.plot(t_range_grad, df2['O_eval1'][(j1g*end_ind_grad):(j1g+1)*end_ind_grad:stride_grad],
    #          color='cornflowerblue', linestyle='--',
    #          linewidth=1, label="online clustering subgrad", marker="v", ms=0.8)
    # _band(quantiles2[q1], quantiles2[q2], 'O_eval1', j1g, end_ind_grad, 'cornflowerblue', t_range_grad, stride_grad)
    # plt.plot(t_range_grad, df2['MRO_eval1'][(j2g*end_ind_grad):(j2g+1)*end_ind_grad:stride_grad],
    #          color='salmon', linestyle='--',
    #          linewidth=1, label="reclustering subgrad", marker="D", ms=0.8)
    # _band(quantiles2[q1], quantiles2[q2], 'MRO_eval1', j2g, end_ind_grad, 'salmon', t_range_grad, stride_grad)
    plt.plot(t_range_grad, df3['DRO_eval2'][(j3g*end_ind_grad):(j3g+1)*end_ind_grad:stride_grad],
             color='gray', linestyle='--',
             linewidth=1, label="DRO subgrad", marker="s", ms=0.8)
    _band(quantiles3[q1], quantiles3[q2], 'DRO_eval2', j3g, end_ind_grad, 'gray', t_range_grad, stride_grad)
    if 'cluster_SAA_eval1' in df.columns:
        plt.plot(t_range, df['cluster_SAA_eval1'][(j1*end_ind):(j1+1)*end_ind:2], color ='m', linewidth=1, label = "cluster SAA",marker="x",ms=0.8)
        if 'cluster_SAA_eval1' in quantiles[q1].columns:
            plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['cluster_SAA_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['cluster_SAA_eval1'][(j1*end_ind):(j1+1)*end_ind:2]).astype(float),alpha=alpha, color = 'm')

    plt.xscale("log")
    # plt.ylim([0.008, 0.04])
    if legend:
        plt.legend(ncol=2, fontsize=fontsize - 2)
    plt.xlabel(r'Time step $(t)$')
    plt.title(f'Out-of-sample expected value, $K$ = {K}')
    plt.grid(True, alpha=alpha)
    plt.savefig(folderout + f'eval_analysis_compare{K}.pdf', bbox_inches='tight', dpi=300)


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
    plt.plot(t_range, df['O_satisfy1'][(j1*end_ind)+1:(j1+1)*end_ind:2], 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)
    plt.plot(t_range, df['MRO_satisfy1'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r-', linewidth=1, label = "reclustering",marker="D",ms=0.8)
    
    #reclustering worst
    plt.plot(t_range, df['MRO_worst_satisfy1'][(j2*end_ind)+1:(j2+1)*end_ind:2], 'r:', linewidth=1, label = r"reclustering $\hat{H}^K_t$",marker="^",ms=0.8)

    # DRO and SAA
    plt.plot(t_range, df1['SA_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], 'g-', linewidth=1, label = "SAA",marker="o",ms=0.8)
    plt.plot(t_range, df1['DRO_satisfy1'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = "DRO",marker="s",ms=0.8)
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

    plt.plot(t_range, 5*df['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]+ np.array([5*np.sum(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'b:', linewidth=1, label = "online clustering UB", marker="o",ms=0.8)
    
    plt.plot(t_range, 5*np.array(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2]), 'b--', label = r"online clustering $\Phi^K_t$",linewidth = 0.5)

    plt.plot(t_range, np.array([np.sum((np.array(df['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)



    plt.plot(t_range, 5*df['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2]+ np.array([5*np.sum(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]) , 'r:', linewidth=1, label = "reclustering UB", marker="D",ms=0.8)

    plt.plot(t_range, 5*np.array(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2]), 'r--',label = r"reclustering $\Phi^K_t$" , linewidth = 0.5)

    plt.plot(t_range, np.array([np.sum((np.array(df['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'r-', linewidth=1, label = "reclustering", marker="D",ms=0.8)
    

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


def plot_regret_new(df, quantiles, df1=None, quantiles1=None,end_ind=61,j=(0,0,0), q = (40,60),K=5, alpha=0.1,ylim=[0.008,0.022]):
    j1,j2,j3 = j
    # Set up LaTeX rendering
    df = df[K]
    # df = quantiles[K][50].copy()
    # df1 = quantiles[0][50].copy()
    quantiles = quantiles[K].copy()
    fontsize= 10
    q1,q2 = q
    radius = [0.0012*(1/((t+5)**(1/(40)))) for t in range(2001)]
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

    # plt.plot(t_range, 5*df['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]+ np.array([5*np.sum(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'b:', linewidth=1, label = "online clustering UB", marker="o",ms=0.8)
    
    # plt.plot(t_range, 5*np.array(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2]), 'b--', label = r"online clustering $\Phi^K_t$",linewidth = 0.5)

    # plt.plot(t_range, np.array([np.sum((np.array(df['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'b-', linewidth=1, label = "online clustering", marker="v",ms=0.8)


    plt.plot(t_range, 5*df['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2]+ np.array([5*np.sum(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), color='cornflowerblue', linestyle = ":", linewidth=1, label = "reclustering UB", marker="D",ms=0.8)

    # plt.plot(t_range, 5*np.array(df['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2]), 'r--',label = r"reclustering $\Phi^K_t$" , linewidth = 0.5)

    plt.plot(t_range, np.array([np.sum((np.array(df['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(df1['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), 'r-', linewidth=1, label = "reclustering empirical", marker="D",ms=0.8)
    

    # plt.fill_between(np.array(t_range),y1=[np.sum((np.array(quantiles[q1]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],y2=[np.sum((np.array(quantiles[q2]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])-np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],alpha=alpha, color = 'b')

    

    plt.fill_between(np.array(t_range),y1=[np.sum((np.array(quantiles[q1]['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],y2=[np.sum((np.array(quantiles[q2]['MRO_worst_values_regret'][(j2*end_ind+1):(j2+1)*end_ind:2])-np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind+1):(j3+1)*end_ind:2]))[:i+1])/(i) for i in range(1,int((end_ind)/2)+1)],alpha=alpha, color = 'r')


    # theoretical bounds
   
    # plt.fill_between(np.array(t_range),y1=np.array(5*quantiles[q1]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+ np.array([5*np.sum(quantiles[q1]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in  range(1,int((end_ind)/2)+1)]) ,y2=np.array(5*quantiles[q2]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+np.array([5*np.sum(quantiles[q2]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]) ,alpha=alpha, color = 'b')

    
    plt.fill_between(np.array(t_range),y1=np.array(5*quantiles[q1]['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2])+ np.array([5*np.sum(quantiles[q1]['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in  range(1,int((end_ind)/2)+1)]) ,y2=np.array(5*quantiles[q2]['MRO_regret_bound'][(j2*end_ind+1):(j2+1)*end_ind:2])+np.array([5*np.sum(quantiles[q2]['MRO_sig_val'][(j2*end_ind+1):(j2+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]),alpha=alpha, color = 'cornflowerblue')


    plt.legend(ncol = 2)
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Dynamic regret')
    plt.ylim(ylim)
    plt.yscale('log')
    plt.xscale('log')
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
    plt.plot(t_range, np.array(df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2])+ 5*np.array(df['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]), color ='purple', linewidth=1, label = r"DRO $H_t + \underline{\psi}^K_t$", marker="s",ms=0.8)

    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2])+5*np.array(quantiles[q1]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]) ,y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2])+5*np.array(quantiles[q2]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]),alpha=alpha, color = 'purple')

    # reclustering upper
    plt.plot(t_range, df['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2] + 5*df['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2], 'r', linewidth=1, label = r"reclustering $H^K_t + \underline{\psi}^K_t$", marker="D",ms=0.8)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float)+ 5*np.array(quantiles[q1]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float)+5*np.array(quantiles[q2]['MRO_sig_val'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')


    # DRO
    plt.plot(t_range, df1['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2], color ='black', linewidth=1, label = r"DRO $H_t$", marker="s",ms=0.8)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),y2=np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind):(j3+1)*end_ind:2]).astype(float),alpha=alpha, color = 'black')
    
    # reclustering 
    plt.plot(t_range, df['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2], 'orange', linewidth=1, label = r"reclustering $H^K_t$", marker="v",ms=0.8)
    plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_obj_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='orange')

    # # reclustering worst
    # plt.fill_between(np.array(t_range),y1=np.array(quantiles[q1]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),y2=np.array(quantiles[q2]['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2]).astype(float),alpha=alpha, color='r')

    # plt.plot(t_range, df['MRO_worst_values'][(j2*end_ind):(j2+1)*end_ind:2], 'r-', linewidth=1, label = r"reclustering $\hat{H}^K_t$", marker="^",ms=0.8)
    

    
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

# foldername = '/scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/port_new/results/orig/p1/1/T'+str(T-1)+'R'+str(R)+'/'

# folderout = '/Users/irina.wang/Desktop/Princeton/Project2/mro_mpc/port_new/results_new_p2_plots/2/T'+str(T-1)+'R'+str(R)+'/'

# folderout = '/scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/port_new/plots_new/orig/1/T'+str(T-1)+'R'+str(R)+'/'

# os.makedirs(folderout, exist_ok=True)

# folderout2 = '/Users/irina.wang/Desktop/Princeton/Project2/mro_mpc/port_new/results_new_p1_plots/2/T'+str(T-1)+'R'+str(R)+'/'
# folderout2 = '/scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/port_new/results_new_p1_plots/2/T'+str(T-1)+'R'+str(R)+'/'

# os.makedirs(folderout2, exist_ok=True)

# setup MRO dfs
def setup_dfs(folderout = None, foldername = None, K_list = K_list, init = False):
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

# preamble = "/Users/irina.wang/Desktop/Princeton/Project2/mro_mpc/"
preamble = "/scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/"

foldername_orig = preamble + 'portfolio_time/results/3/T'+str(T-1)+'R'+str(R)+'/'

folderout_orig = preamble + 'portfolio_time/plots/3/T'+str(T-1)+'R'+str(R)+'/'

os.makedirs(folderout_orig, exist_ok=True)

foldername_orig_dro = preamble + 'portfolio_time/results/1/T'+str(T-1)+'R'+str(R)+'/'

folderout_orig_dro = preamble + 'portfolio_time/plots/1/T'+str(T-1)+'R'+str(R)+'/'

os.makedirs(folderout_orig_dro, exist_ok=True)


folderout = preamble + 'portfolio_time/plots/3/T'+str(T-1)+'R'+str(R)+'/'

os.makedirs(folderout, exist_ok=True)


df_orig, quantiles_orig = setup_dfs(foldername = foldername_orig, folderout = folderout_orig, K_list = [15,25], init = True)

df_orig_dro, quantiles_orig_dro = setup_dfs(foldername = foldername_orig_dro, folderout = folderout_orig_dro, K_list = [0], init = True)

# df1,quantiles1 = setup_dfs(folderout = folderout,init = False)

end_ind_orig = infer_end_ind(df_orig, K=25)
end_ind_dro = infer_end_ind(df_orig_dro, K=25)

plot_eval_all(df_orig,quantiles_orig,df_orig_dro,quantiles_orig_dro,j=(2,3,3),K=25,q=(25,75),ylim=[0.004,0.02],legend = True,val2=2.3, end_ind=end_ind_orig,end_ind_dro = end_ind_dro)

plot_eval(df_orig,quantiles_orig,df_orig_dro,quantiles_orig_dro,j=(2,3,3),K=25,q=(25,75),end_ind=end_ind_orig,legend = True,end_ind_dro = end_ind_dro)


# plot_eval_all_compare(
#     df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
#     df_new, quantiles_new, df_new_dro, quantiles_new_dro,
#     j=(4,4,5), j_grad=(0,0,0), K=25, q=(25,75),
#     end_ind=end_ind_orig, end_ind_grad= end_ind_new,val2=2.3, legend=True,
# )

# plot_eval_compare(
#    df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
#     df_new, quantiles_new, df_new_dro, quantiles_new_dro,
#     j=(4,4,5), j_grad=(0,0,0), K=25, q=(25,75),
#     end_ind=end_ind_orig, end_ind_grad= end_ind_new, legend=True,
# )

# plot_regret_new(df_orig,quantiles_orig,df_orig_dro[0],quantiles_orig_dro[0],j=(4,4,5),K=25,q=(25,75),end_ind = end_ind_orig,ylim=[0.0005,1])