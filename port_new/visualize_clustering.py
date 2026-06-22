"""Visualize the online-clustering update and full reclustering side-by-side
on the synthetic-returns dataset, projected to 2D via PCA, and write the
animation as a GIF.

Mirrors the port_orig.py loop's two clustering branches:
  * online clustering  -- incremental update of (q_dict, k_dict) via
    `online_cluster_init_online` / `online_cluster_update_online`.
  * reclustering       -- fresh `sklearn.KMeans` on running_samples every
    `interval` timesteps, with the previous cluster centers as warm-start.

PCA (top-2 components) is fitted ONCE on the full training set so the 2D
coordinates stay comparable across frames.  Each scatter-marker size is
proportional to that cluster's weight.

Usage
-----
    python visualize_clustering.py

The output GIF is written next to this script.
"""

import os
import sys

# Allow `from utils import ...` when run directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from utils import (
    pca,
    online_cluster_init_online as online_cluster_init,
    online_cluster_update_online as online_cluster_update,
)


# ---------------- configuration ----------------
K          = 25                # number of macro clusters
Q          = 1000               # number of micro clusters (>= K)
T          = 2001              # timesteps 0..2000 inclusive
N_init     = 5                 # initial samples (matches t + N_init == num_dat)
m          = 150                # feature dim (slice synthetic_returns[:, :m])
interval   = 5                # reclustering cadence
fixed_time = 10_000            # set high so we never switch to fixed_cluster
NUM_FRAMES = 60                # frames in the output GIF
FPS        = 6
init_ind   = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = os.path.join(SCRIPT_DIR, 'synthetic_200_1.csv')
OUT_GIF    = os.path.join(SCRIPT_DIR, 'clustering_2d.gif')

# ---------------- data ----------------
synthetic_returns = pd.read_csv(DATA_CSV).to_numpy()[:, 1:][:, :m]
dat, _ = train_test_split(
    synthetic_returns, train_size=19000, test_size=1000, random_state=0,
)

# ---------------- 2D PCA projection (fixed across frames) ----------------
pca_A, pca_b = pca(dat, 2)             # A: (2, m),  b: (m,)
def project(X):
    return (np.asarray(X) - pca_b) @ pca_A.T

# ---------------- simulate online + reclustering ----------------
num_dat = N_init
q_dict, k_dict, _ = online_cluster_init(K, Q, dat[init_ind:(init_ind+num_dat)], m)
new_k_dict = None

frame_times = sorted(set(np.linspace(0, T - 1, NUM_FRAMES, dtype=int)))
frames = []

for t in range(T):
    running_samples = dat[init_ind:(init_ind+num_dat)]

    # MRO reclustering every `interval` timesteps
    if t % interval == 0 and num_dat >= 2:
        cur_K = int(min(K, num_dat))
        if (new_k_dict is not None and
                num_dat > (interval + N_init) and
                new_k_dict['d'].shape[0] == cur_K):
            kmeans = KMeans(n_clusters=cur_K, init=new_k_dict['d'], n_init=1).fit(running_samples)
        else:
            kmeans = KMeans(n_clusters=cur_K, init='k-means++', n_init=1).fit(running_samples)
        new_k_dict = {
            'K': cur_K,
            'd': kmeans.cluster_centers_,
            'w': np.bincount(kmeans.labels_) / num_dat,
        }

    if t in frame_times:
        cur_K_online = int(k_dict['K'])
        frames.append({
            't': t,
            'num_dat': num_dat,
            'samples_2d':        project(running_samples),
            'online_centers_2d': project(k_dict['d'][:cur_K_online]),
            'online_weights':    k_dict['w'][:cur_K_online].copy(),
            'mro_centers_2d':    (project(new_k_dict['d'])
                                  if new_k_dict is not None else None),
            'mro_weights':       (new_k_dict['w'].copy()
                                  if new_k_dict is not None else None),
        })

    # incremental online-cluster update with the next sample
    new_sample = dat[init_ind + num_dat]
    q_dict, k_dict, _ = online_cluster_update(
        K, new_sample, q_dict, k_dict, num_dat, t, fixed_time, m, Q,
    )
    num_dat += 1

# ---------------- animation ----------------
fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=120)

# All 2D points (across frames) for stable axis bounds.  Use a percentile
# clip so a few outlier samples do not blow up the visible range.
all_pts = np.vstack([f['samples_2d'] for f in frames])
p_lo = np.percentile(all_pts, 1.0, axis=0)
p_hi = np.percentile(all_pts, 99.0, axis=0)
pad = 0.05 * (p_hi - p_lo)
ax.set_xlim(p_lo[0] - pad[0], p_hi[0] + pad[0])
ax.set_ylim(p_lo[1] - pad[1], p_hi[1] + pad[1])
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.grid(True, alpha=0.25)

samples_h = ax.scatter([], [], c='lightgray', s=10, alpha=0.5, zorder=1,
                       label='samples')
online_h  = ax.scatter([], [], c='blue', marker='v',
                       edgecolors='black', linewidths=0.6, zorder=3,
                       label='online clustering K')
mro_h     = ax.scatter([], [], c='red', marker='D',
                       edgecolors='black', linewidths=0.6, zorder=3,
                       label='reclustering K')
ax.legend(loc='upper right')
title_h = ax.set_title('')


def _w_to_size(w):
    """Scale a cluster weight (in [0, 1], sums to ~1) into a marker area."""
    return 60.0 + 1500.0 * np.asarray(w, dtype=float)


def init():
    samples_h.set_offsets(np.empty((0, 2)))
    online_h.set_offsets(np.empty((0, 2)))
    mro_h.set_offsets(np.empty((0, 2)))
    title_h.set_text('')
    return samples_h, online_h, mro_h, title_h


def update(idx):
    f = frames[idx]
    samples_h.set_offsets(f['samples_2d'])

    online_h.set_offsets(f['online_centers_2d'])
    online_h.set_sizes(_w_to_size(f['online_weights']))

    if f['mro_centers_2d'] is not None:
        mro_h.set_offsets(f['mro_centers_2d'])
        mro_h.set_sizes(_w_to_size(f['mro_weights']))
    else:
        mro_h.set_offsets(np.empty((0, 2)))

    title_h.set_text(f"K = {K}, t = {f['t']}")
    return samples_h, online_h, mro_h, title_h


anim = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=len(frames), interval=int(1000 / FPS), blit=False,
)
anim.save(OUT_GIF, writer=animation.PillowWriter(fps=FPS))
plt.close(fig)
print(f"Wrote {OUT_GIF}  ({len(frames)} frames at {FPS} fps)")
