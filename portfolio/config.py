"""Default parameters for the portfolio box-support W1-DRO experiments.

Single source of truth for the values ACTUALLY LAUNCHED by
``slurm/portfolio_box.sh`` (m=300, power=0.05, box_q=0.3, T=2001, N_init=5,
...), falling back to the legacy argparse defaults where the slurm script did
not override them.  Legacy drivers (in ``port_box/``):

  ("mro", "subgrad")  <- port_box.py
  ("mro", "exact")    <- port_box_orig.py
  ("dro", "subgrad")  <- port_box_DRO.py
  ("dro", "exact")    <- port_box_DRO_orig.py
  ("true_saa", None)  <- port_box_true_saa.py
"""

# Shared across all paths (slurm passed these to every driver; true_saa uses
# the same data model so it must match -- its legacy argparse default m=200
# predates the m=300 launches).
COMMON = {
    'm': 300,
    'power': 0.05,
    'box_q': 0.3,
    'n_total': 20000,
    'alpha': 0.8,          # a = -1/(1-alpha) = -5
    'seed': 12345,         # generate_returns seed
    'test_size': 1000,     # train_size = min(n_total - test_size, 19000)
    'r_start': 0,
}

# Per-path defaults = slurm-launched values / legacy argparse defaults.
DEFAULTS = {
    ('mro', 'subgrad'): {
        'T': 2001, 'R': 10, 'K': 15, 'interval': 1, 'N_init': 5, 'Q': 500,
        'fixed_time': 2001, 'eta_0': 0.1, 'rmse_mult': 1.25,
        'cluster_interval': 50, 'line_search': False, 'solve_interval': 200,
    },
    ('mro', 'exact'): {
        'T': 2001, 'R': 10, 'K': 15, 'interval': 100, 'N_init': 5, 'Q': 500,
        'fixed_time': 2001, 'rmse_mult': 1.25, 'cluster_interval': 1,
    },
    ('dro', 'subgrad'): {
        'T': 2001, 'R': 10, 'interval': 1, 'N_init': 5, 'eta_0': 0.1,
        'line_search': False, 'solve_interval': 200,
    },
    ('dro', 'exact'): {
        # slurm launched port_box_DRO_orig.py with --R 5; interval_SAA was not
        # overridden, so the legacy argparse default 25 governs the logging grid.
        'T': 2001, 'R': 5, 'interval': 100, 'interval_SAA': 25, 'N_init': 5,
    },
    ('true_saa', None): {
        'R': 10, 'N_true': 20000,
    },
}

# Epsilon sweeps, copied EXACTLY from the legacy drivers.
EPS = {
    ('mro', 'subgrad'): [0.007, 0.005, 0.003, 0.0025, 0.002, 0.001],      # port_box.py:515
    ('mro', 'exact'): [0.02, 0.01, 0.007, 0.005, 0.003, 0.0025],          # port_box_orig.py:623
    ('dro', 'subgrad'): [0.003, 0.0025, 0.002, 0.001],                    # port_box_DRO.py:168
    ('dro', 'exact'): [0.02, 0.01, 0.005, 0.003, 0.0025, 0.002, 0.001],   # port_box_DRO_orig.py:200
}

# port_box_orig.py:624-625: the mro/exact sweep switches to this list when T >= 5000.
EPS_MRO_EXACT_LONG_T = [0.0035, 0.003, 0.0025, 0.002]

# Forced-checkpoint timesteps (logged regardless of interval).
_T_LIST_LONG = [4, 5, 9, 10, 14, 15, 19, 20, 1249, 1250, 1499, 1500, 1749,
                1750, 1999, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
T_LIST = {
    ('mro', 'subgrad'): _T_LIST_LONG,                                     # port_box.py:520
    ('mro', 'exact'): _T_LIST_LONG,                                       # port_box_orig.py:630
    ('dro', 'subgrad'): [4, 5, 9, 10, 49, 50, 99, 100, 249, 250, 499, 500],  # port_box_DRO.py:171
    ('dro', 'exact'): _T_LIST_LONG,                                       # port_box_DRO_orig.py:203
}

# Early re-anchor timesteps for the subgrad paths: a full exact solve replaces
# the subgradient iterate at these t's, in addition to every
# ``solve_interval``-th step (mirrors svm/config.py T_SOLVE_LIST).
T_SOLVE_LIST = {
    ('mro', 'subgrad'): [5, 25, 50, 100],
    ('dro', 'subgrad'): [5, 25, 50, 100],
}
