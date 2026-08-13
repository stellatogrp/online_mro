"""Single entry point for the SPARSE-SVM (ijcnn1) paper experiment.

Provenance: consolidates the ``__main__`` blocks of the five legacy drivers
``svm1/svm_grad.py`` (mro/subgrad), ``svm1/svm_orig.py`` (mro/exact),
``svm1/svm_DRO_grad.py`` (dro/subgrad), ``svm1/svm_DRO_orig.py`` (dro/exact)
and ``svm1/svm_true_saa.py`` (true_saa) into one argparse CLI.  The per-task
workers live in ``methods.py`` (verbatim copies of the legacy workers).

Usage (from ``paper_experiments/``):

    uv run python -m svm.run --method {mro,dro,true_saa} --solver {exact,subgrad} \
        --results_dir DIR [--T 2001] [--R 10] [--r_start 0] [--K 15] \
        [--interval N] [--eps_index I] [--checkpoint_every 200] [...]

Changes vs. legacy launch plumbing (numerics untouched):
  * K comes from ``--K`` (default 15) instead of the legacy hard requirement
    on ``SLURM_ARRAY_TASK_ID`` indexing ``K_arr = [10, 15, 25]``
    (``slurm/svm1.sh`` ran array index 1, i.e. K=15).
  * The final per-seed CSV directory is ``{results_dir}/T{T-1}/`` (legacy:
    ``T{T-1}R{R}/``; the R suffix is dropped so extra seeds can be added
    additively).  File names ``df_K{K}R{r}.csv`` are unchanged; the DRO
    paths hard-write K=0 as legacy.
  * All unset options default to the values actually launched by
    ``slurm/svm1.sh`` (see ``config.py``).
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .methods import (
    output_stream,
    run_true_saa,
    task_dro_exact,
    task_dro_subgrad,
    task_mro_exact,
    task_mro_subgrad,
)
from .utils import get_n_processes, remove_files, save_run_metadata
from .utils_svm import load_svm_dataset


def main():
    parser = argparse.ArgumentParser(
        description='SPARSE-SVM paper experiment (online MRO / DRO / oracle SAA).')
    parser.add_argument('--method', type=str, required=True,
                        choices=['mro', 'dro', 'true_saa'])
    parser.add_argument('--solver', type=str, choices=['exact', 'subgrad'],
                        default='exact',
                        help='exact = MOSEK MILP path; subgrad = IHT path '
                             '(ignored for --method true_saa).')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--R', type=int, default=None)
    parser.add_argument('--r_start', type=int, default=None)
    parser.add_argument('--K', type=int, default=None,
                        help='Number of macro-clusters (MRO paths). Default 15 '
                             '(replaces the legacy SLURM_ARRAY_TASK_ID K_arr indexing).')
    parser.add_argument('--interval', type=int, default=None,
                        help='Logging interval. Per-path default: subgrad 1, exact 200.')
    parser.add_argument('--eps_list', type=float, nargs='+', default=None,
                        help='override the config epsilon grid (descending); '
                             'use a separate --results_dir')
    parser.add_argument('--eps_index', type=int, default=None,
                        help='Run only this index of the epsilon sweep '
                             '(default: full sweep identical to legacy).')
    parser.add_argument('--checkpoint_every', type=int, default=200,
                        help='Write per-task checkpoint CSVs every this many '
                             'logged steps (legacy wrote them at every logged '
                             'step; the final CSV is identical).')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--k', type=int, default=None, help='Cardinality budget.')
    parser.add_argument('--Q', type=int, default=None)
    parser.add_argument('--fixed_time', type=int, default=None,
                        help='Freeze clusters from this step on (MRO paths). '
                             'Default: T (never freeze), as launched.')
    parser.add_argument('--N_init', type=int, default=None)
    parser.add_argument('--rmse_mult', type=float, default=None)
    parser.add_argument('--cluster_interval', type=int, default=None)
    parser.add_argument('--power', type=float, default=None)
    parser.add_argument('--p', type=int, default=None)
    parser.add_argument('--eta', type=float, default=None)
    parser.add_argument('--solve_interval', type=int, default=None)
    parser.add_argument('--interval_SAA', type=int, default=None)
    parser.add_argument('--N_true', type=int, default=None)

    args = parser.parse_args()
    method = args.method
    solver = None if method == 'true_saa' else args.solver
    defaults = config.DEFAULTS[(method, solver)]

    def get(name):
        v = getattr(args, name, None)
        return defaults.get(name) if v is None else v

    # Legacy drivers concatenated paths with a trailing '/'.
    foldername = args.results_dir
    if not foldername.endswith('/'):
        foldername = foldername + '/'

    if method == 'true_saa':
        run_true_saa(foldername, dataset=get('dataset'), k=get('k'),
                     N_true=get('N_true'), R=get('R'), r_start=get('r_start'))
        return

    T = get('T')
    R = get('R')
    r_start = get('r_start')
    dataset = get('dataset')
    k = get('k')
    N_init = get('N_init')
    power = get('power')
    p = get('p')
    interval = get('interval')
    checkpoint_every = args.checkpoint_every

    # Real LIBSVM classification data: one fixed pool (train + test files
    # concatenated); each seed draws its own train/test split inside the
    # worker. m (covariate dimension) is determined by the dataset.
    full_data, m = load_svm_dataset(dataset)
    n_total = full_data.shape[0]
    test_size = max(1000, n_total // 6)
    train_size = n_total - test_size

    init_ind = 0
    njobs = get_n_processes(100)
    if args.eps_list is not None:
        # Explicit epsilon grid override (e.g. confidence-tuning extension
        # runs); write these to a SEPARATE results_dir so block indices of
        # the main sweep stay stable.
        eps_init = list(args.eps_list)
    else:
        eps_init = config.eps_list(method, solver, T)
    M = len(eps_init)
    if args.eps_index is not None and not (0 <= args.eps_index < M):
        raise SystemExit(f"--eps_index must be in [0, {M - 1}] for this path")
    eps_indices = list(range(M)) if args.eps_index is None else [args.eps_index]
    list_inds = list(itertools.product(np.arange(R), np.array(eps_indices)))
    t_list = config.T_LIST
    t_solve_list = config.T_SOLVE_LIST
    # Final per-seed CSV directory: T{T-1}/ WITHOUT the legacy R suffix.
    newdatname = foldername + 'T' + str(T - 1) + '/'

    if method == 'mro':
        K = get('K')
        Q = get('Q')
        rmse_mult = get('rmse_mult')
        cluster_interval = get('cluster_interval')
        fixed_time = get('fixed_time')
        if fixed_time is None:
            fixed_time = T
        newfoldername = foldername + 'K' + str(K) + '_R' + str(R) + '_T' + str(T - 1) + '/'
        os.makedirs(newfoldername, exist_ok=True)
        print(newfoldername)

        if solver == 'subgrad':
            eta = get('eta')
            solve_interval = get('solve_interval')
            metadata = {
                'filename': os.path.basename(__file__),
                'dataset': dataset, 'n_total': n_total,
                'train_size': train_size, 'test_size': test_size,
                'K': K, 'T': T, 'R': R, 'm': m, 'k': k,
                'Q': Q,
                'fixed_time': fixed_time, 'interval': interval, 'N_init': N_init,
                'rmse_mult': rmse_mult, 'cluster_interval': cluster_interval,
                'r_start': r_start,
                'power': power,
                'p': p,
                'eta': eta,
                'solve_interval': solve_interval,
                'epsilon_values': [float(e) for e in eps_init],
                'num_epsilon_values': len(eps_init),
                'num_random_seeds': R,
                'total_test_combinations': len(eps_init) * R,
            }
        else:
            metadata = {
                'filename': os.path.basename(__file__),
                'dataset': dataset, 'n_total': n_total,
                'train_size': train_size, 'test_size': test_size,
                'K': K, 'T': T, 'R': R, 'm': m, 'k': k,
                'Q': Q,
                'fixed_time': fixed_time, 'interval': interval, 'N_init': N_init,
                'rmse_mult': rmse_mult, 'cluster_interval': cluster_interval,
                'r_start': r_start,
                'epsilon_values': [float(e) for e in eps_init],
                'num_epsilon_values': len(eps_init),
                'num_random_seeds': R,
                'total_test_combinations': len(eps_init) * R,
                'power': power,
                'p': p,
            }
        save_run_metadata(
            metadata,
            [newfoldername, (newdatname, f'metadata_K{K}.json')],
        )

        if solver == 'subgrad':
            results = Parallel(n_jobs=njobs)(delayed(task_mro_subgrad)(
                r_input, K, T, N_init, full_data, power, p, r_start,
                m=m, train_size=train_size, test_size=test_size,
                eps_init=eps_init, list_inds=list_inds, init_ind=init_ind,
                Q=Q, k=k, eta=eta, fixed_time=fixed_time,
                cluster_interval=cluster_interval, interval=interval,
                t_list=t_list, solve_interval=solve_interval,
                t_solve_list=t_solve_list, rmse_mult=rmse_mult,
                foldername=foldername)
                for r_input in range(len(list_inds)))
        else:
            results = Parallel(n_jobs=njobs)(delayed(task_mro_exact)(
                r_input, K, T, N_init, full_data, power, p, r_start,
                m=m, train_size=train_size, test_size=test_size,
                eps_init=eps_init, list_inds=list_inds, init_ind=init_ind,
                Q=Q, k=k, fixed_time=fixed_time,
                cluster_interval=cluster_interval, interval=interval,
                t_list=t_list, rmse_mult=rmse_mult, foldername=foldername,
                checkpoint_every=checkpoint_every)
                for r_input in range(len(list_inds)))

        dfs = {}
        for r in range(R):
            dfs[r] = {}
        for r_input in range(len(list_inds)):
            r, epsnum = list_inds[r_input]
            dfs[r][epsnum] = results[r_input]

        findfs = {}
        for r in range(R):
            findfs[r] = pd.concat([dfs[r][i] for i in eps_indices], ignore_index=True)
            findfs[r].to_csv(newfoldername + 'df_' + str(r + r_start) + '.csv')

        for r in range(R):
            if solver == 'exact':
                findfs[r] = findfs[r].drop(columns=['weights', 'MRO_weights'])
            findfs[r].to_csv(newdatname + 'df_' + 'K' + str(K) + 'R' + str(r + r_start) + '.csv')

        # Final files are safely in newdatname; remove exactly the intermediate
        # files this run generated -- loose per-(epsilon, seed) running CSVs in
        # foldername, and the per-seed concatenated CSVs + this run's metadata
        # in newfoldername -- leaving both directories, and anything else in
        # them, untouched.
        try:
            for r_input in range(len(list_inds)):
                r, epsnum = list_inds[r_input]
                stale = foldername + str(epsnum) + '_R' + str(r + r_start) + '_df.csv'
                if os.path.exists(stale):
                    os.remove(stale)
            generated_files = [
                newfoldername + 'df_' + str(r + r_start) + '.csv'
                for r in range(R)
            ] + [
                newfoldername + 'metadata.json',
                newfoldername + 'metadata.txt',
            ]
            remove_files(generated_files)
        except OSError as e:
            print(f"Warning: cleanup of intermediate files failed: {e}", file=output_stream)

        print("DONE")
        return

    # ---- method == 'dro' ----
    newfoldername = foldername + 'R' + str(R) + '_T' + str(T - 1) + '/'
    os.makedirs(newfoldername, exist_ok=True)
    print(newfoldername)

    if solver == 'subgrad':
        eta = get('eta')
        solve_interval = get('solve_interval')
        metadata = {
            'filename': os.path.basename(__file__),
            'dataset': dataset, 'n_total': n_total,
            'train_size': train_size, 'test_size': test_size,
            'T': T, 'R': R, 'm': m, 'k': k,
            'interval': interval, 'N_init': N_init,
            'r_start': r_start,
            'power': power,
            'p': p,
            'eta': eta,
            'solve_interval': solve_interval,
            'epsilon_values': [float(e) for e in eps_init],
            'num_epsilon_values': len(eps_init),
            'num_random_seeds': R,
            'total_test_combinations': len(eps_init) * R,
        }
    else:
        interval_SAA = get('interval_SAA')
        metadata = {
            'filename': os.path.basename(__file__),
            'dataset': dataset, 'n_total': n_total,
            'train_size': train_size, 'test_size': test_size,
            'T': T, 'R': R, 'm': m, 'k': k,
            'interval': interval, 'interval_SAA': interval_SAA, 'N_init': N_init,
            'r_start': r_start,
            'power': power,
            'epsilon_values': [float(e) for e in eps_init],
            'num_epsilon_values': len(eps_init),
            'num_random_seeds': R,
            'total_test_combinations': len(eps_init) * R,
            'p': p,
        }
    save_run_metadata(metadata, [newfoldername, newdatname])

    if solver == 'subgrad':
        results = Parallel(n_jobs=njobs)(delayed(task_dro_subgrad)(
            r_input, T, N_init, full_data, power, r_start, p,
            m=m, train_size=train_size, test_size=test_size,
            eps_init=eps_init, list_inds=list_inds, init_ind=init_ind,
            k=k, eta=eta, solve_interval=solve_interval,
            t_solve_list=t_solve_list, interval=interval, t_list=t_list,
            newfoldername=newfoldername)
            for r_input in range(len(list_inds)))
    else:
        results = Parallel(n_jobs=njobs)(delayed(task_dro_exact)(
            r_input, T, N_init, full_data, power, r_start, p,
            m=m, train_size=train_size, test_size=test_size,
            eps_init=eps_init, list_inds=list_inds, init_ind=init_ind,
            k=k, interval=interval, interval_SAA=interval_SAA, t_list=t_list,
            newfoldername=newfoldername, checkpoint_every=checkpoint_every)
            for r_input in range(len(list_inds)))

    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r, epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in eps_indices], ignore_index=True)
        findfs[r].to_csv(newfoldername + 'DRO_df_' + str(r + r_start) + '.csv')

    for r in range(R):
        findfs[r].to_csv(newdatname + 'df_' + 'K' + str(0) + 'R' + str(r + r_start) + '.csv')

    # Final files are safely in newdatname; remove exactly the intermediate
    # files this run generated in newfoldername (per (r, epsnum) checkpoint
    # CSVs, per-seed concatenated CSVs, and this run's metadata), leaving the
    # directory -- and anything else in it -- untouched.
    try:
        generated_files = [
            newfoldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv'
            for r, epsnum in list_inds
        ] + [
            newfoldername + 'DRO_df_' + str(r + r_start) + '.csv'
            for r in range(R)
        ] + [
            newfoldername + 'metadata.json',
            newfoldername + 'metadata.txt',
        ]
        remove_files(generated_files)
    except OSError as e:
        print(f"Warning: cleanup of intermediate files failed: {e}", file=output_stream)

    print("DONE")


if __name__ == '__main__':
    main()
