"""Entry point for the portfolio box-support W1-DRO experiments.

Consolidates the ``__main__`` blocks of the legacy drivers
(port_box/port_box.py, port_box_orig.py, port_box_DRO.py, port_box_DRO_orig.py,
port_box_true_saa.py): data generation (seed 12345), the epsilon sweep, the
joblib Parallel over R x M (seed, eps) tasks, per-seed concat + final CSV
writes, intermediate-file cleanup, and run metadata.

Usage (from paper_experiments/):
    uv run python -m portfolio.run --method {mro,dro,true_saa} --solver {exact,subgrad} \
        --results_dir DIR [--T 2001] [--R 10] [--K 15] [--eps_index I] ...

Final per-seed CSVs land in {results_dir}/T{T-1}/df_K{K}R{r}.csv (K=0 for the
DRO paths); note the T-directory has NO R suffix (legacy used T{T-1}R{R}/) so
later seed-extension runs land in the same directory.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .methods import (
    run_true_saa,
    task_dro_exact,
    task_dro_subgrad,
    task_mro_exact,
    task_mro_subgrad,
)
from .utils import generate_returns, get_n_processes, remove_files, save_run_metadata

output_stream = sys.stdout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['mro', 'dro', 'true_saa'])
    parser.add_argument('--solver', type=str, default=None, choices=['exact', 'subgrad'])
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--R', type=int, default=None)
    parser.add_argument('--r_start', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--interval', type=int, default=None)
    parser.add_argument('--interval_SAA', type=int, default=None)
    parser.add_argument('--eps_list', type=float, nargs='+', default=None,
                        help='override the config epsilon grid (descending); '
                             'use a separate --results_dir')
    parser.add_argument('--eps_index', type=int, default=None,
                        help='Run only this single epsilon from the sweep (default: full sweep).')
    parser.add_argument('--checkpoint_every', type=int, default=200,
                        help='Write intermediate checkpoint CSVs every this many logged steps.')
    parser.add_argument('--m', type=int, default=None)
    parser.add_argument('--Q', type=int, default=None)
    parser.add_argument('--N_init', type=int, default=None)
    parser.add_argument('--fixed_time', type=int, default=None)
    parser.add_argument('--rmse_mult', type=float, default=None)
    parser.add_argument('--cluster_interval', type=int, default=None)
    parser.add_argument('--eta_0', type=float, default=None)
    parser.add_argument('--solve_interval', type=int, default=None,
                        help='subgrad paths only: every this many steps (plus the '
                             'early steps in config.T_SOLVE_LIST) an exact solve '
                             're-anchors the subgradient iterate; 0 disables '
                             '(legacy pure-subgradient behavior).')
    parser.add_argument('--line_search', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--power', type=float, default=None)
    parser.add_argument('--box_q', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--n_total', type=int, default=None)
    parser.add_argument('--N_true', type=int, default=None)
    return parser.parse_args()


def resolve(args, key):
    """CLI value if given, else per-path default, else common default."""
    cfg = dict(config.COMMON)
    cfg.update(config.DEFAULTS[key])
    for name in vars(args):
        val = getattr(args, name)
        if val is not None:
            cfg[name] = val
    return cfg


def main():
    args = parse_args()
    method = args.method
    solver = args.solver
    if method == 'true_saa':
        key = ('true_saa', None)
    else:
        if solver is None:
            raise SystemExit('--solver is required for --method mro/dro')
        key = (method, solver)
    cfg = resolve(args, key)

    results_dir = args.results_dir
    if not results_dir.endswith('/'):
        results_dir = results_dir + '/'
    os.makedirs(results_dir, exist_ok=True)

    m = cfg['m']
    n_total = cfg['n_total']
    box_q = cfg['box_q']
    power = cfg['power']
    r_start = cfg['r_start']
    R = cfg['R']
    checkpoint_every = cfg['checkpoint_every']

    # Synthetic bounded returns; [lb, ub] is the per-asset L2 support box.
    synthetic_returns, lb, ub = generate_returns(n_total, m, seed=cfg['seed'], box_q=box_q)
    test_size = cfg['test_size']
    train_size = min(n_total - test_size, 19000)
    init_ind = 0
    njobs = get_n_processes(100)

    # ----------------------------------------------------------------- #
    # true SAA (serial over seeds; mirrors port_box_true_saa.py __main__)
    # ----------------------------------------------------------------- #
    if method == 'true_saa':
        alpha = cfg['alpha']
        a = -1.0 / (1.0 - alpha)
        N_true = min(cfg['N_true'], train_size)

        print(f"m={m}  alpha={alpha}  a={a:.4f}")
        print(f"n_total={n_total}  train_size={train_size}  test_size={test_size}  N_true={N_true}")
        print(f"Output: {results_dir}")
        print()

        records = []
        for r in range(R):
            print(f"Seed r={r}/{R - 1} ...")
            rec = run_true_saa(synthetic_returns, m, N_true, a, r, r_start,
                               test_size, train_size)
            records.append(rec)
            pd.DataFrame([rec]).to_csv(
                results_dir + f'true_saa_R{r}.csv', index=False)

        df_all = pd.DataFrame(records)
        df_all.to_csv(results_dir + 'true_saa_all.csv', index=False)

        # Summary: mean and 25/75 quantiles across seeds — used by the pct-diff plot.
        valid = df_all.dropna(subset=['insample_obj'])
        summary = {
            'm': m,
            'alpha': alpha,
            'N_true': N_true,
            'R': R,
            # in-sample objective
            'insample_obj_mean': valid['insample_obj'].mean(),
            'insample_obj_q25':  valid['insample_obj'].quantile(0.25),
            'insample_obj_q75':  valid['insample_obj'].quantile(0.75),
            # out-of-sample window j=0
            'outsample_eval0_mean': valid['outsample_eval0'].mean(),
            'outsample_eval0_q25':  valid['outsample_eval0'].quantile(0.25),
            'outsample_eval0_q75':  valid['outsample_eval0'].quantile(0.75),
            # out-of-sample window j=1 (matches eval1 / eval2 columns in experiment CSVs)
            'outsample_eval1_mean': valid['outsample_eval1'].mean(),
            'outsample_eval1_q25':  valid['outsample_eval1'].quantile(0.25),
            'outsample_eval1_q75':  valid['outsample_eval1'].quantile(0.75),
            # timing
            'solve_time_mean': df_all['solve_time'].mean(),
            'n_converged': int(valid.shape[0]),
        }
        pd.DataFrame([summary]).to_csv(
            results_dir + 'true_saa_summary.csv', index=False)

        print()
        print("=== Summary ===")
        for key_, val in summary.items():
            fmt = f'{val:.6f}' if isinstance(val, float) else str(val)
            print(f"  {key_}: {fmt}")
        print(f"\nSaved to {results_dir}")
        return

    # ----------------------------------------------------------------- #
    # Shared sweep setup (eps list, forced-checkpoint t_list, (r, eps) grid)
    # ----------------------------------------------------------------- #
    T = cfg['T']
    interval = cfg['interval']
    N_init = cfg['N_init']

    if args.eps_list is not None:
        # Explicit epsilon grid override (grid-edge extension runs); write
        # to a SEPARATE results_dir so main-sweep block indices stay stable.
        eps_init = list(args.eps_list)
    else:
        eps_init = list(config.EPS[key])
        if key == ('mro', 'exact') and T >= 5000:
            eps_init = list(config.EPS_MRO_EXACT_LONG_T)
    if args.eps_index is not None:
        eps_init = [eps_init[args.eps_index]]
    M = len(eps_init)
    list_inds = list(itertools.product(np.arange(R), np.arange(M)))
    t_list = list(config.T_LIST[key])

    newdatname = results_dir + 'T' + str(T - 1) + '/'

    # ----------------------------------------------------------------- #
    # MRO paths (mirrors port_box.py / port_box_orig.py __main__)
    # ----------------------------------------------------------------- #
    if method == 'mro':
        K = cfg['K']
        Q = cfg['Q']
        fixed_time = cfg['fixed_time']
        rmse_mult = cfg['rmse_mult']
        cluster_interval = cfg['cluster_interval']

        newfoldername = results_dir + 'K'+str(K)+'_R'+str(R)+'_T'+str(T-1)+'/'
        os.makedirs(newfoldername, exist_ok=True)
        print(newfoldername)

        if solver == 'subgrad':
            eta_0 = cfg['eta_0']
            line_search = cfg['line_search']
            solve_interval = cfg['solve_interval']
            t_solve_list = list(config.T_SOLVE_LIST[key])
            save_run_metadata(
                {
                    'filename': os.path.basename(__file__),
                    'K': K, 'T': T, 'R': R, 'm': m, 'Q': Q,
                    'fixed_time': fixed_time, 'interval': interval, 'N_init': N_init,
                    'rmse_mult': rmse_mult, 'cluster_interval': cluster_interval,
                    'r_start': r_start, 'line_search': line_search,
                    'eta_0': eta_0, 'power': power,
                    'solve_interval': solve_interval,
                    'epsilon_values': [float(e) for e in eps_init],
                    'num_epsilon_values': len(eps_init),
                    'num_random_seeds': R,
                    'total_test_combinations': len(eps_init) * R,
                    'box_q': box_q
                },
                [newfoldername, (newdatname, f'metadata_K{K}.json')],
            )
            results = Parallel(n_jobs=njobs)(delayed(task_mro_subgrad)(
                r_input, K, T, N_init, synthetic_returns, lb, ub, eta_0, r_start,
                newfoldername, power, list_inds, eps_init, m, train_size, test_size,
                init_ind, Q, interval, t_list, fixed_time, rmse_mult, cluster_interval,
                line_search, checkpoint_every, solve_interval=solve_interval,
                t_solve_list=t_solve_list) for r_input in range(len(list_inds)))
        else:
            save_run_metadata(
                {
                    'filename': os.path.basename(__file__),
                    'K': K, 'T': T, 'R': R, 'm': m, 'Q': Q,
                    'fixed_time': fixed_time, 'interval': interval, 'N_init': N_init,
                    'rmse_mult': rmse_mult, 'cluster_interval': cluster_interval,
                    'r_start': r_start, 'power': power,
                    'epsilon_values': [float(e) for e in eps_init],
                    'num_epsilon_values': len(eps_init),
                    'num_random_seeds': R,
                    'total_test_combinations': len(eps_init) * R,
                    'box_q': box_q
                },
                [newfoldername, (newdatname, f'metadata_K{K}.json')],
            )
            results = Parallel(n_jobs=njobs)(delayed(task_mro_exact)(
                r_input, K, T, N_init, synthetic_returns, lb, ub, r_start, power,
                list_inds, eps_init, m, train_size, test_size, init_ind, Q, interval,
                t_list, fixed_time, rmse_mult, cluster_interval, results_dir,
                checkpoint_every) for r_input in range(len(list_inds)))

        dfs = {}
        for r in range(R):
            dfs[r] = {}
        for r_input in range(len(list_inds)):
            r, epsnum = list_inds[r_input]
            dfs[r][epsnum] = results[r_input]

        findfs = {}
        for r in range(R):
            findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))], ignore_index=True)
            findfs[r].to_csv(newfoldername + 'df_' + str(r+r_start) + '.csv')

        weight_cols = ['weights', 'MRO_weights'] if solver == 'subgrad' \
            else ['weights', 'MRO_weights', 'SAA_weights']
        for r in range(R):
            findfs[r] = findfs[r].drop(columns=weight_cols)
            findfs[r].to_csv(newdatname + 'df_' + 'K'+str(K)+'R' + str(r+r_start) + '.csv')

        # Final files are safely in newdatname; remove exactly the intermediate
        # files this run generated, leaving the directories -- and anything else
        # in them, e.g. a concurrent run's own output -- untouched.
        try:
            if solver == 'subgrad':
                generated_files = [
                    newfoldername + 'df_' + str(r_start + r_input) + '.csv'
                    for r_input in range(len(list_inds))
                ]
            else:
                generated_files = [
                    results_dir + str(epsnum) + '_R' + str(r + r_start) + '_df.csv'
                    for r, epsnum in list_inds
                ]
            generated_files += [
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

    # ----------------------------------------------------------------- #
    # DRO paths (mirrors port_box_DRO.py / port_box_DRO_orig.py __main__)
    # ----------------------------------------------------------------- #
    alpha = cfg['alpha']
    foldername = results_dir + 'R' + str(R) + '_T' + str(T - 1) + '/'
    os.makedirs(foldername, exist_ok=True)
    print(foldername)
    dro_newdatname = foldername + 'T' + str(T - 1) + '/'

    if solver == 'subgrad':
        eta_0 = cfg['eta_0']
        line_search = cfg['line_search']
        solve_interval = cfg['solve_interval']
        t_solve_list = list(config.T_SOLVE_LIST[key])
        save_run_metadata(
            {
                'filename': os.path.basename(__file__),
                'problem': 'box-support W1-DRO portfolio CVaR (L2 ground norm, Section 9.5), DRO only',
                'T': T, 'R': R, 'm': m, 'alpha': alpha,
                'interval': interval, 'N_init': N_init,
                'r_start': r_start, 'line_search': line_search, 'eta_0': eta_0, 'power': power,
                'solve_interval': solve_interval,
                'epsilon_values': [float(e) for e in eps_init],
                'num_epsilon_values': len(eps_init), 'num_random_seeds': R,
                'total_test_combinations': len(eps_init) * R,
                'box_lb': [float(v) for v in lb], 'box_ub': [float(v) for v in ub],
                'box_q': box_q
            },
            [foldername, dro_newdatname],
        )
        results = Parallel(n_jobs=njobs)(delayed(task_dro_subgrad)(
            r_input, T, N_init, synthetic_returns, lb, ub, eta_0, r_start, power,
            list_inds, eps_init, alpha, m, train_size, test_size, init_ind,
            interval, t_list, line_search, foldername,
            checkpoint_every, solve_interval=solve_interval,
            t_solve_list=t_solve_list) for r_input in range(len(list_inds)))
    else:
        interval_SAA = cfg['interval_SAA']
        save_run_metadata(
            {
                'filename': os.path.basename(__file__),
                'problem': 'box-support W1-DRO portfolio CVaR (L2 ground norm) -- exact SOCP solve',
                'T': T, 'R': R, 'm': m, 'alpha': alpha,
                'interval': interval, 'interval_SAA': interval_SAA, 'N_init': N_init,
                'r_start': r_start, 'power': power,
                'epsilon_values': [float(e) for e in eps_init],
                'num_epsilon_values': len(eps_init), 'num_random_seeds': R,
                'total_test_combinations': len(eps_init) * R,
                'box_lb': [float(v) for v in lb], 'box_ub': [float(v) for v in ub],
                'box_q': box_q
            },
            [foldername, dro_newdatname],
        )
        results = Parallel(n_jobs=njobs)(delayed(task_dro_exact)(
            r_input, T, N_init, synthetic_returns, lb, ub, r_start, power,
            list_inds, eps_init, alpha, m, train_size, test_size, init_ind,
            interval, interval_SAA, t_list, foldername,
            checkpoint_every) for r_input in range(len(list_inds)))

    dfs = {}
    for r in range(R):
        dfs[r] = {}
    for r_input in range(len(list_inds)):
        r, epsnum = list_inds[r_input]
        dfs[r][epsnum] = results[r_input]

    perseed_prefix = 'DRO_new_df_' if solver == 'subgrad' else 'DRO_df_'
    findfs = {}
    for r in range(R):
        findfs[r] = pd.concat([dfs[r][i] for i in range(len(eps_init))], ignore_index=True)
        findfs[r].to_csv(foldername + perseed_prefix + str(r + r_start) + '.csv')

    # Final concatenated outputs -> shared 'T{T-1}/' folder (K=0 marks DRO rows).
    final_datname = results_dir + 'T' + str(T - 1) + '/'
    os.makedirs(final_datname, exist_ok=True)
    for r in range(R):
        findfs[r].to_csv(final_datname + 'df_' + 'K' + str(0) + 'R' + str(r + r_start) + '.csv')

    # Final files are safely in final_datname; remove exactly the intermediate
    # files this run generated in foldername (per (r, epsnum) checkpoint CSVs,
    # per-seed concatenated CSVs, and this run's metadata), leaving the
    # directory -- and anything else in it, e.g. a concurrent run's own
    # output -- untouched.
    try:
        generated_files = [
            foldername + 'DRO' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv'
            for r, epsnum in list_inds
        ]
        if solver == 'subgrad':
            generated_files += [
                foldername + 'DRO_new' + str(epsnum) + '_R' + str(r + r_start) + '_df.csv'
                for r, epsnum in list_inds
            ]
        generated_files += [
            foldername + perseed_prefix + str(r + r_start) + '.csv'
            for r in range(R)
        ] + [
            foldername + 'metadata.json',
            foldername + 'metadata.txt',
        ]
        remove_files(generated_files)
    except OSError as e:
        print(f"Warning: cleanup of intermediate files failed: {e}", file=output_stream)

    print("DONE")


if __name__ == '__main__':
    main()
