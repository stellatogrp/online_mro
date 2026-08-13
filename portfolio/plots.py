"""Portfolio (CVaR) plotting driver.

Port of ``port_box/plots_w1.py`` to the paper_experiments layout.  The
computation drivers write per-seed results as

    {results_dir}/{method}_{solver}/T{T-1}/df_K{K}R{r}.csv

with method_solver in {mro_subgrad, mro_exact, dro_subgrad, dro_exact}
(plus a ``true_saa`` folder with ``true_saa_summary.csv``).

Layout mapping from the legacy plots_w1.py paths:

    legacy results/orig/p2/12/T{T-1}R{R}/  (K=Kval rows) -> {results_dir}/mro_exact/T{T-1}/
    legacy results/orig/p2/12/T{T-1}R{R}/  (K=0 rows)    -> {results_dir}/dro_exact/T{T-1}/
    legacy results/grad/p2/12/T{T-1}R{R}/  (K=Kval rows) -> {results_dir}/mro_subgrad/T{T-1}/
    legacy results/grad/p2/12/T{T-1}R{R}/  (K=0 rows)    -> {results_dir}/dro_subgrad/T{T-1}/
    legacy results/true_saa/true_saa_summary.csv         -> {results_dir}/true_saa/true_saa_summary.csv

i.e. orig -> *_exact, grad -> *_subgrad, DRO data (K=0) comes from the dro_*
runs, and the legacy run-id path segments (p2/12) and the R{R} suffix
disappear (seeds are discovered by glob in ``setup_dfs``).

Usage (from paper_experiments/):
    uv run python -m portfolio.plots --results_dir <results> --out_dir <plots>
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plotting.paper_style import set_paper_style
from plotting.plots_utils import (
    plot_eval_all, plot_eval_all_compare, plot_eval_all_compare_eval,
    plot_eval_all_compare_eval_pct, plot_confidence_regret,
    plot_eval, plot_eval_compare, plot_regret_new, plot_cumulative_time,
    plot_combined, setup_dfs, infer_end_ind, select_eps_indices, merge_eps_extension,
)

# WARNING: duplicated from portfolio/config.py (which may define the same eps
# grids) to keep this driver importable while the computation modules are
# still in flux.  The index tuples (inda, indb, indd) below index into these
# lists; keep them in sync with config.py.  TODO: import from config.py once
# the layout settles.
EPS_INIT = [0.004, 0.003, 0.002, 0.001, 0.0005, 0.0001, 0.00001]
EPS_DRO = [0.0035, 0.00325, 0.003, 0.0025]

QUANT_LIST = [25, 75]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results_dir", required=True,
                        help="root of computation results "
                             "({method}_{solver}/T{T-1}/df_K{K}R{r}.csv)")
    parser.add_argument("--out_dir", required=True,
                        help="where aggregated CSVs and PDFs are written")
    parser.add_argument("--T", type=int, default=2001,
                        help="horizon; results live under T{T-1}/")
    parser.add_argument("--K", type=int, default=15,
                        help="cluster count for the MRO runs (Kval)")
    parser.add_argument("--init", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="--init re-aggregates from per-seed CSVs; "
                             "--no-init reads cached aggregates")
    parser.add_argument("--tune_eps", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="--tune_eps (default) picks, per curve, the "
                             "epsilon block minimizing the late-horizon mean "
                             "out-of-sample value; --no-tune_eps restores the "
                             "legacy hard-coded indices")
    args = parser.parse_args()

    set_paper_style()

    Kval = args.K
    tdir = "T" + str(args.T - 1)

    def rdir(variant):
        # trailing separator: plots_utils joins paths by string concatenation
        return os.path.join(args.results_dir, variant, tdir) + os.sep

    def odir(variant):
        d = os.path.join(args.out_dir, variant, tdir) + os.sep
        os.makedirs(d, exist_ok=True)
        return d

    # Combined PDFs go directly under out_dir/T{T-1}/.
    folderout = os.path.join(args.out_dir, tdir) + os.sep
    os.makedirs(folderout, exist_ok=True)

    df_orig, quantiles_orig = setup_dfs(
        foldername=rdir("mro_exact"), folderout=odir("mro_exact"),
        K_list=[Kval], quant_list=QUANT_LIST, init=args.init)
    df_new, quantiles_new = setup_dfs(
        foldername=rdir("mro_subgrad"), folderout=odir("mro_subgrad"),
        K_list=[Kval], quant_list=QUANT_LIST, init=args.init)
    df_orig_dro, quantiles_orig_dro = setup_dfs(
        foldername=rdir("dro_exact"), folderout=odir("dro_exact"),
        K_list=[0], quant_list=QUANT_LIST, init=args.init)
    df_new_dro, quantiles_new_dro = setup_dfs(
        foldername=rdir("dro_subgrad"), folderout=odir("dro_subgrad"),
        K_list=[0], quant_list=QUANT_LIST, init=args.init)

    # Epsilon-extension runs (results_epsx trees) append extra blocks so the
    # tuning selector sees one contiguous grid; silent no-op when absent.
    def _epsx(variant):
        return os.path.join(args.results_dir + "_epsx", variant, tdir) + os.sep

    def _epsx_out(variant):
        d = os.path.join(args.out_dir, variant + "_epsx", tdir) + os.sep
        os.makedirs(d, exist_ok=True)
        return d

    df_orig, quantiles_orig = merge_eps_extension(
        df_orig, quantiles_orig, _epsx("mro_exact"), _epsx_out("mro_exact"), Kval)
    df_new, quantiles_new = merge_eps_extension(
        df_new, quantiles_new, _epsx("mro_subgrad"), _epsx_out("mro_subgrad"), Kval)
    df_orig_dro, quantiles_orig_dro = merge_eps_extension(
        df_orig_dro, quantiles_orig_dro, _epsx("dro_exact"), _epsx_out("dro_exact"), 0)
    df_new_dro, quantiles_new_dro = merge_eps_extension(
        df_new_dro, quantiles_new_dro, _epsx("dro_subgrad"), _epsx_out("dro_subgrad"), 0)
    # Second extension batch (separate tree so batches never overwrite
    # each other's per-seed files).
    def _epsx2(variant):
        return os.path.join(args.results_dir + "_epsx2", variant, tdir) + os.sep

    def _epsx2_out(variant):
        d = os.path.join(args.out_dir, variant + "_epsx2", tdir) + os.sep
        os.makedirs(d, exist_ok=True)
        return d

    df_orig, quantiles_orig = merge_eps_extension(
        df_orig, quantiles_orig, _epsx2("mro_exact"), _epsx2_out("mro_exact"), Kval)
    df_new, quantiles_new = merge_eps_extension(
        df_new, quantiles_new, _epsx2("mro_subgrad"), _epsx2_out("mro_subgrad"), Kval)
    df_orig_dro, quantiles_orig_dro = merge_eps_extension(
        df_orig_dro, quantiles_orig_dro, _epsx2("dro_exact"), _epsx2_out("dro_exact"), 0)
    df_new_dro, quantiles_new_dro = merge_eps_extension(
        df_new_dro, quantiles_new_dro, _epsx2("dro_subgrad"), _epsx2_out("dro_subgrad"), 0)

    end_ind_orig = infer_end_ind(df_orig, K=Kval)
    end_ind_orig_dro = infer_end_ind(df_orig_dro, K=0)
    end_ind_new = infer_end_ind(df_new, K=Kval)
    end_ind_new_dro = infer_end_ind(df_new_dro, K=0)

    # (inda, indb, indd) index the epsilon blocks stacked in each aggregated
    # frame: inda -> mro_exact block for online clustering, indb -> mro_exact
    # block for reclustering, indd -> dro_exact block for DRO.  The legacy
    # values below were carried over verbatim from port_box/plots_w1.py; by
    # default they are replaced by per-curve tuning (argmin of the
    # late-horizon mean out-of-sample value, see select_eps_indices), with
    # --no-tune_eps restoring them.
    inda_legacy, indb_legacy, indd_legacy = 2, 5, 2
    j_grad_legacy = (2, 2, 3)
    j_grad_pct_legacy = (1, 2, 3)  # legacy quirk: pct plot used j1g=1

    # SAA is only solved for epsnum==0 (SA_eval2 is NaN outside block 0 in
    # the dro_exact frames -- verified on all current result sets), so it
    # stays pinned at block 0 inside plots_utils and is not tuned here.
    # Limitation: cluster SAA's columns live in the mro frames and every
    # panel slices them with the online index j1, so it follows the online
    # tuning rather than getting its own argmin.
    if args.tune_eps:
        def _tune(curve, df_dict, K, end_ind, col, legacy, conf_col=None):
            idx, eps, table = select_eps_indices(df_dict, K, end_ind, col,
                                                 fallback_index=legacy,
                                                 conf_col=conf_col)
            if not table:
                print(f"  {curve:22s} frame/column missing -> "
                      f"keeping legacy block {legacy}")
                return idx
            row = dict((r[0], r) for r in table)[idx]
            cand = "  ".join(f"{i}:{lm:.4g}/c{cf:.2f}"
                             for i, _, lm, cf in table)
            print(f"  {curve:22s} block {idx}  eps={eps:.4g}  "
                  f"late OOS={row[2]:.6g}  conf={row[3]:.2f}  "
                  f"(legacy {legacy})  [blocks OOS/conf: {cand}]")
            return idx

        print(f"Tuned epsilon blocks (portfolio, {tdir}): per curve, argmin "
              "of the mean OOS value over the last ~10% of the block")
        inda = _tune("online (exact)", df_orig, Kval, end_ind_orig,
                     'O_eval1', inda_legacy, conf_col='O_satisfy0')
        indb = _tune("reclustering (exact)", df_orig, Kval, end_ind_orig,
                     'MRO_eval1', indb_legacy, conf_col='MRO_satisfy0')
        indd = _tune("DRO (exact)", df_orig_dro, 0, end_ind_orig_dro,
                     'DRO_eval2', indd_legacy, conf_col='DRO_satisfy1')
        j1g = _tune("online subgrad", df_new, Kval, end_ind_new,
                    'O_eval1', j_grad_legacy[0], conf_col='O_satisfy0')
        j4g = _tune("reclustering subgrad", df_new, Kval, end_ind_new,
                    'MRO_eval1', j_grad_legacy[1], conf_col='MRO_satisfy0')
        j3g = _tune("DRO subgrad", df_new_dro, 0, end_ind_new_dro,
                    'DRO_eval2', j_grad_legacy[2], conf_col='DRO_satisfy1')
        j_grad = (j1g, j4g, j3g)
        j_grad_pct = j_grad
    else:
        inda, indb, indd = inda_legacy, indb_legacy, indd_legacy
        j_grad = j_grad_legacy
        j_grad_pct = j_grad_pct_legacy

    ylim = [-0.04, 0.01]
    series = ["online", "online sgd", "DRO", "DRO sgd", "SAA", "cluster SAA"]

    # True-SAA oracle (N=20000 samples): J* dotted reference line in the
    # out-of-sample panels + reference for the pct-diff plot.  Degrades
    # gracefully (jstar=None, pct plot skipped) when the summary is absent.
    true_saa_summary_path = os.path.join(args.results_dir, "true_saa",
                                         "true_saa_summary.csv")
    true_ref = None
    jstar = None
    if os.path.exists(true_saa_summary_path):
        true_ref = pd.read_csv(true_saa_summary_path)
        if "outsample_eval1_mean" in true_ref.columns and len(true_ref):
            jstar = float(true_ref["outsample_eval1_mean"].iloc[0])
            print(f"J* (true SAA, mean outsample_eval1): {jstar:.6g}")
    else:
        print(f"true-SAA summary not found at {true_saa_summary_path}; "
              "no J* line, pct-diff plot skipped")

    plot_eval_all(df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
                  j=(inda, indb, indd), K=Kval, q=(25, 75), ylim=ylim,
                  legend=True, val2=2.3, end_ind=end_ind_orig,
                  end_ind_dro=end_ind_orig_dro, folderout=folderout)

    plot_eval(df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
              j=(inda, indb, indd), K=Kval, q=(25, 75), end_ind=end_ind_orig,
              end_ind_dro=end_ind_orig_dro, ylim=ylim, legend=True,
              jstar=jstar, folderout=folderout)

    plot_eval_all_compare(
        df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
        df_new, quantiles_new, df_new_dro, quantiles_new_dro,
        j=(inda, indb, indd), j_grad=j_grad, K=Kval, q=(25, 75), ylim=ylim,
        end_ind=end_ind_orig, end_ind_dro=end_ind_orig_dro,
        end_ind_grad=end_ind_new, end_ind_dro_grad=end_ind_new_dro,
        val2=2.3, legend=True,
        folderout=folderout, series=series,
    )

    plot_eval_compare(
        df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
        df_new, quantiles_new, df_new_dro, quantiles_new_dro,
        j=(inda, indb, indd), j_grad=j_grad, K=Kval, q=(25, 75), ylim=ylim,
        end_ind=end_ind_orig, end_ind_dro=end_ind_orig_dro,
        end_ind_grad=end_ind_new, end_ind_dro_grad=end_ind_new_dro,
        legend=True, jstar=jstar,
        folderout=folderout, series=series,
    )

    plot_eval_all_compare_eval(
        df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
        df_new, quantiles_new, df_new_dro, quantiles_new_dro,
        j=(inda, indb, indd), j_grad=j_grad, K=Kval, q=(25, 75), ylim=ylim,
        end_ind=end_ind_orig, end_ind_dro=end_ind_orig_dro,
        end_ind_grad=end_ind_new, end_ind_dro_grad=end_ind_new_dro,
        val2=2.3, legend=True, jstar=jstar,
        folderout=folderout, series=series, legend_ncol=6,
        trim_start=1, trim_end=-2, trim_start_grad=2, trim_end_grad=-8,
        resolve_t_list=None,
    )

    plot_regret_new(df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
                    j=(inda, indb, indd), K=Kval, q=(25, 75),
                    end_ind=end_ind_orig, end_ind_dro=end_ind_orig_dro,
                    ylim=[0.0001, 3], folderout=folderout)

    # LHS: confidence panel (plot_eval_all_compare_eval-style: full+subgrad
    # overlay, series filter, trims); RHS: plot_regret_new's regret panel.
    plot_confidence_regret(
        df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
        df_new, quantiles_new, df_new_dro, quantiles_new_dro,
        j=(inda, indb, indd), j_grad=j_grad, K=Kval, q=(25, 75),
        end_ind=end_ind_orig, end_ind_dro=end_ind_orig_dro,
        end_ind_grad=end_ind_new, end_ind_dro_grad=end_ind_new_dro,
        val2=2.1, legend=True,
        folderout=folderout, series=series, legend_ncol=6,
        trim_start=1, trim_end=-2, trim_start_grad=2, trim_end_grad=-8,
        resolve_t_list=None,
        trim_start_regret=1,
        ylim_regret=[0.0001, 3],
    )

    # Cumulative computation time of the online variants: clustered (K
    # centroids) vs full running sample -- the data-compression benefit.
    plot_cumulative_time(df_new, df_new_dro, j_grad=j_grad, K=Kval,
                         end_ind_grad=end_ind_new,
                         end_ind_dro_grad=end_ind_new_dro,
                         folderout=folderout)

    # Single 2x3 figure consolidating the main comparison: in-sample /
    # out-of-sample / confidence / per-iteration time (exact) / cumulative
    # time (online) / dynamic regret.  Same frames and args as the
    # plot_eval_all_compare_eval + plot_confidence_regret calls above.
    plot_combined(
        df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
        df_new, quantiles_new, df_new_dro, quantiles_new_dro,
        j=(inda, indb, indd), j_grad=j_grad, K=Kval, q=(25, 75), ylim=ylim,
        end_ind=end_ind_orig, end_ind_dro=end_ind_orig_dro,
        end_ind_grad=end_ind_new, end_ind_dro_grad=end_ind_new_dro,
        legend=True, jstar=jstar,
        folderout=folderout, series=series,
        trim_start=1, trim_end=-2, trim_start_grad=2, trim_end_grad=-8,
        trim_start_regret=1, ylim_regret=[0.0001, 3],
    )

    # Absolute-percentage-difference plot vs. the N=20000 SAA oracle
    # (true_ref read above).  Run the true-SAA driver first to generate
    # true_saa_summary.csv.
    if true_ref is not None:
        plot_eval_all_compare_eval_pct(
            df_orig, quantiles_orig, df_orig_dro, quantiles_orig_dro,
            df_new, quantiles_new, df_new_dro, quantiles_new_dro,
            true_ref=true_ref,
            j=(inda, indb, indd), j_grad=j_grad_pct, K=Kval, q=(25, 75),
            end_ind=end_ind_orig, end_ind_dro=end_ind_orig_dro,
            end_ind_grad=end_ind_new, end_ind_dro_grad=end_ind_new_dro,
            val2=2.3, legend=True,
            folderout=folderout, series=series, legend_ncol=6,
            trim_start=1, trim_end=-2, trim_start_grad=2, trim_end_grad=-8,
            yscale_log=False, resolve_t_list=None,
        )
    else:
        print(f"Skipping pct-diff plot: {true_saa_summary_path} not found. "
              "Run the true-SAA driver first.")


if __name__ == "__main__":
    main()
