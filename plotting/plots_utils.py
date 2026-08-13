"""Shared plotting helpers for the paper experiments (portfolio + svm).

Trimmed copy of the repo-level ``plots_utils.py``: only the functions the two
plotting drivers (``portfolio/plots.py`` and ``svm/plots.py``) actually call
are kept, plus their internal helpers.  Plotting logic is preserved verbatim
from the legacy module, with one exception: ``setup_dfs`` discovers per-seed
CSVs via a glob over ``df_K{K}R*.csv`` (sorted by seed number) instead of a
fixed ``range(R)`` loop, so extension runs with additional seeds are picked up
automatically.

Domains differ in one respect these functions must be told about explicitly:
the "cluster SAA" (K-means-centroid sample-average baseline) columns are
named ``SAA_*`` in the portfolio experiment but ``cluster_SAA_*`` in svm.
Pass ``saa_prefix='cluster_SAA'`` from the svm driver; the default ``'SAA'``
matches portfolio.

All per-series colors, linestyles, line widths, band alphas, and z-orders come
from the single authoritative ``METHOD_STYLE`` map in
``plotting.paper_style`` -- lines via ``_line(..., style='<key>')`` /
``method_kwargs``, quantile bands via ``_band(..., '<key>', ...)`` /
``band_kwargs``.  Nothing in this module hard-codes a color or linestyle;
restyle a method by editing its METHOD_STYLE entry.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting.paper_style import (
    METHOD_STYLE, method_kwargs as _mkw, band_kwargs as _bkw,
)


def _pick(d, key):
    """Return ``d[key]`` when it exists, otherwise ``None``.

    Lets the plotting routines silently drop a data source -- and every line
    that depends on it -- when its dataframe was never produced (e.g. an
    experiment folder that has not been run yet, so ``setup_dfs`` returned an
    empty dict or one missing this ``K``)."""
    if d is None:
        return None
    if isinstance(d, pd.DataFrame):
        # Already a bare DataFrame (pre-indexed by the caller) rather than a
        # dict keyed by K/0.  Treat it as present.
        return d
    try:
        return d[key] if key in d else None
    except TypeError:
        return None


def _tsource(*dfs, end_ind=61, stride=2, offset=0, plus=0):
    """t-axis from the first available dataframe among ``dfs`` (or None)."""
    for d in dfs:
        if d is not None:
            return np.array(d['t'])[(0*end_ind)+offset:(1)*end_ind:stride] + plus
    return None


def _hc(df, col):
    """True iff ``df`` exists and has column ``col``."""
    return df is not None and col in getattr(df, 'columns', ())


def _jstar_line(ax, jstar):
    """Horizontal dotted reference at the true-SAA oracle value ``jstar``.

    Drawn only when ``jstar`` is not None; returns the Line2D handle (or
    None) so callers can add it to a hand-built legend.  Style comes from
    METHOD_STYLE['true_saa'] with the linestyle forced to dotted so the
    reference reads as an oracle level, not another method trace."""
    if jstar is None:
        return None
    return ax.axhline(jstar, label=r"$H^\star$",
                      **{**_mkw('true_saa'), 'linestyle': ':'})


def select_eps_indices(df_dict, K, end_ind, col, fallback_index=0,
                       n_tail_frac=0.1, conf_col=None, conf_threshold=0.60):
    """Pick the epsilon block for a plotted curve (the paper's tuning rule).

    The aggregated frames stack M epsilon blocks of ``end_ind`` rows each
    vertically; the plotting calls slice out one block via an integer index.
    Late-horizon statistics are means over the last ``ceil(end_ind *
    n_tail_frac)`` rows of each block (~10% tail).

    Rule (PI, 2026-08-10): among blocks whose late-horizon confidence
    (``conf_col``) is at least ``conf_threshold`` (default 0.60), pick the one with the
    LOWEST late-horizon out-of-sample value (``col``).  If no block reaches
    the threshold, pick the block with the HIGHEST confidence.  When
    ``conf_col`` is None/absent the selection falls back to the plain
    argmin of ``col`` (used e.g. for curves without a certificate).

    Returns ``(best_index, best_eps, table)``; ``table`` rows are
    ``(idx, eps, late_oos, late_conf)`` for logging (late_conf is NaN when
    ``conf_col`` is unavailable).  Blocks with NaN late OOS are never
    selected.

    Guard: when the frame or ``col`` is missing (experiment not run yet),
    returns ``(fallback_index, nan, [])`` so callers can keep their legacy
    hard-coded index."""
    dfk = _pick(df_dict, K)
    if dfk is None or col not in getattr(dfk, 'columns', ()) or end_ind <= 0:
        return fallback_index, float('nan'), []
    have_conf = conf_col is not None and conf_col in dfk.columns
    n_blocks = max(1, len(dfk) // end_ind)
    n_tail = max(1, int(np.ceil(end_ind * n_tail_frac)))
    table = []
    for idx in range(n_blocks):
        block = dfk.iloc[idx * end_ind:(idx + 1) * end_ind]
        eps = (float(block['epsilon'].iloc[0])
               if 'epsilon' in block.columns and len(block) else float('nan'))
        late = pd.to_numeric(block[col].iloc[-n_tail:], errors='coerce').mean()
        conf = (pd.to_numeric(block[conf_col].iloc[-n_tail:],
                              errors='coerce').mean()
                if have_conf else float('nan'))
        table.append((idx, eps, float(late), float(conf)))
    finite = [row for row in table if np.isfinite(row[2])]
    if not finite:
        return fallback_index, float('nan'), table
    if have_conf and any(np.isfinite(row[3]) for row in finite):
        eligible = [row for row in finite
                    if np.isfinite(row[3]) and row[3] >= conf_threshold]
        if eligible:
            best = min(eligible, key=lambda row: row[2])
        else:
            best = max(finite, key=lambda row: (row[3] if np.isfinite(row[3])
                                                else -np.inf))
    else:
        best = min(finite, key=lambda row: row[2])
    return best[0], best[1], table


def merge_eps_extension(df_dict, quant_dict, ext_foldername, ext_folderout,
                        K, quant_list=(25, 75)):
    """Append epsilon-extension blocks (results_epsx runs) to the aggregated
    frames so tuning and plotting see one contiguous grid.

    Aggregates the per-seed CSVs under ``ext_foldername`` via ``setup_dfs``
    (cached in ``ext_folderout``) and concatenates the blocks below the main
    frame's blocks (ignore_index, so positional block slicing keeps working;
    the extension runs share T/logging cadence with the main sweep by
    construction).  Missing/empty extension dirs are a silent no-op."""
    import os as _os
    have_raw = ext_foldername and _os.path.isdir(ext_foldername)
    have_cache = ext_folderout and _os.path.isfile(
        _os.path.join(ext_folderout, f'df_K{K}.csv'))
    if not have_raw and not have_cache:
        return df_dict, quant_dict
    # Re-aggregate from raw per-seed CSVs when present; otherwise fall back
    # to the committed aggregate cache (fresh-clone reproducibility).
    ext_df, ext_q = setup_dfs(foldername=ext_foldername,
                              folderout=ext_folderout, K_list=[K],
                              quant_list=list(quant_list), init=have_raw)
    if _pick(ext_df, K) is None or _pick(df_dict, K) is None:
        return df_dict, quant_dict
    df_dict = dict(df_dict)
    df_dict[K] = pd.concat([df_dict[K], ext_df[K]], ignore_index=True)
    # setup_dfs returns quantiles keyed {K: {q: df}} (K outer, q inner).
    out_q = dict(quant_dict)
    if isinstance(quant_dict.get(K), dict) and isinstance(ext_q.get(K), dict):
        out_q[K] = dict(quant_dict[K])
        for q in quant_dict[K]:
            if q in ext_q[K]:
                out_q[K][q] = pd.concat([quant_dict[K][q], ext_q[K][q]],
                                        ignore_index=True)
    return df_dict, out_q


def _trimmed_bounds(tr_trim_pairs):
    """Min/max t actually plotted across ``[(t_range, trim), ...]`` pairs,
    applying each ``trim`` (an ``(a, b)`` slice, or ``None``) the same way
    ``_line``/``_band`` do. Returns ``(None, None)`` if nothing is plottable.

    Used to keep resolve-timestep markers from being drawn past the visible
    (post-trim) extent of the lines they're meant to annotate."""
    mins, maxs = [], []
    for tr, trim in tr_trim_pairs:
        if tr is None or len(tr) == 0:
            continue
        arr = np.asarray(tr)
        if trim is not None:
            a, b = trim
            arr = arr[a:b]
        if len(arr) == 0:
            continue
        mins.append(np.nanmin(arr))
        maxs.append(np.nanmax(arr))
    if not maxs:
        return None, None
    return min(mins), max(maxs)


def _line(ax, df, col, ji, ei, step, tr, *args, offset=0, trim=None, style=None, **kw):
    """Plot ``df[col]`` block on ``ax`` (an Axes or ``plt``) iff the column
    exists and a t-axis is available; return the Line2D handle or ``None``.

    Skips silently when the dataframe is missing OR lacks ``col`` -- so a line
    whose backing column was never written (e.g. ``SA_*`` in a DRO-only CSV)
    simply does not appear.

    ``trim``, if given, is an ``(a, b)`` pair applied as a final ``[a:b]``
    slice to both the t-axis and the y-values -- on top of the ``ji``/``ei``/
    ``step`` block slicing -- e.g. ``trim=(None, -2)`` drops the last 2 points.

    ``style``, if given, is a METHOD_STYLE key: color/linestyle/linewidth/
    zorder (and the series' sparse marker, if any) are looked up there
    (explicit kwargs still win)."""
    if tr is None or not _hc(df, col):
        return None
    x = np.asarray(tr)
    y = np.array(df[col][(ji*ei)+offset:(ji+1)*ei:step])
    if trim is not None:
        a, b = trim
        x, y = x[a:b], y[a:b]
    if len(y) == 0 or len(x) != len(y):
        # Requested epsilon block absent (or shorter than the t-axis) in this
        # dataframe -- e.g. a run that logged fewer eps values than the index
        # ``ji`` assumes.  Degrade gracefully like a missing column instead of
        # letting matplotlib raise on the length mismatch.
        return None
    if style is not None:
        kw = {**_mkw(style), **kw}
        if not np.isfinite(pd.to_numeric(pd.Series(y), errors='coerce')).any():
            # All-NaN series (e.g. an epsilon block the method never solved):
            # the line is invisible, but a distance-based ``markevery`` on an
            # empty transformed path crashes matplotlib -- drop the marker
            # kwargs and keep the (invisible) line for legend compatibility.
            for mk in ("marker", "markersize", "markevery", "markerfacecolor",
                       "markeredgecolor", "markeredgewidth"):
                kw.pop(mk, None)
    (h,) = ax.plot(x, y, *args, **kw)
    return h


def _band(ax, qd, col, ji, ei, step, tr, key, alpha, q1, q2, offset=0, trim=None):
    """Shade the [q1, q2] quantile band for ``df[col]`` iff both quantile
    frames exist and contain ``col``.  No-op otherwise.

    ``key`` is a METHOD_STYLE key: the band takes that series' hue at its
    ``band_alpha``, with no edge line, below every plotted line.  ``alpha``
    is kept for call-site compatibility but METHOD_STYLE is authoritative.

    ``trim``, if given, is an ``(a, b)`` pair applied as a final ``[a:b]``
    slice to the t-axis and both quantile arrays, matching ``_line``."""
    if tr is None or qd is None or q1 not in qd or q2 not in qd:
        return
    if col not in getattr(qd[q1], 'columns', ()) or col not in getattr(qd[q2], 'columns', ()):
        return
    x = np.asarray(tr)
    y1 = np.array(qd[q1][col][(ji*ei)+offset:(ji+1)*ei:step]).astype(float)
    y2 = np.array(qd[q2][col][(ji*ei)+offset:(ji+1)*ei:step]).astype(float)
    if trim is not None:
        a, b = trim
        x, y1, y2 = x[a:b], y1[a:b], y2[a:b]
    if len(y1) == 0 or len(x) != len(y1) or len(x) != len(y2):
        # Same graceful skip as _line: epsilon block absent from the
        # quantile frames.
        return
    ax.fill_between(x, y1=y1, y2=y2, **_bkw(key))


# Valid keys for the ``series`` filter accepted by plot_eval_compare /
# plot_eval_all_compare / plot_eval_all_compare_eval -- pass a list/set of
# these to draw only a subset of methods; ``None`` (default) draws all of
# them, matching prior behavior.
COMPARE_SERIES = (
    "online", "reclustering", "DRO", "SAA",
    "online sgd", "reclustering sgd", "DRO sgd", "cluster SAA",
)


def _want(series, key):
    """True iff ``key`` should be drawn -- everything is drawn when
    ``series`` is None; otherwise only keys present in ``series``."""
    return series is None or key in series


# ---------------------------------------------------------------------------
# Shared-legend grouping: one legend column per semantic group, in canonical
# within-group order.  Column 1 = exact methods, column 2 = subgradient
# variants, column 3 = reference lines.  Labels are matched exactly as the
# plot calls emit them.
# ---------------------------------------------------------------------------
# Each column pairs an exact method with its subgrad variant (mirroring the
# filled/open marker pairing in METHOD_STYLE), keeping the shared legend to
# at most 2 rows so the combined figure stays short on the page.
LEGEND_GROUPS = (
    ("online clustering", "online clustering subgrad"),
    ("reclustering", "reclustering subgrad"),
    ("DRO", "DRO subgrad"),
    ("SAA", "cluster SAA"),
    (r"$H^\star$",),
    ("upper bound", "empirical regret"),
)


def _grouped_legend_entries(handles, labels):
    """Order legend entries so each LEGEND_GROUPS group fills its own column.

    Deduplicates by label (first occurrence wins), sorts each group's
    surviving entries into the canonical order above, drops empty groups,
    and pads shorter columns with invisible dummy handles (blank label) so
    a column-major ``fig.legend`` -- matplotlib fills DOWN each column --
    keeps every group in one clean column.

    Any label not listed in LEGEND_GROUPS is appended to the first
    (exact-methods) column so nothing silently disappears.

    Returns ``(handles, labels, ncol)`` ready to splat into ``fig.legend``.
    """
    entries = {}
    for h, lab in zip(handles, labels):
        if lab not in entries:
            entries[lab] = h
    columns, used = [], set()
    for group in LEGEND_GROUPS:
        col = [(entries[lab], lab) for lab in group if lab in entries]
        used.update(lab for _, lab in col)
        if col:
            columns.append(col)
    leftovers = [(h, lab) for lab, h in entries.items() if lab not in used]
    if leftovers:
        if not columns:
            columns.append([])
        columns[0].extend(leftovers)
    if not columns:
        return [], [], 1
    nrow = max(len(col) for col in columns)
    out_h, out_l = [], []
    for col in columns:
        for h, lab in col:
            out_h.append(h)
            out_l.append(lab)
        for _ in range(nrow - len(col)):
            out_h.append(Line2D([], [], alpha=0.0))
            out_l.append(' ')
    return out_h, out_l, len(columns)


def _grouped_fig_legend(fig, handles, labels, y=0.01, fontsize=None):
    """Grouped shared legend hanging below ``fig`` (see
    ``_grouped_legend_entries``).  Anchored by its TOP edge just under the
    figure so any number of legend rows extends downward without covering
    the bottom axes' x-labels (``bbox_inches='tight'`` picks up the rest)."""
    gh, gl, ncol = _grouped_legend_entries(handles, labels)
    if not gh:
        return None
    kw = {} if fontsize is None else {"fontsize": fontsize}
    # Tight horizontal metrics so the (up to) 6 pairwise columns fit within
    # the 9in figure width without widening the tight bounding box.
    return fig.legend(gh, gl, loc='upper center',
                      bbox_to_anchor=(0.5, y), ncol=ncol,
                      columnspacing=1.1, handlelength=1.6,
                      handletextpad=0.5, **kw)


def plot_eval_all(df, quantiles, df1=None, quantiles1=None, end_ind=61, end_ind_dro=None,
                   j=(0, 0, 0), q=(40, 60), K=5, alpha=0.05, ylim=[0.008, 0.022], legend=True,
                   val2=3, xscale_log=False, saa_prefix='SAA', folderout=None):
    j1, j4, j3 = j
    if end_ind_dro is None:
        end_ind_dro = end_ind
    saa_time, saa_obj, saa_satisfy = (f'{saa_prefix}_time', f'{saa_prefix}_obj_values',
                                       f'{saa_prefix}_satisfy1')
    df = _pick(df, K)
    df1 = _pick(df1, 0)
    quantiles = _pick(quantiles, K)
    quantiles1 = _pick(quantiles1, 0)
    q1, q2 = q
    t_range = _tsource(df, end_ind=end_ind)
    t_range_dro = _tsource(df1, end_ind=end_ind_dro)
    if t_range is None:
        t_range = t_range_dro
    if t_range is None:
        print("plot_eval_all: no dataframes available; skipping plot")
        return
    fig, (ax2, ax3, ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)

    # ax1: computation time
    _line(ax1, df, 'online_time', j1, end_ind, 2, t_range, style='online', label="online clustering")
    _band(ax1, quantiles, 'online_time', j1, end_ind, 2, t_range, 'online', alpha, q1, q2)
    _line(ax1, df, 'MRO_time', j4, end_ind, 2, t_range, style='recluster', label="reclustering")
    _band(ax1, quantiles, 'MRO_time', j4, end_ind, 2, t_range, 'recluster', alpha, q1, q2)
    _line(ax1, df1, 'DRO_time', j3, end_ind_dro, 2, t_range_dro, style='dro', label="DRO")
    _band(ax1, quantiles1, 'DRO_time', j3, end_ind_dro, 2, t_range_dro, 'dro', alpha, q1, q2)
    _line(ax1, df1, 'SA_time', j3, end_ind_dro, 2, t_range_dro, style='saa', label="SAA")
    _band(ax1, quantiles1, 'SA_time', j3, end_ind_dro, 2, t_range_dro, 'saa', alpha, q1, q2)
    _line(ax1, df, saa_time, j1, end_ind, 2, t_range, style='cluster_saa', label="cluster SAA")
    _band(ax1, quantiles, saa_time, j1, end_ind, 2, t_range, 'cluster_saa', alpha, q1, q2)

    ax1.set_xlabel(r'Time step $(t)$')
    ax1.set_xscale("log")
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.25, linewidth=0.4)
    ax1.set_yscale("log")

    # ax2: in-sample objective value
    lines1 = _line(ax2, df, 'obj_values', j1, end_ind, 2, t_range, style='online', label="online clustering")
    _band(ax2, quantiles, 'obj_values', j1, end_ind, 2, t_range, 'online', alpha, q1, q2)
    lines2 = _line(ax2, df, 'MRO_obj_values', j4, end_ind, 2, t_range, style='recluster', label="reclustering")
    _band(ax2, quantiles, 'MRO_obj_values', j4, end_ind, 2, t_range, 'recluster', alpha, q1, q2)
    lines3 = _line(ax2, df1, 'SA_obj_values', j3, end_ind_dro, 2, t_range_dro, style='saa', label="SAA")
    _band(ax2, quantiles1, 'SA_obj_values', j3, end_ind_dro, 2, t_range_dro, 'saa', alpha, q1, q2)
    lines4 = _line(ax2, df1, 'DRO_obj_values', j3, end_ind_dro, 2, t_range_dro, style='dro', label="DRO")
    _band(ax2, quantiles1, 'DRO_obj_values', j3, end_ind_dro, 2, t_range_dro, 'dro', alpha, q1, q2)
    lines_cluster = _line(ax2, df, saa_obj, j1, end_ind, 2, t_range, style='cluster_saa', label="cluster SAA")
    _band(ax2, quantiles, saa_obj, j1, end_ind, 2, t_range, 'cluster_saa', alpha, q1, q2)

    ax2.set_xlabel(r'Time step $(t)$')
    if xscale_log:
        ax2.set_xscale("log")
    ax2.set_title(r'In-sample objective value')
    ax2.grid(True, alpha=0.25, linewidth=0.4)
    ax2.set_ylim(ylim)

    # ax3: confidence
    _line(ax3, df, 'O_satisfy0', j1, end_ind, 2, t_range, style='online', label="online clustering")
    _line(ax3, df, 'MRO_satisfy0', j4, end_ind, 2, t_range, style='recluster', label="reclustering")
    _line(ax3, df1, 'SA_satisfy1', j3, end_ind_dro, 2, t_range_dro, style='saa', label="SAA")
    _line(ax3, df1, 'DRO_satisfy1', j3, end_ind_dro, 2, t_range_dro, style='dro', label="DRO")
    _line(ax3, df, saa_satisfy, j1, end_ind, 2, t_range, style='cluster_saa', label="cluster SAA")
    ax3.set_xlabel(r'Time step $(t)$')
    ax3.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax3.grid(True, alpha=0.25, linewidth=0.4)

    # Shared legend beneath the plots (only over lines that were drawn).
    lines = [ln for ln in (lines1, lines2, lines3, lines4, lines_cluster) if ln is not None]
    labels = [line.get_label() for line in lines]
    if legend and lines:
        fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    plt.savefig(folderout + 'obj_analysis' + str(K) + '.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_eval_all_compare(
    df,  quantiles,  df1, quantiles1,
    df2, quantiles2, df3, quantiles3,
    end_ind=61,
    end_ind_grad=None,     # subgrad epsilon-block length; falls back to end_ind if None
    end_ind_dro=None,      # full DRO/SAA (df1) block length; falls back to end_ind if None
    end_ind_dro_grad=None, # subgrad DRO (df3) block length; falls back to end_ind_grad if None
    j=(0, 0, 0),           # full epsilon-indices: (online, reclustering, DRO/SAA)
    j_grad=(0, 0, 0),      # subgrad epsilon-indices: (online, reclustering, DRO)
    stride=2,              # subsample stride within an epsilon block (full)
    stride_grad=None,      # subgrad stride; falls back to stride if None
    q=(40, 60),
    K=5,
    alpha=0.05,
    ylim=[0.008, 0.022],
    legend=True,
    val2=3,
    saa_prefix='SAA',
    series=None,
    legend_ncol=6,
    folderout=None,
):
    """3-panel comparison plot overlaying two experiment sets.

    Mirrors ``plot_eval_all`` for the *full* set (df/df1, solid lines, suffix
    "full") and overlays a second *subgrad* set (df2/df3, dashed lines, suffix
    "subgrad") on the same axes.  Same colors per method, same markers.

    The two experiment sets are allowed to have different per-epsilon block
    lengths -- pass ``end_ind`` for the full set and ``end_ind_grad`` for the
    subgrad set.  Each set is sliced with its own block length and its own
    ``t``-axis taken from the corresponding dataframe.  Within each set, the
    DRO/SAA dataframe (df1 / df3) can have its own block length distinct from
    the online/reclustering dataframe (df / df2) via ``end_ind_dro`` /
    ``end_ind_dro_grad``.

    Method -> data source:
      online clustering : df[K]   / df2[K]
      reclustering      : df[K]   / df2[K]
      DRO               : df1[0]  / df3[0]
      SAA               : df1[0]  ONLY  (no subgrad overlay)

    ``series`` optionally restricts which methods are drawn -- pass a
    list/set drawn from ``COMPARE_SERIES`` (e.g. ``["online", "DRO",
    "cluster SAA"]``); ``None`` (default) draws everything, as before.
    """
    if end_ind_grad is None:
        end_ind_grad = end_ind
    if end_ind_dro is None:
        end_ind_dro = end_ind
    if end_ind_dro_grad is None:
        end_ind_dro_grad = end_ind_grad
    if stride_grad is None:
        stride_grad = stride
    saa_time, saa_obj = f'{saa_prefix}_time', f'{saa_prefix}_obj_values'
    j1, j4, j3 = j
    j1g, j4g, j3g = j_grad
    df  = _pick(df, K)
    df1 = _pick(df1, 0)
    df2 = _pick(df2, K)
    df3 = _pick(df3, 0)
    quantiles  = _pick(quantiles, K)
    quantiles1 = _pick(quantiles1, 0)
    quantiles2 = _pick(quantiles2, K)
    quantiles3 = _pick(quantiles3, 0)
    q1, q2 = q

    t_range          = _tsource(df, end_ind=end_ind, stride=stride)
    t_range_dro       = _tsource(df1, end_ind=end_ind_dro, stride=stride)
    t_range_grad      = _tsource(df2, end_ind=end_ind_grad, stride=stride_grad)
    t_range_dro_grad  = _tsource(df3, end_ind=end_ind_dro_grad, stride=stride_grad)
    if t_range is None:
        t_range = t_range_dro
    if t_range is None:
        t_range = t_range_grad
    if t_range is None:
        t_range = t_range_dro_grad
    if t_range is None:
        print("plot_eval_all_compare: no dataframes available; skipping plot")
        return
    fig, (ax2, ax3, ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)

    # ============================================================
    # ax1: computation time
    # ============================================================
    if _want(series, "online"):
        _line(ax1, df, 'online_time', j1, end_ind, stride, t_range, style='online', label="online clustering")
        _band(ax1, quantiles, 'online_time', j1, end_ind, stride, t_range, 'online', alpha, q1, q2)
    if _want(series, "reclustering"):
        _line(ax1, df, 'MRO_time', j4, end_ind, stride, t_range, style='recluster', label="Reclustering full")
        _band(ax1, quantiles, 'MRO_time', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2)
    if _want(series, "DRO"):
        _line(ax1, df1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO")
        _band(ax1, quantiles1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2)
    if _want(series, "SAA"):
        _line(ax1, df1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA")
        _band(ax1, quantiles1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2)
    # Per-iteration time panel shows EXACT methods only; the online/subgrad
    # variants' time story lives in plot_cumulative_time (paper decision:
    # per-iteration for exact, cumulative for online variants).
    if _want(series, "cluster SAA"):
        _line(ax1, df, saa_time, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA")
        _band(ax1, quantiles, saa_time, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2)

    ax1.set_xlabel(r'Time step $(t)$')
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.25, linewidth=0.4)
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    # ============================================================
    # ax2: in-sample objective value  (line handles captured for legend)
    # ============================================================
    line_online_full = line_recluster_full = line_DRO_full = None
    line_SAA = line_online_grad = line_recluster_grad = line_DRO_grad = lines_cluster = None
    if _want(series, "online"):
        line_online_full = _line(ax2, df, 'obj_values', j1, end_ind, stride, t_range, style='online', label="online clustering")
        _band(ax2, quantiles, 'obj_values', j1, end_ind, stride, t_range, 'online', alpha, q1, q2)
    if _want(series, "reclustering"):
        line_recluster_full = _line(ax2, df, 'MRO_obj_values', j4, end_ind, stride, t_range, style='recluster', label="reclustering")
        _band(ax2, quantiles, 'MRO_obj_values', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2)
    if _want(series, "DRO"):
        line_DRO_full = _line(ax2, df1, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO")
        _band(ax2, quantiles1, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2)
    if _want(series, "SAA"):
        line_SAA = _line(ax2, df1, 'SA_obj_values', j3, end_ind_dro, stride, t_range_dro, style='saa', label="SAA")
        _band(ax2, quantiles1, 'SA_obj_values', j3, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2)
    if _want(series, "online sgd"):
        line_online_grad = _line(ax2, df2, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad")
        _band(ax2, quantiles2, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2)
    if _want(series, "reclustering sgd"):
        line_recluster_grad = _line(ax2, df2, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad")
        _band(ax2, quantiles2, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2)
    if _want(series, "DRO sgd"):
        line_DRO_grad = _line(ax2, df3, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad")
        _band(ax2, quantiles3, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2)
    if _want(series, "cluster SAA"):
        lines_cluster = _line(ax2, df, saa_obj, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA")
        _band(ax2, quantiles, saa_obj, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2)

    ax2.set_xlabel(r'Time step $(t)$')
    ax2.set_title(r'In-sample objective value')
    ax2.grid(True, alpha=0.25, linewidth=0.4)
    ax2.set_xscale("log")
    ax2.set_ylim(ylim)

    # ============================================================
    # ax3: confidence
    # ============================================================
    if _want(series, "online"):
        _line(ax3, df, 'O_satisfy0', j1, end_ind, stride, t_range, style='online', label="online clustering")
    if _want(series, "reclustering"):
        _line(ax3, df, 'MRO_satisfy0', j4, end_ind, stride, t_range, style='recluster', label="reclustering")
    if _want(series, "DRO"):
        _line(ax3, df1, 'DRO_satisfy1', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO")
    if _want(series, "SAA"):
        _line(ax3, df1, 'SA_satisfy1', j3, end_ind_dro, stride, t_range_dro, style='saa', label="SAA")
    if _want(series, "online sgd"):
        _line(ax3, df2, 'O_satisfy0', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad")
    if _want(series, "reclustering sgd"):
        _line(ax3, df2, 'MRO_satisfy0', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad")
    if _want(series, "DRO sgd"):
        _line(ax3, df3, 'DRO_satisfy1', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad")
    if _want(series, "cluster SAA"):
        _line(ax3, df, f'{saa_prefix}_satisfy1', j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA")

    ax3.set_xlabel(r'Time step $(t)$')
    ax3.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax3.grid(True, alpha=0.25, linewidth=0.4)

    # ---- shared legend over whichever lines were drawn, grouped into
    # exact | subgrad columns (``legend_ncol`` kept for API compat) ----------
    lines = [ln for ln in (
        line_online_full, line_recluster_full, line_DRO_full,
        line_DRO_grad, line_SAA, lines_cluster,
    ) if ln is not None]
    labels = [ln.get_label() for ln in lines]
    if legend and lines:
        _grouped_fig_legend(fig, lines, labels)
    plt.tight_layout()
    plt.savefig(folderout + 'obj_analysis_compare' + str(K) + '.pdf',
                bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_eval_all_compare_eval(
    df,  quantiles,  df1, quantiles1,
    df2, quantiles2, df3, quantiles3,
    end_ind=61,
    end_ind_grad=None,
    end_ind_dro=None,      # full DRO/SAA (df1) block length; falls back to end_ind if None
    end_ind_dro_grad=None, # subgrad DRO (df3) block length; falls back to end_ind_grad if None
    j=(0, 0, 0),
    j_grad=(0, 0, 0),
    stride=2,
    stride_grad=None,
    q=(40, 60),
    K=5,
    alpha=0.05,
    ylim=[0.008, 0.022],
    legend=True,
    val2=3,
    saa_prefix='SAA',
    filename_suffix='',
    series=None,
    legend_ncol=4,
    trim_start=None,
    trim_end=None,
    trim_start_grad=None,
    trim_end_grad=None,
    resolve_t_list=[5,25,50,100],
    resolve_interval=None,
    resolve_t_max=None,
    yscale_log=False,
    yscale_log_out=False,
    jstar=None,
    folderout=None,
):
    """Like plot_eval_all_compare but replaces the confidence panel with
    the out-of-sample evaluation panel from plot_eval_compare.
    All methods (full + subgrad + cluster SAA) are included in the legend.

    ``jstar``, if given, draws a horizontal dotted reference line at the
    true-SAA oracle out-of-sample value on the out-of-sample panel (ax3).

    ``yscale_log_out`` toggles log-scale on just the out-of-sample (ax3)
    panel's y-axis. ``yscale_log`` toggles log-scale on *both* the in-sample
    (ax2) and out-of-sample (ax3) panels' y-axes -- it takes ax3 regardless
    of ``yscale_log_out``. Both default False, matching prior behavior
    (linear).

    ``series`` optionally restricts which methods are drawn -- pass a
    list/set drawn from ``COMPARE_SERIES``; ``None`` (default) draws
    everything, as before.

    ``trim_start``/``trim_end`` optionally cut off every plotted series (t-axis
    and y-values alike) with a final ``[trim_start:trim_end]`` slice, applied
    on top of the usual ``j``/``end_ind``/``stride`` block slicing -- e.g.
    ``trim_end=-2`` drops the last 2 plotted points from every line/band.
    ``trim_start_grad``/``trim_end_grad`` apply that same cutoff to only the
    subgrad series ("online sgd"/"reclustering sgd"/"DRO sgd") instead,
    falling back to ``trim_start``/``trim_end`` if not given -- so the
    full and subgrad experiment sets (which can have different lengths) can
    each be cut off at their own index.

    ``resolve_t_list``/``resolve_interval`` optionally mark, on all 3 panels,
    the timesteps at which a full resolve happened -- thin dotted red
    vertical lines -- matching the periodic-full-solve gating used by the
    ``_comb``/``_grad`` experiment scripts (``t in resolve_t_list`` or
    ``t % resolve_interval == 0``), e.g. ``resolve_t_list=[5,25,50,100],
    resolve_interval=200``.  ``resolve_t_max`` bounds how far the
    ``resolve_interval`` multiples are generated; defaults to the largest
    plotted t.  Neither is drawn if both are left as ``None``.
    """
    if end_ind_grad is None:
        end_ind_grad = end_ind
    if end_ind_dro is None:
        end_ind_dro = end_ind
    if end_ind_dro_grad is None:
        end_ind_dro_grad = end_ind_grad
    if stride_grad is None:
        stride_grad = stride
    if trim_start_grad is None:
        trim_start_grad = trim_start
    if trim_end_grad is None:
        trim_end_grad = trim_end
    trim = None if (trim_start is None and trim_end is None) else (trim_start, trim_end)
    trim_grad = None if (trim_start_grad is None and trim_end_grad is None) else (trim_start_grad, trim_end_grad)
    saa_time, saa_obj, saa_eval = (f'{saa_prefix}_time', f'{saa_prefix}_obj_values',
                                    f'{saa_prefix}_eval1')
    j1, j4, j3 = j
    j1g, j4g, j3g = j_grad
    df  = _pick(df, K)
    df1 = _pick(df1, 0)
    df2 = _pick(df2, K)
    df3 = _pick(df3, 0)
    quantiles  = _pick(quantiles, K)
    quantiles1 = _pick(quantiles1, 0)
    quantiles2 = _pick(quantiles2, K)
    quantiles3 = _pick(quantiles3, 0)
    q1, q2 = q

    t_range          = _tsource(df, end_ind=end_ind, stride=stride)
    t_range_dro       = _tsource(df1, end_ind=end_ind_dro, stride=stride)
    t_range_grad      = _tsource(df2, end_ind=end_ind_grad, stride=stride_grad)
    t_range_dro_grad  = _tsource(df3, end_ind=end_ind_dro_grad, stride=stride_grad)
    if t_range is None:
        t_range = t_range_dro
    if t_range is None:
        t_range = t_range_grad
    if t_range is None:
        t_range = t_range_dro_grad
    if t_range is None:
        print("plot_eval_all_compare_eval: no dataframes available; skipping plot")
        return
    fig, (ax2, ax3, ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)

    # ============================================================
    # ax1: computation time
    # ============================================================
    if _want(series, "online"):
        _line(ax1, df, 'online_time', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax1, quantiles, 'online_time', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        _line(ax1, df, 'MRO_time', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax1, quantiles, 'MRO_time', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        _line(ax1, df1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax1, quantiles1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        _line(ax1, df1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax1, quantiles1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    # Time panel: exact methods only (see plot_cumulative_time for subgrad).
    if _want(series, "cluster SAA"):
        _line(ax1, df, saa_time, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax1, quantiles, saa_time, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    ax1.set_xlabel(r'Time step $(t)$')
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.25, linewidth=0.4)
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    # ============================================================
    # ax2: in-sample objective value
    # ============================================================
    line_online_full = line_recluster_full = line_DRO_full = None
    line_SAA = line_online_grad = line_recluster_grad = line_DRO_grad = lines_cluster = None
    if _want(series, "online"):
        line_online_full = _line(ax2, df, 'obj_values', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax2, quantiles, 'obj_values', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        line_recluster_full = _line(ax2, df, 'MRO_obj_values', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax2, quantiles, 'MRO_obj_values', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        line_DRO_full = _line(ax2, df1, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax2, quantiles1, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        line_SAA = _line(ax2, df1, 'SA_obj_values', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax2, quantiles1, 'SA_obj_values', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    if _want(series, "online sgd"):
        line_online_grad = _line(ax2, df2, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
        _band(ax2, quantiles2, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "reclustering sgd"):
        line_recluster_grad = _line(ax2, df2, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
        _band(ax2, quantiles2, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "DRO sgd"):
        line_DRO_grad = _line(ax2, df3, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
        _band(ax2, quantiles3, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "cluster SAA"):
        lines_cluster = _line(ax2, df, saa_obj, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax2, quantiles, saa_obj, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    _jstar_line(ax2, jstar)
    ax2.set_xlabel(r'Time step $(t)$')
    ax2.set_title(r'In-sample objective value')
    ax2.grid(True, alpha=0.25, linewidth=0.4)
    ax2.set_xscale("log")
    if yscale_log:
        ax2.set_yscale("log")

    # ============================================================
    # ax3: out-of-sample evaluation
    # ============================================================
    if _want(series, "online"):
        _line(ax3, df, 'O_eval1', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax3, quantiles, 'O_eval1', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        _line(ax3, df, 'MRO_eval1', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax3, quantiles, 'MRO_eval1', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        _line(ax3, df1, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax3, quantiles1, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        _line(ax3, df1, 'SA_eval2', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax3, quantiles1, 'SA_eval2', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    if _want(series, "online sgd"):
        _line(ax3, df2, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
        _band(ax3, quantiles2, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "reclustering sgd"):
        _line(ax3, df2, 'MRO_eval1', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
        _band(ax3, quantiles2, 'MRO_eval1', j4g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "DRO sgd"):
        _line(ax3, df3, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
        _band(ax3, quantiles3, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "cluster SAA"):
        _line(ax3, df, saa_eval, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax3, quantiles, saa_eval, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    line_jstar = _jstar_line(ax3, jstar)
    ax3.set_xlabel(r'Time step $(t)$')
    ax3.set_title(r'Out-of-sample objective value')
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.25, linewidth=0.4)
    if yscale_log or yscale_log_out:
        ax3.set_yscale("log")
    # In-sample and out-of-sample value panels share the same y-axis (PI
    # request): identical scale and limits (union of both data ranges,
    # or the explicit ylim when given).
    if not (yscale_log or yscale_log_out):
        _lo = min(ax2.get_ylim()[0], ax3.get_ylim()[0])
        _hi = max(ax2.get_ylim()[1], ax3.get_ylim()[1])
        ax2.set_ylim(_lo, _hi)
        ax3.set_ylim(_lo, _hi)
    elif yscale_log_out and not yscale_log:
        # A log OOS panel next to a linear in-sample panel cannot share an
        # axis; force log on both when all in-sample data is positive,
        # else linear on both.
        _ins_lo = ax2.get_ylim()[0]
        if _ins_lo > 0:
            ax2.set_yscale("log")
        else:
            ax3.set_yscale("linear")
        _lo = min(ax2.get_ylim()[0], ax3.get_ylim()[0])
        _hi = max(ax2.get_ylim()[1], ax3.get_ylim()[1])
        ax2.set_ylim(_lo, _hi)
        ax3.set_ylim(_lo, _hi)

    # ---- mark full-resolve timesteps on all 3 panels -------------------------
    if resolve_t_list or resolve_interval:
        resolve_times = set(resolve_t_list or [])
        # Bounds of what's actually plotted (post-trim) -- markers outside
        # this range don't correspond to any visible line/band and would
        # otherwise show up floating past the trimmed-off tail (or before a
        # trimmed-off head).
        tmin_data, tmax_eata = _trimmed_bounds([
            (t_range, trim), (t_range_dro, trim),
            (t_range_grad, trim_grad), (t_range_dro_grad, trim_grad),
        ])
        if resolve_interval:
            tmax = resolve_t_max if resolve_t_max is not None else (
                int(tmax_eata) if tmax_eata is not None else 0)
            resolve_times.update(range(resolve_interval, int(tmax) + 1, resolve_interval))
        if tmin_data is not None and tmax_eata is not None:
            resolve_times = {rt for rt in resolve_times if tmin_data <= rt <= tmax_eata}
        for ax in (ax1, ax2, ax3):
            for rt in sorted(resolve_times):
                ax.axvline(rt, **_mkw('resolve_marker'))

    ax2.set_ylim(ylim)
    ax3.set_ylim(ylim)

    # ---- shared legend over whichever lines were drawn, grouped into
    # exact | subgrad | reference columns (``legend_ncol`` kept for API
    # compat; the grouping decides the column count) --------------------------
    lines = [ln for ln in (
        line_online_full, line_recluster_full, line_DRO_full, line_SAA,
        line_online_grad, line_recluster_grad, line_DRO_grad, lines_cluster,
        line_jstar,
    ) if ln is not None]
    labels = [ln.get_label() for ln in lines]
    if legend and lines:
        _grouped_fig_legend(fig, lines, labels)
    plt.tight_layout()
    plt.savefig(folderout + 'obj_analysis_compare_eval' + str(K) + filename_suffix + '.pdf',
                bbox_inches='tight', dpi=300)
    plt.close(fig)


def _pct_diff_df(df, col_ref):
    """Return a copy of ``df`` with each column in ``col_ref`` replaced by
    signed-ratio-difference from its reference value:
        new_col = (col - ref) / |ref|
    Columns not in ``col_ref``, or whose reference is zero/None, are unchanged.
    """
    if df is None:
        return None
    df2 = df.copy()
    for col, ref in col_ref.items():
        if col in df2.columns and ref is not None and abs(ref) > 1e-15:
            df2[col] = ((np.array(df2[col]).astype(float) - ref)
                        / abs(ref))
    return df2


def plot_eval_all_compare_eval_pct(
    df,  quantiles,  df1, quantiles1,
    df2, quantiles2, df3, quantiles3,
    true_ref,
    end_ind=61,
    end_ind_grad=None,
    end_ind_dro=None,
    end_ind_dro_grad=None,
    j=(0, 0, 0),
    j_grad=(0, 0, 0),
    stride=2,
    stride_grad=None,
    q=(40, 60),
    K=5,
    alpha=0.05,
    ylim_pct=None,
    legend=True,
    val2=3,
    saa_prefix='SAA',
    filename_suffix='',
    series=None,
    legend_ncol=4,
    trim_start=None,
    trim_end=None,
    trim_start_grad=None,
    trim_end_grad=None,
    resolve_t_list=None,
    resolve_interval=None,
    resolve_t_max=None,
    yscale_log=True,
    folderout=None,
):
    """Like plot_eval_all_compare_eval but the in-sample (ax2) and out-of-sample
    (ax3) panels show **signed ratio difference** ((value - ref) / |ref|) from
    the oracle reference solution stored in ``true_ref`` -- positive means the
    method's value is above the reference, negative means below.

    ``true_ref`` is a dict (or single-row DataFrame) with keys:
        insample_obj_mean   -- oracle in-sample objective (scalar)
        outsample_eval1_mean -- oracle out-of-sample hinge (j=1 window, scalar)

    The computation-time panel (ax1) is unchanged from plot_eval_all_compare_eval.

    ``trim_start``/``trim_end`` optionally cut off every plotted series (t-axis
    and y-values alike) with a final ``[trim_start:trim_end]`` slice, applied
    on top of the usual ``j``/``end_ind``/``stride`` block slicing -- e.g.
    ``trim_end=-2`` drops the last 2 plotted points from every line/band.
    ``trim_start_grad``/``trim_end_grad`` apply that same cutoff to only the
    subgrad series ("online sgd"/"reclustering sgd"/"DRO sgd") instead,
    falling back to ``trim_start``/``trim_end`` if not given.

    ``resolve_t_list``/``resolve_interval`` optionally mark, on all 3 panels,
    the timesteps at which a full resolve happened -- thin dotted red
    vertical lines -- matching the periodic-full-solve gating used by the
    ``_comb``/``_grad`` experiment scripts (``t in resolve_t_list`` or
    ``t % resolve_interval == 0``), e.g. ``resolve_t_list=[5,25,50,100],
    resolve_interval=200``.  ``resolve_t_max`` bounds how far the
    ``resolve_interval`` multiples are generated; defaults to the largest
    plotted t.  Neither is drawn if both are left as ``None``.

    ``yscale_log`` toggles log-scale on ax2/ax3's y-axis (default True,
    matching prior behavior).  Since the ratio is now signed (can be
    negative), pass ``yscale_log=False`` when values may cross zero --
    matplotlib's log scale silently drops non-positive points.

    Output filename: ``obj_analysis_compare_eval_pct{K}{filename_suffix}.pdf``.
    """
    # --- resolve true reference scalars ---
    if isinstance(true_ref, pd.DataFrame):
        true_ref = true_ref.iloc[0].to_dict()

    ref_insample = true_ref.get('insample_obj_mean')
    ref_oos      = true_ref.get('outsample_eval1_mean')

    if end_ind_grad is None:
        end_ind_grad = end_ind
    if end_ind_dro is None:
        end_ind_dro = end_ind
    if end_ind_dro_grad is None:
        end_ind_dro_grad = end_ind_grad
    if stride_grad is None:
        stride_grad = stride
    if trim_start_grad is None:
        trim_start_grad = trim_start
    if trim_end_grad is None:
        trim_end_grad = trim_end
    trim = None if (trim_start is None and trim_end is None) else (trim_start, trim_end)
    trim_grad = None if (trim_start_grad is None and trim_end_grad is None) else (trim_start_grad, trim_end_grad)

    saa_time, saa_obj, saa_eval = (f'{saa_prefix}_time', f'{saa_prefix}_obj_values',
                                    f'{saa_prefix}_eval1')
    j1, j4, j3 = j
    j1g, j4g, j3g = j_grad

    # --- pick K-slices ---
    df  = _pick(df,  K);  df1 = _pick(df1, 0)
    df2 = _pick(df2, K);  df3 = _pick(df3, 0)
    quantiles  = _pick(quantiles,  K);  quantiles1 = _pick(quantiles1, 0)
    quantiles2 = _pick(quantiles2, K);  quantiles3 = _pick(quantiles3, 0)
    q1, q2 = q

    t_range          = _tsource(df,  end_ind=end_ind,          stride=stride)
    t_range_dro      = _tsource(df1, end_ind=end_ind_dro,      stride=stride)
    t_range_grad     = _tsource(df2, end_ind=end_ind_grad,     stride=stride_grad)
    t_range_dro_grad = _tsource(df3, end_ind=end_ind_dro_grad, stride=stride_grad)
    if t_range is None: t_range = t_range_dro
    if t_range is None: t_range = t_range_grad
    if t_range is None: t_range = t_range_dro_grad
    if t_range is None:
        print("plot_eval_all_compare_eval_pct: no dataframes available; skipping")
        return

    # --- build pct-diff DataFrames (in-sample columns) ---
    insample_cols = {
        'obj_values':     ref_insample,
        'MRO_obj_values': ref_insample,
        'DRO_obj_values': ref_insample,
        'SA_obj_values':  ref_insample,
        saa_obj:          ref_insample,
    }
    oos_cols = {
        'O_eval1':   ref_oos,
        'MRO_eval1': ref_oos,
        'DRO_eval2': ref_oos,
        'SA_eval2':  ref_oos,
        saa_eval:    ref_oos,
    }
    all_cols = {**insample_cols, **oos_cols}

    df_p  = _pct_diff_df(df,  all_cols)
    df1_p = _pct_diff_df(df1, all_cols)
    df2_p = _pct_diff_df(df2, all_cols)
    df3_p = _pct_diff_df(df3, all_cols)

    def _pct_qd(qd, col_ref):
        if qd is None:
            return None
        return {qv: _pct_diff_df(qd[qv], col_ref) for qv in qd}

    qd_p  = _pct_qd(quantiles,  all_cols)
    qd1_p = _pct_qd(quantiles1, all_cols)
    qd2_p = _pct_qd(quantiles2, all_cols)
    qd3_p = _pct_qd(quantiles3, all_cols)

    fig, (ax2, ax3, ax1) = plt.subplots(1, 3, figsize=(9, val2), dpi=300)

    # ============================================================
    # ax1: computation time (unchanged)
    # ============================================================
    if _want(series, "online"):
        _line(ax1, df, 'online_time', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax1, quantiles, 'online_time', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        _line(ax1, df, 'MRO_time', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax1, quantiles, 'MRO_time', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        _line(ax1, df1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax1, quantiles1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        _line(ax1, df1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax1, quantiles1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    # Time panel: exact methods only (see plot_cumulative_time for subgrad).
    if _want(series, "cluster SAA"):
        _line(ax1, df, saa_time, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax1, quantiles, saa_time, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    ax1.set_xlabel(r'Time step $(t)$')
    ax1.set_title(r'Computation time per iteration (s)')
    ax1.grid(True, alpha=0.25, linewidth=0.4)
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    # ============================================================
    # ax2: in-sample objective — signed ratio diff from oracle
    # ============================================================
    line_online_full = line_recluster_full = line_DRO_full = None
    line_SAA = line_online_grad = line_recluster_grad = line_DRO_grad = lines_cluster = None
    if _want(series, "online"):
        line_online_full = _line(ax2, df_p, 'obj_values', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax2, qd_p, 'obj_values', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        line_recluster_full = _line(ax2, df_p, 'MRO_obj_values', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax2, qd_p, 'MRO_obj_values', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        line_DRO_full = _line(ax2, df1_p, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax2, qd1_p, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        line_SAA = _line(ax2, df1_p, 'SA_obj_values',0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax2, qd1_p, 'SA_obj_values', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    if _want(series, "online sgd"):
        line_online_grad = _line(ax2, df2_p, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
        _band(ax2, qd2_p, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "reclustering sgd"):
        line_recluster_grad = _line(ax2, df2_p, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
        _band(ax2, qd2_p, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "DRO sgd"):
        line_DRO_grad = _line(ax2, df3_p, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
        _band(ax2, qd3_p, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "cluster SAA"):
        lines_cluster = _line(ax2, df_p, saa_obj, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax2, qd_p, saa_obj, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    ax2.set_xlabel(r'Time step $(t)$')
    ax2.set_title(r'In-sample obj.\ $(\cdot - \mathrm{opt})/|\mathrm{opt}|$')
    ax2.grid(True, alpha=0.25, linewidth=0.4)
    ax2.set_xscale("log")
    if yscale_log:
        ax2.set_yscale("log")
    if ylim_pct is not None:
        ax2.set_ylim(ylim_pct)

    # ============================================================
    # ax3: out-of-sample — signed ratio diff from oracle
    # ============================================================
    if _want(series, "online"):
        _line(ax3, df_p, 'O_eval1', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax3, qd_p, 'O_eval1', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        _line(ax3, df_p, 'MRO_eval1', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax3, qd_p, 'MRO_eval1', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        _line(ax3, df1_p, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax3, qd1_p, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        _line(ax3, df1_p, 'SA_eval2', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax3, qd1_p, 'SA_eval2', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    if _want(series, "online sgd"):
        _line(ax3, df2_p, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
        _band(ax3, qd2_p, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "reclustering sgd"):
        _line(ax3, df2_p, 'MRO_eval1', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
        _band(ax3, qd2_p, 'MRO_eval1', j4g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "DRO sgd"):
        _line(ax3, df3_p, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
        _band(ax3, qd3_p, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "cluster SAA"):
        _line(ax3, df_p, saa_eval, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax3, qd_p, saa_eval, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    ax3.set_xlabel(r'Time step $(t)$')
    ax3.set_title(r'Out-of-sample $(\cdot - \mathrm{opt})/|\mathrm{opt}|$')
    ax3.set_xscale("log")
    if yscale_log:
        ax3.set_yscale("log")
    ax3.grid(True, alpha=0.25, linewidth=0.4)
    if ylim_pct is not None:
        ax3.set_ylim(ylim_pct)

    # ---- mark full-resolve timesteps on all 3 panels -------------------------
    if resolve_t_list or resolve_interval:
        resolve_times = set(resolve_t_list or [])
        # Bounds of what's actually plotted (post-trim) -- markers outside
        # this range don't correspond to any visible line/band and would
        # otherwise show up floating past the trimmed-off tail (or before a
        # trimmed-off head).
        tmin_data, tmax_eata = _trimmed_bounds([
            (t_range, trim), (t_range_dro, trim),
            (t_range_grad, trim_grad), (t_range_dro_grad, trim_grad),
        ])
        if resolve_interval:
            tmax = resolve_t_max if resolve_t_max is not None else (
                int(tmax_eata) if tmax_eata is not None else 0)
            resolve_times.update(range(resolve_interval, int(tmax) + 1, resolve_interval))
        if tmin_data is not None and tmax_eata is not None:
            resolve_times = {rt for rt in resolve_times if tmin_data <= rt <= tmax_eata}
        for ax in (ax1, ax2, ax3):
            for rt in sorted(resolve_times):
                ax.axvline(rt, **_mkw('resolve_marker'))

    # Grouped legend (exact | subgrad columns); ``legend_ncol`` kept for
    # API compat.
    lines = [ln for ln in (
        line_online_full, line_recluster_full, line_DRO_full, line_SAA,
        line_online_grad, line_recluster_grad, line_DRO_grad, lines_cluster,
    ) if ln is not None]
    labels = [ln.get_label() for ln in lines]
    if legend and lines:
        _grouped_fig_legend(fig, lines, labels)
    plt.tight_layout()
    fname = (folderout + 'obj_analysis_compare_eval_pct'
             + str(K) + filename_suffix + '.pdf')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_confidence_regret(
    df,  quantiles,  df1, quantiles1,
    df2=None, quantiles2=None, df3=None, quantiles3=None,
    end_ind=61,
    end_ind_grad=None,
    end_ind_dro=None,
    end_ind_dro_grad=None,
    j=(0, 0, 0),
    j_grad=(0, 0, 0),
    stride=2,
    stride_grad=None,
    q=(40, 60),
    K=5,
    alpha=0.05,
    ylim_regret=[0.008, 0.022],
    legend=True,
    val2=2.1,
    saa_prefix='SAA',
    filename_suffix='',
    series=None,
    legend_ncol=6,
    trim_start=None,
    trim_end=None,
    trim_start_grad=None,
    trim_end_grad=None,
    trim_start_regret=None,
    trim_end_regret=None,
    resolve_t_list=[5, 25, 50, 100],
    resolve_interval=None,
    resolve_t_max=None,
    folderout=None,
):
    """2-panel figure combining two existing plots side by side.

    LHS -- the "confidence" panel from ``plot_eval_all_compare`` (empirical
    coverage $1-\\hat\\beta_t$ vs. time), built with
    ``plot_eval_all_compare_eval``'s fuller feature set: full + subgrad
    overlay (``df2``/``df3``), ``series`` filtering, ``trim_start``/
    ``trim_end`` cutoffs, and full-resolve marker lines
    (``resolve_t_list``/``resolve_interval``) -- see that function's
    docstring for what each of those does.

    RHS -- ``plot_regret_new``'s online-clustering dynamic-regret panel
    (``df``/``df1`` only; that plot has never supported a subgrad overlay,
    so ``df2``/``df3`` don't feed it).  ``trim_start_regret``/
    ``trim_end_regret`` apply a final ``[a:b]`` slice to this panel only,
    independent of ``trim_start``/``trim_end`` (which only affect the LHS
    confidence panel) -- e.g. ``trim_start_regret=2`` drops the first 2
    plotted points from every regret-panel line/band.

    Output filename: ``confidence_regret{K}{filename_suffix}.pdf``.
    """
    if end_ind_grad is None:
        end_ind_grad = end_ind
    if end_ind_dro is None:
        end_ind_dro = end_ind
    if end_ind_dro_grad is None:
        end_ind_dro_grad = end_ind_grad
    if stride_grad is None:
        stride_grad = stride
    if trim_start_grad is None:
        trim_start_grad = trim_start
    if trim_end_grad is None:
        trim_end_grad = trim_end
    trim = None if (trim_start is None and trim_end is None) else (trim_start, trim_end)
    trim_grad = None if (trim_start_grad is None and trim_end_grad is None) else (trim_start_grad, trim_end_grad)
    trim_regret = None if (trim_start_regret is None and trim_end_regret is None) else (trim_start_regret, trim_end_regret)
    saa_satisfy = f'{saa_prefix}_satisfy1'
    j1, j4, j3 = j
    j1g, j4g, j3g = j_grad

    df_c  = _pick(df, K);  df1_c = _pick(df1, 0)
    df2_c = _pick(df2, K); df3_c = _pick(df3, 0)
    quantiles_c  = _pick(quantiles, K);  quantiles1_c = _pick(quantiles1, 0)
    quantiles2_c = _pick(quantiles2, K); quantiles3_c = _pick(quantiles3, 0)
    q1, q2 = q

    t_range          = _tsource(df_c,  end_ind=end_ind,          stride=stride)
    t_range_dro      = _tsource(df1_c, end_ind=end_ind_dro,      stride=stride)
    t_range_grad     = _tsource(df2_c, end_ind=end_ind_grad,     stride=stride_grad)
    t_range_dro_grad = _tsource(df3_c, end_ind=end_ind_dro_grad, stride=stride_grad)
    if t_range is None: t_range = t_range_dro
    if t_range is None: t_range = t_range_grad
    if t_range is None: t_range = t_range_dro_grad
    if t_range is None:
        print("plot_confidence_regret: no dataframes available; skipping")
        return

    fig, (ax_donf, ax_regret) = plt.subplots(1, 2, figsize=(8.5, val2), dpi=300)

    # ============================================================
    # LHS: confidence panel (plot_eval_all_compare's ax3, eval-style features)
    # ============================================================
    line_online_full = line_recluster_full = line_DRO_full = None
    line_SAA = line_online_grad = line_recluster_grad = line_DRO_grad = lines_cluster = None
    if _want(series, "online"):
        line_online_full = _line(ax_donf, df_c, 'O_satisfy0', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
    if _want(series, "reclustering"):
        line_recluster_full = _line(ax_donf, df_c, 'MRO_satisfy0', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
    if _want(series, "DRO"):
        line_DRO_full = _line(ax_donf, df1_c, 'DRO_satisfy1', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
    if _want(series, "SAA"):
        line_SAA = _line(ax_donf, df1_c, 'SA_satisfy1', j3, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
    if _want(series, "online sgd"):
        line_online_grad = _line(ax_donf, df2_c, 'O_satisfy0', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
    if _want(series, "reclustering sgd"):
        line_recluster_grad = _line(ax_donf, df2_c, 'MRO_satisfy0', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
    if _want(series, "DRO sgd"):
        line_DRO_grad = _line(ax_donf, df3_c, 'DRO_satisfy1', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
    if _want(series, "cluster SAA"):
        lines_cluster = _line(ax_donf, df_c, saa_satisfy, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
    ax_donf.set_xlabel(r'Time step $(t)$')
    ax_donf.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax_donf.grid(True, alpha=0.25, linewidth=0.4)
    ax_donf.set_xscale("log")

    if resolve_t_list or resolve_interval:
        resolve_times = set(resolve_t_list or [])
        tmin_data, tmax_eata = _trimmed_bounds([
            (t_range, trim), (t_range_dro, trim),
            (t_range_grad, trim_grad), (t_range_dro_grad, trim_grad),
        ])
        if resolve_interval:
            tmax = resolve_t_max if resolve_t_max is not None else (
                int(tmax_eata) if tmax_eata is not None else 0)
            resolve_times.update(range(resolve_interval, int(tmax) + 1, resolve_interval))
        if tmin_data is not None and tmax_eata is not None:
            resolve_times = {rt for rt in resolve_times if tmin_data <= rt <= tmax_eata}
        for rt in sorted(resolve_times):
            ax_donf.axvline(rt, **_mkw('resolve_marker'))

    conf_lines = [ln for ln in (
        line_online_full, line_recluster_full, line_DRO_full, line_SAA,
        line_online_grad, line_recluster_grad, line_DRO_grad, lines_cluster,
    ) if ln is not None]

    # ============================================================
    # RHS: plot_regret_new's dynamic-regret panel, unchanged
    # ============================================================
    t_range_regret = _tsource(df_c, end_ind=end_ind, offset=1)
    if t_range_regret is None:
        t_range_regret = _tsource(df1_c, end_ind=end_ind_dro, offset=1)

    line_bound = line_regret = None
    if t_range_regret is not None:
        n = min(len(t_range_regret), int(end_ind_dro / 2))
        t_range_combo = t_range_regret[:n]

        def _trimmed(x, y):
            x, y = np.asarray(x), np.asarray(y)
            if trim_regret is not None:
                a, b = trim_regret
                x, y = x[a:b], y[a:b]
            return x, y

        if _hc(df_c, 'regret_bound') and _hc(df_c, 'sig_val'):
            y_ub = 5*df_c['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2] + np.array([5*np.sum(df_c['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1, int(end_ind/2)+1)])
            x_ub, y_ub = _trimmed(t_range_regret, y_ub)
            (line_bound,) = ax_regret.plot(x_ub, y_ub, label="upper bound", **_mkw('bound'))

        if _hc(df_c, 'worst_values_regret') and _hc(df1_c, 'DRO_obj_values'):
            online_worst = np.array(df_c['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
            dro_obj = np.array(df1_c['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
            y_er = np.array([np.sum((online_worst-dro_obj)[:i+1])/(i) for i in range(1, n+1)])
            x_er, y_er = _trimmed(t_range_combo, y_er)
            (line_regret,) = ax_regret.plot(x_er, y_er, label="empirical regret", **_mkw('online'))

        def _qhas(qd, *cols):
            return qd is not None and q1 in qd and q2 in qd and all(_hc(qd[q1], c) and _hc(qd[q2], c) for c in cols)

        if _qhas(quantiles_c, 'worst_values_regret') and _qhas(quantiles1_c, 'DRO_obj_values'):
            online_worst_q1 = np.array(quantiles_c[q1]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
            online_worst_q2 = np.array(quantiles_c[q2]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
            dro_obj_q1 = np.array(quantiles1_c[q1]['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
            dro_obj_q2 = np.array(quantiles1_c[q2]['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
            y1_er = np.array([np.sum((online_worst_q1-dro_obj_q1)[:i+1])/(i) for i in range(1, n+1)])
            y2_er = np.array([np.sum((online_worst_q2-dro_obj_q2)[:i+1])/(i) for i in range(1, n+1)])
            x_erb, y1_erb = _trimmed(t_range_combo, y1_er)
            _,     y2_erb = _trimmed(t_range_combo, y2_er)
            ax_regret.fill_between(x_erb, y1=y1_erb, y2=y2_erb, **_bkw('online'))

        if _qhas(quantiles_c, 'regret_bound', 'sig_val'):
            y1_ub = np.array(5*quantiles_c[q1]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]) + np.array([5*np.sum(quantiles_c[q1]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1, int(end_ind/2)+1)])
            y2_ub = np.array(5*quantiles_c[q2]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]) + np.array([5*np.sum(quantiles_c[q2]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1, int(end_ind/2)+1)])
            x_ubb, y1_ubb = _trimmed(t_range_regret, y1_ub)
            _,     y2_ubb = _trimmed(t_range_regret, y2_ub)
            ax_regret.fill_between(x_ubb, y1=y1_ubb, y2=y2_ubb, **_bkw('bound'))
    else:
        print("plot_confidence_regret: no regret data available; right panel left blank")

    ax_regret.set_xlabel(r'Time step $(t)$')
    ax_regret.set_title(r'Online clustering dynamic regret')
    ax_regret.set_ylim(ylim_regret)
    ax_regret.set_yscale('log')
    ax_regret.set_xscale('log')
    ax_regret.grid(True, alpha=0.25, linewidth=0.4)

    # ---- single shared legend beneath the figure: confidence-panel lines
    # plus the regret panel's empirical-regret / upper-bound lines, grouped
    # into exact | subgrad | reference columns (``legend_ncol`` kept for
    # API compat) -----------------------------------------------------------
    all_lines = conf_lines + [ln for ln in (line_regret, line_bound) if ln is not None]
    all_labels = [ln.get_label() for ln in all_lines]
    if legend and all_lines:
        _grouped_fig_legend(fig, all_lines, all_labels)

    plt.tight_layout()
    fname = folderout + 'confidence_regret' + str(K) + filename_suffix + '.pdf'
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_eval(df, quantiles, df1=None, quantiles1=None, end_ind=61, end_ind_dro=None,
              j=(0, 0, 0), q=(40, 60), K=5, alpha=0.05, legend=True, ylim=[0.008, 0.05],
              xscale_log=True, saa_prefix='SAA', jstar=None, folderout=None):
    j1, j2, j3 = j
    if end_ind_dro is None:
        end_ind_dro = end_ind
    saa_eval = f'{saa_prefix}_eval1'
    df = _pick(df, K)
    df1 = _pick(df1, 0)
    quantiles = _pick(quantiles, K)
    quantiles1 = _pick(quantiles1, 0)
    q1, q2 = q
    t_range = _tsource(df, end_ind=end_ind, plus=1)
    t_range_dro = _tsource(df1, end_ind=end_ind_dro, plus=1)
    if t_range is None:
        t_range = t_range_dro
    if t_range is None:
        print("plot_eval: no dataframes available; skipping plot")
        return
    fig = plt.figure(figsize=(4.3, 2.1), dpi=300)

    _line(plt, df, 'O_eval1', j1, end_ind, 2, t_range, style='online', label="online clustering")
    _band(plt, quantiles, 'O_eval1', j1, end_ind, 2, t_range, 'online', alpha, q1, q2)
    _line(plt, df, 'MRO_eval1', j2, end_ind, 2, t_range, style='recluster', label="reclustering")
    _band(plt, quantiles, 'MRO_eval1', j2, end_ind, 2, t_range, 'recluster', alpha, q1, q2)
    _line(plt, df1, 'DRO_eval2', j3, end_ind_dro, 2, t_range_dro, style='dro', label="DRO")
    _band(plt, quantiles1, 'DRO_eval2', j3, end_ind_dro, 2, t_range_dro, 'dro', alpha, q1, q2)
    _line(plt, df1, 'SA_eval2', j3, end_ind_dro, 2, t_range_dro, style='saa', label="SAA")
    _band(plt, quantiles1, 'SA_eval2', j3, end_ind_dro, 2, t_range_dro, 'saa', alpha, q1, q2)
    _line(plt, df, saa_eval, j1, end_ind, 2, t_range, style='cluster_saa', label="cluster SAA")
    _band(plt, quantiles, saa_eval, j1, end_ind, 2, t_range, 'cluster_saa', alpha, q1, q2)
    _jstar_line(plt.gca(), jstar)

    if xscale_log:
        plt.xscale("log")
    plt.ylim(ylim)
    if legend:
        plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.title(f'Out-of-sample expected value, $K$ = {K}')
    plt.grid(True, alpha=0.25, linewidth=0.4)
    suffix = '_log' if xscale_log else ''
    plt.savefig(folderout + f'eval_analysis{K}{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_eval_compare(
    df,  quantiles,  df1, quantiles1,
    df2, quantiles2, df3, quantiles3,
    end_ind=61,
    end_ind_grad=None,     # subgrad epsilon-block length; falls back to end_ind if None
    end_ind_dro=None,      # full DRO/SAA (df1) block length; falls back to end_ind if None
    end_ind_dro_grad=None, # subgrad DRO (df3) block length; falls back to end_ind_grad if None
    j=(0, 0, 0),           # full epsilon-indices: (online, reclustering, DRO/SAA)
    j_grad=(0, 0, 0),      # subgrad epsilon-indices: (online, reclustering, DRO)
    stride=2,              # subsample stride within an epsilon block (full)
    stride_grad=None,      # subgrad stride; falls back to stride if None
    q=(40, 60),
    K=5,
    alpha=0.05,
    legend=True,
    ylim=[0.008, 0.05],
    saa_prefix='SAA',
    series=None,
    legend_ncol=2,
    jstar=None,
    folderout=None,
):
    """Out-of-sample expected value -- full vs subgrad overlay.

    Mirrors ``plot_eval`` for the full set (solid lines, suffix "full") and
    overlays a second subgrad set (dashed lines + lighter colors, suffix
    "subgrad").  SAA is not overlaid.  ``end_ind_dro`` / ``end_ind_dro_grad``
    let the DRO/SAA dataframe (df1 / df3) use a block length distinct from
    df / df2.

    Method -> data source:
      online clustering : df[K]   / df2[K]
      reclustering      : df[K]   / df2[K]
      DRO               : df1[0]  / df3[0]
      SAA               : df1[0]  ONLY  (no subgrad overlay)

    ``series`` optionally restricts which methods are drawn -- pass a
    list/set drawn from ``COMPARE_SERIES``; ``None`` (default) draws
    everything, as before.
    """
    if end_ind_grad is None:
        end_ind_grad = end_ind
    if end_ind_dro is None:
        end_ind_dro = end_ind
    if end_ind_dro_grad is None:
        end_ind_dro_grad = end_ind_grad
    if stride_grad is None:
        stride_grad = stride
    saa_eval = f'{saa_prefix}_eval1'
    j1, j2, j3 = j
    j1g, j2g, j3g = j_grad
    df  = _pick(df, K)
    df1 = _pick(df1, 0)
    df2 = _pick(df2, K)
    df3 = _pick(df3, 0)
    quantiles  = _pick(quantiles, K)
    quantiles1 = _pick(quantiles1, 0)
    quantiles2 = _pick(quantiles2, K)
    quantiles3 = _pick(quantiles3, 0)
    q1, q2 = q

    t_range          = _tsource(df, end_ind=end_ind, stride=stride, plus=1)
    t_range_dro       = _tsource(df1, end_ind=end_ind_dro, stride=stride, plus=1)
    t_range_grad      = _tsource(df2, end_ind=end_ind_grad, stride=stride_grad, plus=1)
    t_range_dro_grad  = _tsource(df3, end_ind=end_ind_dro_grad, stride=stride_grad, plus=1)
    if t_range is None:
        t_range = t_range_dro
    if t_range is None:
        t_range = t_range_grad
    if t_range is None:
        t_range = t_range_dro_grad
    if t_range is None:
        print("plot_eval_compare: no dataframes available; skipping plot")
        return
    fig = plt.figure(figsize=(4.3, 2.1), dpi=300)

    # full
    if _want(series, "online"):
        _line(plt, df, 'O_eval1', j1, end_ind, stride, t_range, style='online', label="online clustering")
        _band(plt, quantiles, 'O_eval1', j1, end_ind, stride, t_range, 'online', alpha, q1, q2)
    if _want(series, "reclustering"):
        _line(plt, df, 'MRO_eval1', j2, end_ind, stride, t_range, style='recluster', label="reclustering")
        _band(plt, quantiles, 'MRO_eval1', j2, end_ind, stride, t_range, 'recluster', alpha, q1, q2)
    if _want(series, "DRO"):
        _line(plt, df1, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO")
        _band(plt, quantiles1, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2)
    if _want(series, "SAA"):
        _line(plt, df1, 'SA_eval2', j3, end_ind_dro, stride, t_range_dro, style='saa', label="SAA")
        _band(plt, quantiles1, 'SA_eval2', j3, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2)
    # subgrad overlay (no SAA) -- lighter shades, dashed.
    if _want(series, "online sgd"):
        _line(plt, df2, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad")
        _band(plt, quantiles2, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2)
    if _want(series, "reclustering sgd"):
        _line(plt, df2, 'MRO_eval1', j2g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad")
        _band(plt, quantiles2, 'MRO_eval1', j2g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2)
    if _want(series, "DRO sgd"):
        _line(plt, df3, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad")
        _band(plt, quantiles3, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2)
    if _want(series, "cluster SAA"):
        _line(plt, df, saa_eval, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA")
        _band(plt, quantiles, saa_eval, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2)
    _jstar_line(plt.gca(), jstar)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(ylim)
    if legend:
        plt.legend(ncol=legend_ncol)
    plt.xlabel(r'Time step $(t)$')
    plt.title(f'Out-of-sample expected value, $K$ = {K}')
    plt.grid(True, alpha=0.25, linewidth=0.4)
    plt.savefig(folderout + f'eval_analysis_compare{K}.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_cumulative_time(df_grad, df_grad_dro, j_grad=(0, 0, 0), K=15,
                         end_ind_grad=61, end_ind_dro_grad=None,
                         folderout=None):
    """Cumulative computation time of the ONLINE (subgradient) variants:
    clustered (K centroids, O(K) per step) vs full-sample (O(t) per step).

    Mean lines only: the aggregated frames hold per-seed means, and
    mean-of-cumsums == cumsum-of-means, so these lines are exact; the
    per-step quantile files cannot produce valid cumulative bands.
    """
    if end_ind_dro_grad is None:
        end_ind_dro_grad = end_ind_grad
    j1, _, j3 = j_grad
    df_grad = _pick(df_grad, K)
    df_grad_dro = _pick(df_grad_dro, 0)
    if df_grad is None or not _hc(df_grad, 'online_time'):
        print("plot_cumulative_time: no clustered online data; skipping plot")
        return
    fig = plt.figure(figsize=(4.3, 2.4), dpi=300)
    t_on = np.array(df_grad['t'][(j1 * end_ind_grad):(j1 + 1) * end_ind_grad])
    y_on = np.cumsum(np.array(
        df_grad['online_time'][(j1 * end_ind_grad):(j1 + 1) * end_ind_grad]))
    plt.plot(t_on[1:], y_on[1:], label=f'online clustering ($K$ = {K})',
             **_mkw('online_subgrad'))
    if df_grad_dro is not None and _hc(df_grad_dro, 'DRO_time'):
        t_dr = np.array(df_grad_dro['t'][
            (j3 * end_ind_dro_grad):(j3 + 1) * end_ind_dro_grad])
        y_dr = np.cumsum(np.array(df_grad_dro['DRO_time'][
            (j3 * end_ind_dro_grad):(j3 + 1) * end_ind_dro_grad]))
        plt.plot(t_dr[1:], y_dr[1:], label='online full data (DRO subgrad)',
                 **_mkw('dro_subgrad'))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Time step $(t)$')
    plt.title('Cumulative computation time (s)')
    plt.legend()
    plt.grid(True, alpha=0.25, linewidth=0.4)
    plt.savefig(folderout + f'time_cumulative{K}.pdf',
                bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_combined(
    df,  quantiles,  df1, quantiles1,
    df2, quantiles2, df3, quantiles3,
    end_ind=61,
    end_ind_grad=None,
    end_ind_dro=None,      # full DRO/SAA (df1) block length; falls back to end_ind if None
    end_ind_dro_grad=None, # subgrad DRO (df3) block length; falls back to end_ind_grad if None
    j=(0, 0, 0),
    j_grad=(0, 0, 0),
    stride=2,
    stride_grad=None,
    q=(40, 60),
    K=5,
    alpha=0.05,
    ylim=None,
    legend=True,
    saa_prefix='SAA',
    filename_suffix='',
    series=None,
    trim_start=None,
    trim_end=None,
    trim_start_grad=None,
    trim_end_grad=None,
    trim_start_regret=None,
    trim_end_regret=None,
    ylim_regret=None,
    yscale_log=False,
    yscale_log_out=False,
    jstar=None,
    folderout=None,
):
    """Single 2x3 summary figure consolidating the main comparison.

    ``jstar``, if given, draws a horizontal dotted reference line at the
    true-SAA oracle out-of-sample value on panel (b).

    Panels (times shown cumulative or per-solve depending on the baseline):
      (a) top-left     : in-sample objective value -- exact methods solid +
          subgrad variants dashed, log-x (``plot_eval_all_compare_eval`` ax2).
      (b) top-center   : out-of-sample expected value, same series
          (``plot_eval_all_compare_eval`` ax3).
      (c) top-right    : confidence $1-\\hat\\beta_t$, exact + subgrad
          overlays (``plot_confidence_regret``'s LHS panel).
      (d) bottom-left  : computation time per iteration, log-log, EXACT
          methods only (``plot_eval_all_compare_eval`` ax1; the online/
          subgrad variants' time story is panel (e) -- paper decision:
          per-iteration for exact, cumulative for online variants).
      (e) bottom-center: cumulative computation time, log-log, ONLINE
          (subgradient) variants only (``plot_cumulative_time``'s panel).
      (f) bottom-right : online-clustering dynamic regret, empirical vs
          theoretical upper bound (``plot_regret_new``'s panel).  Hidden
          (axis off) when the experiment has no regret data (e.g. svm).

    Parameters follow ``plot_eval_all_compare_eval`` (see its docstring for
    ``series``, trims, strides, block lengths): ``df``/``quantiles`` is the
    exact MRO set, ``df1``/``quantiles1`` exact DRO/SAA, ``df2``/
    ``quantiles2`` subgrad MRO, ``df3``/``quantiles3`` subgrad DRO.
    ``ylim`` (if given) applies to the two value panels (a)+(b);
    ``yscale_log`` toggles log-y on both value panels, ``yscale_log_out``
    on just (b).  ``trim_start_regret``/``trim_end_regret`` and
    ``ylim_regret`` affect only panel (f), as in ``plot_confidence_regret``/
    ``plot_regret_new``.  One shared frameless legend sits below the figure
    -- the union of every panel's handles, deduplicated by label and grouped
    into exact | subgrad | reference columns (``_grouped_legend_entries``).

    Every series degrades gracefully when its dataframe/columns are absent
    (``_pick``/``_hc`` guards), so a partially-run experiment still renders
    and gains the missing lines on a re-run once its data exists.

    Output filename: ``combined{K}{filename_suffix}.pdf``.
    """
    if end_ind_grad is None:
        end_ind_grad = end_ind
    if end_ind_dro is None:
        end_ind_dro = end_ind
    if end_ind_dro_grad is None:
        end_ind_dro_grad = end_ind_grad
    if stride_grad is None:
        stride_grad = stride
    if trim_start_grad is None:
        trim_start_grad = trim_start
    if trim_end_grad is None:
        trim_end_grad = trim_end
    trim = None if (trim_start is None and trim_end is None) else (trim_start, trim_end)
    trim_grad = None if (trim_start_grad is None and trim_end_grad is None) else (trim_start_grad, trim_end_grad)
    saa_time, saa_obj, saa_eval = (f'{saa_prefix}_time', f'{saa_prefix}_obj_values',
                                    f'{saa_prefix}_eval1')
    j1, j4, j3 = j
    j1g, j4g, j3g = j_grad
    df  = _pick(df, K)
    df1 = _pick(df1, 0)
    df2 = _pick(df2, K)
    df3 = _pick(df3, 0)
    quantiles  = _pick(quantiles, K)
    quantiles1 = _pick(quantiles1, 0)
    quantiles2 = _pick(quantiles2, K)
    quantiles3 = _pick(quantiles3, 0)
    q1, q2 = q

    t_range          = _tsource(df, end_ind=end_ind, stride=stride)
    t_range_dro       = _tsource(df1, end_ind=end_ind_dro, stride=stride)
    t_range_grad      = _tsource(df2, end_ind=end_ind_grad, stride=stride_grad)
    t_range_dro_grad  = _tsource(df3, end_ind=end_ind_dro_grad, stride=stride_grad)
    if t_range is None:
        t_range = t_range_dro
    if t_range is None:
        t_range = t_range_grad
    if t_range is None:
        t_range = t_range_dro_grad
    if t_range is None:
        print("plot_combined: no dataframes available; skipping plot")
        return
    trim_regret = None if (trim_start_regret is None and trim_end_regret is None) else (trim_start_regret, trim_end_regret)
    fig, ((ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f)) = plt.subplots(2, 3, figsize=(9, 3.5), dpi=300)

    # ============================================================
    # (a) in-sample objective value (plot_eval_all_compare_eval ax2)
    # ============================================================
    if _want(series, "online"):
        _line(ax_a, df, 'obj_values', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax_a, quantiles, 'obj_values', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        _line(ax_a, df, 'MRO_obj_values', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax_a, quantiles, 'MRO_obj_values', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        _line(ax_a, df1, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax_a, quantiles1, 'DRO_obj_values', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        _line(ax_a, df1, 'SA_obj_values', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax_a, quantiles1, 'SA_obj_values', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    if _want(series, "online sgd"):
        _line(ax_a, df2, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
        _band(ax_a, quantiles2, 'obj_values', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "reclustering sgd"):
        _line(ax_a, df2, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
        _band(ax_a, quantiles2, 'MRO_obj_values', j4g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "DRO sgd"):
        _line(ax_a, df3, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
        _band(ax_a, quantiles3, 'DRO_obj_values', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "cluster SAA"):
        _line(ax_a, df, saa_obj, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax_a, quantiles, saa_obj, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    _jstar_line(ax_a, jstar)
    ax_a.set_title(r'In-sample objective value')
    ax_a.grid(True, alpha=0.25, linewidth=0.4)
    ax_a.set_xscale("log")
    if yscale_log:
        ax_a.set_yscale("log")
    if ylim is not None:
        ax_a.set_ylim(ylim)

    # ============================================================
    # (b) out-of-sample expected value (plot_eval_all_compare_eval ax3)
    # ============================================================
    if _want(series, "online"):
        _line(ax_b, df, 'O_eval1', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax_b, quantiles, 'O_eval1', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        _line(ax_b, df, 'MRO_eval1', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax_b, quantiles, 'MRO_eval1', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        _line(ax_b, df1, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax_b, quantiles1, 'DRO_eval2', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        _line(ax_b, df1, 'SA_eval2', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax_b, quantiles1, 'SA_eval2', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    if _want(series, "online sgd"):
        _line(ax_b, df2, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
        _band(ax_b, quantiles2, 'O_eval1', j1g, end_ind_grad, stride_grad, t_range_grad, 'online_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "reclustering sgd"):
        _line(ax_b, df2, 'MRO_eval1', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
        _band(ax_b, quantiles2, 'MRO_eval1', j4g, end_ind_grad, stride_grad, t_range_grad, 'recluster_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "DRO sgd"):
        _line(ax_b, df3, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
        _band(ax_b, quantiles3, 'DRO_eval2', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, 'dro_subgrad', alpha, q1, q2, trim=trim_grad)
    if _want(series, "cluster SAA"):
        _line(ax_b, df, saa_eval, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax_b, quantiles, saa_eval, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    _jstar_line(ax_b, jstar)
    ax_b.set_title(r'Out-of-sample objective value')
    ax_b.set_xscale("log")
    ax_b.grid(True, alpha=0.25, linewidth=0.4)
    if yscale_log or yscale_log_out:
        ax_b.set_yscale("log")
    if ylim is not None:
        ax_b.set_ylim(ylim)
    # Panels (a) and (b) share the same y-axis (PI request): identical
    # scale and limits (union of both ranges, or the explicit ylim).
    if yscale_log_out and not yscale_log:
        if ax_a.get_ylim()[0] > 0:
            ax_a.set_yscale("log")
        else:
            ax_b.set_yscale("linear")
    _lo = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0])
    _hi = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
    ax_a.set_ylim(_lo, _hi)
    ax_b.set_ylim(_lo, _hi)

    # ============================================================
    # (c) confidence (plot_confidence_regret's LHS panel: lines only,
    #     exact + subgrad overlays)
    # ============================================================
    saa_satisfy = f'{saa_prefix}_satisfy1'
    if _want(series, "online"):
        _line(ax_c, df, 'O_satisfy0', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
    if _want(series, "reclustering"):
        _line(ax_c, df, 'MRO_satisfy0', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
    if _want(series, "DRO"):
        _line(ax_c, df1, 'DRO_satisfy1', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
    if _want(series, "SAA"):
        _line(ax_c, df1, 'SA_satisfy1', j3, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
    if _want(series, "online sgd"):
        _line(ax_c, df2, 'O_satisfy0', j1g, end_ind_grad, stride_grad, t_range_grad, style='online_subgrad', label="online clustering subgrad", trim=trim_grad)
    if _want(series, "reclustering sgd"):
        _line(ax_c, df2, 'MRO_satisfy0', j4g, end_ind_grad, stride_grad, t_range_grad, style='recluster_subgrad', label="reclustering subgrad", trim=trim_grad)
    if _want(series, "DRO sgd"):
        _line(ax_c, df3, 'DRO_satisfy1', j3g, end_ind_dro_grad, stride_grad, t_range_dro_grad, style='dro_subgrad', label="DRO subgrad", trim=trim_grad)
    if _want(series, "cluster SAA"):
        _line(ax_c, df, saa_satisfy, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
    ax_c.set_title(r'Confidence $1-\hat{\beta}_t$')
    ax_c.grid(True, alpha=0.25, linewidth=0.4)
    ax_c.set_xscale("log")

    # ============================================================
    # (d) computation time per iteration, EXACT methods only
    #     (plot_eval_all_compare_eval ax1)
    # ============================================================
    if _want(series, "online"):
        _line(ax_d, df, 'online_time', j1, end_ind, stride, t_range, style='online', label="online clustering", trim=trim)
        _band(ax_d, quantiles, 'online_time', j1, end_ind, stride, t_range, 'online', alpha, q1, q2, trim=trim)
    if _want(series, "reclustering"):
        _line(ax_d, df, 'MRO_time', j4, end_ind, stride, t_range, style='recluster', label="reclustering", trim=trim)
        _band(ax_d, quantiles, 'MRO_time', j4, end_ind, stride, t_range, 'recluster', alpha, q1, q2, trim=trim)
    if _want(series, "DRO"):
        _line(ax_d, df1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, style='dro', label="DRO", trim=trim)
        _band(ax_d, quantiles1, 'DRO_time', j3, end_ind_dro, stride, t_range_dro, 'dro', alpha, q1, q2, trim=trim)
    if _want(series, "SAA"):
        _line(ax_d, df1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, style='saa', label="SAA", trim=trim)
        _band(ax_d, quantiles1, 'SA_time', 0, end_ind_dro, stride, t_range_dro, 'saa', alpha, q1, q2, trim=trim)
    # Time panel: exact methods only (panel (e) covers the subgrad variants).
    if _want(series, "cluster SAA"):
        _line(ax_d, df, saa_time, j1, end_ind, stride, t_range, style='cluster_saa', label="cluster SAA", trim=trim)
        _band(ax_d, quantiles, saa_time, j1, end_ind, stride, t_range, 'cluster_saa', alpha, q1, q2, trim=trim)
    ax_d.set_xlabel(r'Time step $(t)$')
    ax_d.set_title(r'Per-iteration time (s), exact')
    ax_d.grid(True, alpha=0.25, linewidth=0.4)
    ax_d.set_yscale("log")
    ax_d.set_xscale("log")

    # ============================================================
    # (e) cumulative computation time, ONLINE (subgradient) variants only
    #     (plot_cumulative_time's panel: clustered K centroids, O(K) per
    #     step, vs full running sample, O(t) per step)
    # ============================================================
    # Mean lines only: the aggregated frames hold per-seed means, and
    # mean-of-cumsums == cumsum-of-means, so these lines are exact; the
    # per-step quantile files cannot produce valid cumulative bands.
    if df2 is not None and _hc(df2, 'online_time'):
        t_on = np.array(df2['t'][(j1g * end_ind_grad):(j1g + 1) * end_ind_grad])
        y_on = np.cumsum(np.array(
            df2['online_time'][(j1g * end_ind_grad):(j1g + 1) * end_ind_grad]))
        # Same label as the subgrad series in the value panels so the shared
        # legend deduplicates (same style key, same semantics).
        ax_e.plot(t_on[1:], y_on[1:], label='online clustering subgrad',
                  **_mkw('online_subgrad'))
    else:
        print("plot_combined: no clustered online data; cumulative-time "
              "panel drawn without it")
    if df3 is not None and _hc(df3, 'DRO_time'):
        t_dr = np.array(df3['t'][
            (j3g * end_ind_dro_grad):(j3g + 1) * end_ind_dro_grad])
        y_dr = np.cumsum(np.array(df3['DRO_time'][
            (j3g * end_ind_dro_grad):(j3g + 1) * end_ind_dro_grad]))
        ax_e.plot(t_dr[1:], y_dr[1:], label='DRO subgrad',
                  **_mkw('dro_subgrad'))
    ax_e.set_xlabel(r'Time step $(t)$')
    ax_e.set_title(r'Cumulative time (s), online')
    ax_e.set_xscale('log')
    ax_e.set_yscale('log')
    ax_e.grid(True, alpha=0.25, linewidth=0.4)

    # ============================================================
    # (f) online-clustering dynamic regret (plot_regret_new's panel:
    #     empirical regret vs theoretical upper bound, with bands).
    #     Hidden entirely when the experiment logged no regret data.
    # ============================================================
    t_range_regret = _tsource(df, end_ind=end_ind, offset=1)
    if t_range_regret is None:
        t_range_regret = _tsource(df1, end_ind=end_ind_dro, offset=1)

    if t_range_regret is not None:
        n = min(len(t_range_regret), int(end_ind_dro / 2))
        t_range_combo = t_range_regret[:n]

        def _trimmed(x, y):
            x, y = np.asarray(x), np.asarray(y)
            if trim_regret is not None:
                a, b = trim_regret
                x, y = x[a:b], y[a:b]
            return x, y

        if _hc(df, 'regret_bound') and _hc(df, 'sig_val'):
            y_ub = 5*df['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2] + np.array([5*np.sum(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1, int(end_ind/2)+1)])
            x_ub, y_ub = _trimmed(t_range_regret, y_ub)
            ax_f.plot(x_ub, y_ub, label="upper bound", **_mkw('bound'))

        if _hc(df, 'worst_values_regret') and _hc(df1, 'DRO_obj_values'):
            online_worst = np.array(df['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
            dro_obj = np.array(df1['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
            y_er = np.array([np.sum((online_worst-dro_obj)[:i+1])/(i) for i in range(1, n+1)])
            x_er, y_er = _trimmed(t_range_combo, y_er)
            ax_f.plot(x_er, y_er, label="empirical regret", **_mkw('online'))

        def _qhas(qd, *cols):
            return qd is not None and q1 in qd and q2 in qd and all(_hc(qd[q1], c) and _hc(qd[q2], c) for c in cols)

        if _qhas(quantiles, 'worst_values_regret') and _qhas(quantiles1, 'DRO_obj_values'):
            online_worst_q1 = np.array(quantiles[q1]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
            online_worst_q2 = np.array(quantiles[q2]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
            dro_obj_q1 = np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
            dro_obj_q2 = np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
            y1_er = np.array([np.sum((online_worst_q1-dro_obj_q1)[:i+1])/(i) for i in range(1, n+1)])
            y2_er = np.array([np.sum((online_worst_q2-dro_obj_q2)[:i+1])/(i) for i in range(1, n+1)])
            x_erb, y1_erb = _trimmed(t_range_combo, y1_er)
            _,     y2_erb = _trimmed(t_range_combo, y2_er)
            ax_f.fill_between(x_erb, y1=y1_erb, y2=y2_erb, **_bkw('online'))

        if _qhas(quantiles, 'regret_bound', 'sig_val'):
            y1_ub = np.array(5*quantiles[q1]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]) + np.array([5*np.sum(quantiles[q1]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1, int(end_ind/2)+1)])
            y2_ub = np.array(5*quantiles[q2]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]) + np.array([5*np.sum(quantiles[q2]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1, int(end_ind/2)+1)])
            x_ubb, y1_ubb = _trimmed(t_range_regret, y1_ub)
            _,     y2_ubb = _trimmed(t_range_regret, y2_ub)
            ax_f.fill_between(x_ubb, y1=y1_ubb, y2=y2_ubb, **_bkw('bound'))

    if ax_f.has_data():
        ax_f.set_xlabel(r'Time step $(t)$')
        ax_f.set_title(r'Online clustering dynamic regret')
        if ylim_regret is not None:
            ax_f.set_ylim(ylim_regret)
        ax_f.set_yscale('log')
        ax_f.set_xscale('log')
        ax_f.grid(True, alpha=0.25, linewidth=0.4)
    else:
        # No regret data in this experiment (e.g. svm logs no regret_bound /
        # worst_values_regret columns): hide the cell instead of showing an
        # empty frame.
        print("plot_combined: no regret data available; hiding regret panel")
        ax_f.axis('off')

    # ---- one shared legend below the whole figure: union of every panel's
    # handles, deduplicated by label and grouped into exact | subgrad |
    # reference columns (see _grouped_legend_entries) ------------------------
    handles, labels = [], []
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        hs, ls = ax.get_legend_handles_labels()
        handles += hs
        labels += ls
    if legend and handles:
        _grouped_fig_legend(fig, handles, labels)
    plt.tight_layout(h_pad=0.6)
    plt.savefig(folderout + 'combined' + str(K) + filename_suffix + '.pdf',
                bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_regret_new(df, quantiles, df1=None, quantiles1=None, end_ind=61, end_ind_dro=None,
                     j=(0, 0, 0), q=(40, 60), K=5, alpha=0.05, ylim=[0.008, 0.022],
                     folderout=None):
    j1, j2, j3 = j
    if end_ind_dro is None:
        end_ind_dro = end_ind
    df = _pick(df, K)
    df1 = _pick(df1, 0)
    quantiles = _pick(quantiles, K)
    quantiles1 = _pick(quantiles1, 0)
    q1, q2 = q
    t_range = _tsource(df, end_ind=end_ind, offset=1)
    if t_range is None:
        t_range = _tsource(df1, end_ind=end_ind_dro, offset=1)
    if t_range is None:
        print("plot_regret_new: no dataframes available; skipping plot")
        return
    fig = plt.figure(figsize=(4.3, 2.1), dpi=300)

    # df and df1 (DRO) may have differently-sized epsilon blocks; combined
    # (regret) computations below only make sense over the overlap, so they
    # are truncated to n -- the shorter of the two block lengths.
    n = min(len(t_range), int((end_ind_dro)/2))
    t_range_combo = t_range[:n]

    # online clustering regret (empirical line + band need both df and df1;
    # the theoretical upper-bound only needs df).
    if _hc(df, 'regret_bound') and _hc(df, 'sig_val'):
        plt.plot(t_range, 5*df['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2]+ np.array([5*np.sum(df['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]), label="upper bound", **_mkw('bound'))

    if _hc(df, 'worst_values_regret') and _hc(df1, 'DRO_obj_values'):
        online_worst = np.array(df['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
        dro_obj = np.array(df1['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
        plt.plot(t_range_combo, np.array([np.sum((online_worst-dro_obj)[:i+1])/(i) for i in range(1,n+1)]), label="empirical regret", **_mkw('online'))

    def _qhas(qd, *cols):
        return qd is not None and q1 in qd and q2 in qd and all(_hc(qd[q1], c) and _hc(qd[q2], c) for c in cols)

    if _qhas(quantiles, 'worst_values_regret') and _qhas(quantiles1, 'DRO_obj_values'):
        online_worst_q1 = np.array(quantiles[q1]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
        online_worst_q2 = np.array(quantiles[q2]['worst_values_regret'][(j1*end_ind+1):(j1+1)*end_ind:2])[:n]
        dro_obj_q1 = np.array(quantiles1[q1]['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
        dro_obj_q2 = np.array(quantiles1[q2]['DRO_obj_values'][(j3*end_ind_dro+1):(j3+1)*end_ind_dro:2])[:n]
        plt.fill_between(np.array(t_range_combo),y1=[np.sum((online_worst_q1-dro_obj_q1)[:i+1])/(i) for i in range(1,n+1)],y2=[np.sum((online_worst_q2-dro_obj_q2)[:i+1])/(i) for i in range(1,n+1)],**_bkw('online'))

    # theoretical bound band (needs only the online clustering quantiles)
    if _qhas(quantiles, 'regret_bound', 'sig_val'):
        plt.fill_between(np.array(t_range),y1=np.array(5*quantiles[q1]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+ np.array([5*np.sum(quantiles[q1]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in  range(1,int((end_ind)/2)+1)]) ,y2=np.array(5*quantiles[q2]['regret_bound'][(j1*end_ind+1):(j1+1)*end_ind:2])+np.array([5*np.sum(quantiles[q2]['sig_val'][(j1*end_ind+1):(j1+1)*end_ind:2][:i+1])/(i) for i in range(1,int((end_ind)/2)+1)]),**_bkw('bound'))

    plt.legend(ncol = 2)
    plt.xlabel(r'Time step $(t)$')
    plt.title(r'Online clustering dynamic regret')
    plt.ylim(ylim)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.25, linewidth=0.4)
    plt.savefig(folderout + 'regret_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def _seed_csvs(foldername, K):
    """All per-seed CSVs ``df_K{K}R{r}.csv`` under ``foldername``, sorted by
    seed number ``r``.  A glob (rather than a fixed ``range(R)`` loop) so that
    additional seeds from extension runs are picked up automatically."""
    rx = re.compile(rf'^df_K{K}R(\d+)\.csv$')
    found = []
    for path in glob.glob(os.path.join(foldername, f'df_K{K}R*.csv')):
        m = rx.match(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return [path for _, path in sorted(found)]


def setup_dfs(folderout=None, foldername=None, K_list=[0, 15, 25, 50], quant_list=[25, 75],
              R=None, init=False):
    """Aggregate per-seed result CSVs into mean + quantile frames.

    ``init=True`` re-aggregates from the per-seed files ``df_K{K}R{r}.csv``
    in ``foldername`` (discovered by glob, see ``_seed_csvs``; the mean and
    each quantile are taken across however many seed files exist) and caches
    ``df_K{K}.csv`` / ``quantiles_{q}K{K}.csv`` in ``folderout``.
    ``init=False`` reads those cached aggregates instead.

    ``R`` is accepted for backwards compatibility but ignored: the glob
    determines the seed set.
    """
    if init:
        quantiles = {}
        for K in K_list:
            dfs_list = []
            for csv_path in _seed_csvs(foldername, K):
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


def infer_end_ind(df_dict, K=None, t_col='t', default=61):
    if not df_dict:
        # No dataframes for this experiment set (e.g. folder not run yet).
        # Degrade gracefully so the plotting calls that do have data still run.
        print(f"infer_end_ind: no dataframes available; falling back to end_ind={default}")
        return default

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
