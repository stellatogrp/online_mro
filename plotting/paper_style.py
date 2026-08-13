"""Paper (arxiv) matplotlib style for the online-MRO experiment figures.

Call :func:`set_paper_style` once, before any figure is created.  It sets a
serif (Computer Modern via usetex) style at paper font sizes, consistent with
a 10pt arxiv paper, and installs a color-blind-friendly color + linestyle
cycle for any *new* plotting code.

:data:`METHOD_STYLE` below is the single authoritative mapping from semantic
series keys to their visual style; the plot functions in
``plotting/plots_utils.py`` look every line, quantile band, and marker up
here (via :func:`method_kwargs` / :func:`band_kwargs`), so changing a
method's appearance in every figure means editing one entry in this dict.
"""
import matplotlib.pyplot as plt
from cycler import cycler

# Okabe-Ito palette: color-blind friendly, soft, publication quality.
PAPER_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

PAPER_LINESTYLES = ["-", "--", "-.", ":"]

# ---------------------------------------------------------------------------
# Authoritative per-series style map (Okabe-Ito hues).
#
# Pairing rule: each exact method and its subgradient variant share a hue
# family -- the subgrad line is dashed and a lighter tint of the same hue --
# AND the same marker shape, with the subgrad marker unfilled
# (``mfc='none'``) so the pairing survives grayscale printing.
# zorder ordering puts the series the paper argues about (online, DRO) on
# top; quantile bands are drawn below every line (see band_kwargs).
#
# Markers: every method series carries a distinct sparse marker (see
# ``method_kwargs``); ``markevery`` is an (offset, spacing) fraction pair --
# spacing is measured along the line in display space, so markers stay
# evenly spread on log-x axes, and the per-series offsets keep markers of
# different series from stacking at the same x.  Spacing is 0.15 for the
# exact methods and 0.25 for the subgrad variants, whose jagged traces
# accumulate arc length faster and would otherwise collect markers in the
# oscillating stretches.  Reference lines (``true_saa``/``bound``) and the
# ``resolve_marker`` vlines carry NO marker on purpose.
#
# Keys:
#   online / online_subgrad       online clustering (exact / subgradient)
#   recluster / recluster_subgrad reclustering MRO (exact / subgradient)
#   dro / dro_subgrad             full DRO (exact / subgradient)
#   saa                           sample average approximation
#   cluster_saa                   K-means-centroid SAA baseline
#   true_saa                      oracle reference (neutral dark, thin dashed)
#   bound                         theoretical regret upper bound
#   resolve_marker                full-resolve timestep vlines
# ---------------------------------------------------------------------------
METHOD_STYLE = {
    "online":            {"color": "#0072B2", "linestyle": "-",  "lw": 1.4, "band_alpha": 0.15, "zorder": 9,
                          "marker": "o", "markevery": (0.000, 0.15)},
    "online_subgrad":    {"color": "#56B4E9", "linestyle": "--", "lw": 1.2, "band_alpha": 0.15, "zorder": 6,
                          "marker": "o", "mfc": "none", "markevery": (0.020, 0.25)},
    "recluster":         {"color": "#D55E00", "linestyle": "-",  "lw": 1.4, "band_alpha": 0.15, "zorder": 7,
                          "marker": "s", "markevery": (0.040, 0.15)},
    "recluster_subgrad": {"color": "#EC9A64", "linestyle": "--", "lw": 1.2, "band_alpha": 0.15, "zorder": 5,
                          "marker": "s", "mfc": "none", "markevery": (0.060, 0.25)},
    "dro":               {"color": "#000000", "linestyle": "-",  "lw": 1.4, "band_alpha": 0.12, "zorder": 8,
                          "marker": "^", "markevery": (0.080, 0.15)},
    "dro_subgrad":       {"color": "#848484", "linestyle": "--", "lw": 1.2, "band_alpha": 0.12, "zorder": 4,
                          "marker": "^", "mfc": "none", "markevery": (0.100, 0.25)},
    "saa":               {"color": "#009E73", "linestyle": "-",  "lw": 1.4, "band_alpha": 0.15, "zorder": 3,
                          "marker": "D", "markevery": (0.120, 0.15)},
    "cluster_saa":       {"color": "#CC79A7", "linestyle": "-",  "lw": 1.4, "band_alpha": 0.15, "zorder": 3,
                          "marker": "v", "markevery": (0.140, 0.15)},
    "true_saa":          {"color": "#333333", "linestyle": "--", "lw": 1.0, "band_alpha": 0.0,  "zorder": 2},
    "bound":             {"color": "#E69F00", "linestyle": ":",  "lw": 1.4, "band_alpha": 0.15, "zorder": 5},
    "resolve_marker":    {"color": "#BBBBBB", "linestyle": ":",  "lw": 0.6, "band_alpha": 0.0,  "zorder": 1.5},
}


def method_kwargs(key):
    """Line kwargs (``ax.plot``) for the series ``key`` in METHOD_STYLE.

    Series with a ``marker`` entry additionally get sparse-marker kwargs:
    size ~4, edge in the series hue, face either the hue (exact methods)
    or ``'none'`` (subgrad variants, unfilled), and the (offset, spacing)
    ``markevery`` from the map.  Keys without ``marker`` (references,
    resolve vlines) emit no marker kwargs at all."""
    s = METHOD_STYLE[key]
    kw = {
        "color": s["color"],
        "linestyle": s["linestyle"],
        "linewidth": s["lw"],
        "zorder": s["zorder"],
    }
    if s.get("marker") is not None:
        kw.update({
            "marker": s["marker"],
            "markersize": s.get("markersize", 4),
            "markevery": s.get("markevery", 0.15),
            "markerfacecolor": s.get("mfc", s["color"]),
            "markeredgecolor": s["color"],
            "markeredgewidth": 0.9,
        })
    return kw


def band_kwargs(key):
    """Quantile-band kwargs (``ax.fill_between``) for the series ``key``.

    Same hue as the line, translucent, no edge line; band zorder is squeezed
    into (1, 2) so every band sits below every line while preserving the
    series' relative stacking among bands."""
    s = METHOD_STYLE[key]
    return {
        "color": s["color"],
        "alpha": s["band_alpha"],
        "linewidth": 0.0,
        "edgecolor": "none",
        "zorder": 1.0 + 0.01 * s["zorder"],
    }


def set_paper_style():
    """Set rcParams for paper figures (arxiv serif style, 10pt base)."""
    plt.rcParams.update({
        # LaTeX text rendering; serif math/text matching the arxiv template
        # (Computer Modern comes with usetex).
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        # Paper-scale font sizes (10pt document).
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Keep any non-usetex mathtext consistent with the serif body.
        "mathtext.fontset": "cm",
        # Color-blind-friendly cycle (fallback for any code that does not
        # look up METHOD_STYLE).
        "axes.prop_cycle": cycler(color=PAPER_COLORS),
        # Uncluttered legends and grids.
        "legend.frameon": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        # Output quality.
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": False,
        "lines.linewidth": 1.0,
    })
