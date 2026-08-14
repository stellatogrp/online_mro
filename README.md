# Fast Online Distributionally Robust Optimization via Data Compression

This repository contains the numerical experiments for the paper:

> **[Fast Online Distributionally Robust Optimization via Data Compression](https://arxiv.org/abs/2504.08097)**
> [Irina Wang](https://sites.google.com/view/irina-wang/) and [Bartolomeo Stellato](https://stellato.io)

**Abstract.** We propose an online data compression approach for efficiently
solving Wasserstein distributionally robust optimization (DRO) problems with
streaming data. Our method constructs adaptive ambiguity sets around a
compressed distribution obtained by online clustering, so the problem size
stays fixed as data accumulate. We construct a dynamic regret bound with
respect to a one-step-ahead non-compressed DRO oracle, and establish online
clustering conditions such that, with high probability, the regret converges
sublinearly to a clustering-discrepancy-based performance gap. This gap is
defined in terms of the discrepancies between the true and compressed
distributions, so that by varying the number of clusters, our method trades
off robustness against computational effort. We additionally provide fast
subgradient-based updates that replace direct solutions of both the full and
compressed problems, and extend the regret analysis to this setting.
Numerical experiments on portfolio optimization and sparse support vector
machine problems, including mixed-integer formulations, show over an order of
magnitude reduction in cumulative computation time, even compared to
non-robust methods, with minimal loss in solution quality.

## Citation

```bibtex
@misc{wang2025fastonlinedro,
  title  = {Fast Online Distributionally Robust Optimization via Data Compression},
  author = {Wang, Irina and Stellato, Bartolomeo},
  year   = {2025},
  eprint = {2504.08097},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2504.08097},
}
```

## Experiments

Self-contained reproduction pipeline for the two paper experiments:

- **portfolio/** — portfolio CVaR under W1-DRO with box support (m=300 assets),
  exact SOCP (Clarabel) and projected-subgradient variants.
- **svm/** — sparse DRO-SVM on ijcnn1 (m=22 features, k=10 sparsity),
  exact MILP (MOSEK) and IHT/Q-step variants.

Each experiment compares: online-MRO, batch-MRO (re-clustering), cluster-SAA,
full-data DRO (stored as K=0), full-data SAA, and a true-SAA oracle (N=20000).

Every optimization formulation is unit-tested against independent CVXPY
models (`tests/`), and the fast worst-case kernel is validated against both
a reference implementation and an SOCP oracle.

## Setup

```
uv venv --python 3.12 && uv sync
```

MOSEK license: set `MOSEKLM_LICENSE_FILE` (needed for the svm experiment and
some tests; tests skip if unavailable).

## Run

```
uv run python -m portfolio.run --method mro --solver subgrad --results_dir portfolio/results/mro_subgrad
uv run python -m svm.run       --method dro --solver exact   --results_dir svm/results/dro_exact
```

`--method {mro,dro,true_saa} --solver {exact,subgrad}`; defaults (T=2001, R=10,
K=15, per-path epsilon lists and intervals) live in each experiment's
`config.py` and match the paper runs. `--eps_index I` runs a single epsilon;
`--r_start` offsets seeds so extra repetitions can be added additively.

Results: `{results_dir}/T{T-1}/df_K{K}R{r}.csv` per seed (DRO writes `K0`).

## Plots

```
uv run python -m portfolio.plots --results_dir portfolio/results --out_dir portfolio/plots --init
uv run python -m svm.plots       --results_dir svm/results       --out_dir svm/plots       --init
```

Aggregates per-seed CSVs (mean + 25/75 quantiles across all seeds found) and
writes the paper PDFs with the style in `plotting/paper_style.py`.

## Running on a SLURM cluster

The `slurm/` folder contains ready-made batch infrastructure: `env_setup.sh`
bootstraps a uv environment and builds the C kernel, `submit_all.sh` submits
the full experiment suite (per-job cpu/memory/time sizing in its table,
calibrated with `profiling/profile_task.py`), `job_array_seed.slurm` runs
one seed per array element, and `sync_results.sh` syncs code and results
between a workstation and the cluster. Adapt the host alias, account, and
scratch path at the top of `sync_results.sh` and `common.sh` to your site.

## Tests

```
uv run pytest                 # fast suite: formulations vs CVXPY oracles, smoke, aggregation
uv run pytest -m slow         # extended equivalence gate (auto-skips if reference code absent)
```

## Epsilon selection for figures

Plotted curves display, for each method independently, the epsilon selected
by: among sweep values whose end-of-horizon (last 10% of steps) certificate
confidence reaches at least 0.60, take the one minimizing the late-horizon
mean out-of-sample value; if no value reaches 0.60 confidence, take the one
with the highest confidence. Extension sweeps (`results_epsx` trees) are merged into the
selection automatically. The selection happens at plot time (`--tune_eps`,
default on) and the chosen
epsilon/value table is printed with each figure generation. The SAA baseline
is only solved at the first epsilon block (by construction), and cluster-SAA
shares the online method's block. `H*` reference lines come from the
`true_saa` oracle (SAA at N=20000).
