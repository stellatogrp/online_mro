#!/bin/bash
# One-time environment bootstrap on della. Run on a login node:
#   bash slurm/env_setup.sh
set -euo pipefail

PROJECT_DIR=/scratch/gpfs/BSTELLATO/bs37/online_mro_paper
cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12
uv sync

# Sanity check: solvers importable, MOSEK license valid.
export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic
.venv/bin/python - <<'EOF'
import cvxpy as cp
x = cp.Variable(2, boolean=True)
p = cp.Problem(cp.Maximize(cp.sum(x)), [x <= 1])
p.solve(solver=cp.MOSEK)
assert p.value == 2.0
print("env OK: MOSEK licensed, solvers:", [s for s in cp.installed_solvers() if s in ("MOSEK", "CLARABEL")])
EOF

# Build the C worst-case kernel (falls back to numpy silently if absent).
sh portfolio/cworst/build.sh
