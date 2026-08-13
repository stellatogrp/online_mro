# Shared environment for all slurm jobs. Sourced by every *.slurm script.
# Assumes the project lives at $PROJECT_DIR (set below) with a synced uv venv.

PROJECT_DIR=/scratch/gpfs/BSTELLATO/bs37/online_mro_paper
cd "$PROJECT_DIR"

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic

# One BLAS/OpenMP thread per process: joblib already runs R x M workers in
# parallel, and MOSEK is pinned to 1 thread via MSK_IPAR_NUM_THREADS.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY="$PROJECT_DIR/.venv/bin/python"
