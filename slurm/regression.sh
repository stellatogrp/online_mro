#!/bin/bash
#SBATCH --job-name=regtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Right-sized from profiling (see PROFILING.md): reg_dro peaks ~1.5 GB/worker and
# finishes in well under an hour/task; the old 4G/23h request just queued slowly.
# For the reg_orig_p2 (ours) line, 2G/2:00:00 is also comfortable.
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4:00:00
#SBATCH -o /scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/reg_test_p2_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#  SBATCH --array=0-2           # job array with index values 0, 1, 2, 3


cd "$SLURM_SUBMIT_DIR"

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic  # Direct MOSEK to the repo license

module purge
module load gurobi/13.0.0
module load anaconda3/2024.2
conda activate lropt_rev

# Avoid BLAS/OpenMP oversubscription: each joblib worker should use one core
# (MOSEK itself is pinned to 1 thread via MOSEK_PARAMS in regression/utils.py).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# python regression/reg_orig_p2.py --foldername regression/results/new/p2/4/ --R 10 --T 2001 --fixed_time 2001 --interval 100 --Q 500 --K 10 --N_init 5 --r_start 0 --m 50 --noise 3 --rmse_mult 1.1 --k 5 --k_true 5 --power 0.033 --p 2 --kappa 10

python regression/reg_DRO_orig_p2.py --foldername regression/results/new/p2/4/ --R 5 --T 2001 --interval 100 --interval_SAA 100 --N_init 5 --r_start 0 --m 50 --noise 3 --k 5 --k_true 5 --p 2 --power 0.25 --kappa 10

