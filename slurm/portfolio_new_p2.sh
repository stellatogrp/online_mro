#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Right-sized from profiling (see PROFILING.md): port_DRO_orig_p2 is a sparse LP
# (CLARABEL), peaks < ~0.5 GB/worker and finishes in minutes/task; the old
# 5G/24h request (175 GB!) is why this job queued for hours. For the port_p2
# (ours) line, use ~1G and 1:30:00 once the CSV-checkpoint fix is in.
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1:00:00
#SBATCH -o /scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/portfolio_test_p2_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
# SBATCH --array=0-1           # job array with index values 0, 1, 2, 3


cd "$SLURM_SUBMIT_DIR"

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic  # Direct MOSEK to the repo license

module purge
module load anaconda3/2024.2
conda activate lropt_rev

# Avoid BLAS/OpenMP oversubscription across the joblib workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# python port_new/port_orig_p2.py --foldername port_new/results/orig/p2/5/ --R 5 --T 2001 --fixed_time 2001 --interval 20 --Q 500 --K 15 --N_init 5 --r_start 0 --m 50 --rmse_mult 1.25  --cluster_interval 10

# python port_new/port_p2.py --foldername port_new/results/new/p2/2/ --R 5 --T 2001 --fixed_time 2001 --interval 1 --Q 500 --K 15 --N_init 5 --r_start 0 --m 50 --eta_0 0.1 --rmse_mult 1.25 --no-line_search --cluster_interval 1

# python port_new/port_orig_pca.py --foldername port_new/results/orig/p1/pca5/ --R 5 --T 2001 --fixed_time 2001 --interval 10 --Q 1000 --K 15 --N_init 5 --r_start 5 --m 150 --lr_var_frac 0.95 --pca_interval 50 --pca_init_interval 5 --pca_init_timesteps 200

# python port_new/port_DRO_p2.py --foldername port_new/results/new/p2/0/ --R 5 --T 2001 --interval 1 --interval_SAA 1 --N_init 5 --r_start 0 --m 150 --eta_0 0.1 --no-line_search

python port_new/port_DRO_orig_p2.py --foldername port_new/results/orig/p2/0/ --R 5 --T 2001 --interval 100 --interval_SAA 100 --N_init 5 --r_start 0 --m 150

# python port_new/port_DRO_orig_pca.py --foldername port_new/results/orig/p1/pca5/ --R 2 --T 2001 --interval 10 --interval_SAA 10 --N_init 5 --r_start 0 --m 150 --lr_var_frac 0.95 --pca_interval 100 --pca_init_interval 10 --pca_init_timesteps 200
