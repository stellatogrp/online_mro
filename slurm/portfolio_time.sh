#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/portfolio_test_p2_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#   SBATCH --array=0-1           # job array with index values 0, 1, 2, 3

cd "$SLURM_SUBMIT_DIR"

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic  # Direct MOSEK to the repo license
module purge
module load anaconda3/2024.2
conda activate lropt_rev

# python portfolio_time/portMIP.py --foldername portfolio_time/results/2/ --R 5 --T 2001 --fixed_time 2001 --interval 50 --Q 500 --K 15 --N_init 5 --r_start 0 --m 50 --rmse_mult 1.5

python portfolio_time/portMIP_DRO.py --foldername portfolio_time/results/0/ --R 5 --T 2001 --interval 100  --interval_SAA 100  --N_init 5 --r_start 5 --m 50
