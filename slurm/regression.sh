#!/bin/bash
#SBATCH --job-name=regtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=3G
#SBATCH --time=10:00:00
#SBATCH -o /scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/reg_test_p2_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=0-2           # job array with index values 0, 1, 2, 3


cd "$SLURM_SUBMIT_DIR"

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic  # Direct MOSEK to the repo license

module purge
module load anaconda3/2024.2
conda activate lropt_rev

python regression/reg_orig_p1.py --foldername regression/results/p1/23/ --R 10 --T 2001 --fixed_time 2001 --interval 100 --Q 500 --K 10 --N_init 5 --r_start 0 --m 100 --noise 5 --rmse_mult 1.2 --k 5 --k_true 5 --power 0.03333

# python regression/reg_DRO_orig_p1.py --foldername regression/results/p1/23/ --R 5 --T 2001 --interval 100 --interval_SAA 100 --N_init 5 --r_start 5 --m 100 --noise 5 --k 5 --k_true 5

