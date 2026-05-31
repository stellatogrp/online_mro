#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=10G
#SBATCH --time=10:00:00
#SBATCH -o /scratch/gpfs/BSTELLATO/iywang/low_rank/online_mro/portfolio_test_p2_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=0-2           # job array with index values 0, 1, 2, 3


cd "$SLURM_SUBMIT_DIR"

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic  # Direct MOSEK to the repo license

module purge
module load anaconda3/2024.2
conda activate lropt_rev

python port_new/port_orig.py --foldername port_new/results/orig/p1/3/ --R 5 --T 2001 --fixed_time 2001 --interval 10 --Q 1000 --K 15 --N_init 5 --r_start 5 --m 50

# python port_new/port.py --foldername port_new/results/new/p1/14/ --R 5 --T 2001 --fixed_time 2001 --interval 1 --Q 1000 --K 15 --N_init 5 --r_start 0 --m 150 --eta_0 0.001

# python port_new/port_orig_pca.py --foldername port_new/results/orig/p1/pca5/ --R 5 --T 2001 --fixed_time 2001 --interval 10 --Q 1000 --K 15 --N_init 5 --r_start 5 --m 150 --lr_var_frac 0.95 --pca_interval 50 --pca_init_interval 5 --pca_init_timesteps 200

# python port_new/port_DRO.py --foldername port_new/results/new/p1/10/ --R 2 --T 2001 --interval 1 --N_init 5 --r_start 8 --m 150 --eta_0 1

# python port_new/port_DRO_orig.py --foldername port_new/results/orig/p1/4/ --R 2 --T 2001 --interval 10 --interval_SAA 10 --N_init 5 --r_start 0 --m 150

# python port_new/port_DRO_orig_pca.py --foldername port_new/results/orig/p1/pca5/ --R 2 --T 2001 --interval 10 --interval_SAA 10 --N_init 5 --r_start 0 --m 150 --lr_var_frac 0.95 --pca_interval 100 --pca_init_interval 10 --pca_init_timesteps 200

# python portfolio_new/port.py --foldername portfolio_new/results/ --R 10 --T 10000 --fixed_time 8500  --interval 500 --Q 500 --K 5 --r_start 20 --m 50 --N_init 5
