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


cd "$SLURM_SUBMIT_DIR"

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/BSTELLATO/iywang/low_rank/low-rank-dro/mosek/mosek.lic  # Direct MOSEK to the repo license

module purge
module load anaconda3/2024.2
conda activate lropt_rev

# python port_new/port_orig.py --foldername regression/results/p2/0/ --R 5 --T 2001 --fixed_time 2001 --interval 10 --Q 500 --K 15 --N_init 5 --r_start 5 --m 50

python port_new/port_DRO_orig.py --foldername regression/results/p2/0/ --R 2 --T 2001 --interval 10 --interval_SAA 10 --N_init 5 --r_start 0 --m 50


# python portfolio_new/port.py --foldername portfolio_new/results/ --R 10 --T 10000 --fixed_time 8500  --interval 500 --Q 500 --K 5 --r_start 20 --m 50 --N_init 5
