#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=2G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio_output/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=1-2           # job array with index values 0, 1, 2, 3

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2024.2
conda activate mroenv

python portfolio_time/portMIP.py --foldername /scratch/gpfs/iywang/mro_results/portfolio_50/ --R 5 --T 2001 --fixed_time 1500 --interval 50 --Q 500 --K 15 --N_init 5 --r_start 25 --m 50

# python portfolio_time/portMIP.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 10 --T 10001 --fixed_time 7500  --interval 500 --Q 500 --K 5 --r_start 0 --m 50
