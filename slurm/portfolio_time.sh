#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=6G
#SBATCH --time=8:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio_output/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=1           # job array with index values 0, 1, 2, 3

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2024.2
conda activate mroenv

# python portfolio_time/portMIP.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 20 --T 3001 --fixed_time 2500 --interval 100 --Q 500 --K 15 --N_init 50 --r_start 40

python portfolio_time/portMIP.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 20 --T 10001 --fixed_time 7000  --interval 100 --Q 500 --K 5 --r_start 40 
