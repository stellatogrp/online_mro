#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio_output/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=0-3              # job array with index values 0, 1, 2, 3

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2024.2
conda activate mroenv

python portfolio_time/port_dfs.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 5 --T 3001 --fixed_time 2500 --interval 120 --interval_online 120 --Q 1000 --K 15

python portfolio_time/port_dfs.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 5 --T 2001 --fixed_time 1500 --interval 100 --interval_online 100 --Q 500 --K 15

# python portfolio_time/port_dfs.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 5 --T 10001 --fixed_time 7000 --interval 2500 --interval_online 100 --Q 2000 --K 5

# python portfolio/MIP/plots.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/new/m30_K1000_r10/
