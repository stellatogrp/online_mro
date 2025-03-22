#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=5G
#SBATCH --time=15:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio_output/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2024.2
conda activate mroenv

python portfolio_time/portMIP_DRO.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 5 --T 3001 --interval 100  --interval_SAA 100 --r_start 55

# python portfolio_time/portMIP.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 5 --T 10001 --fixed_time 7000 --interval 2500 --interval_online 100 --Q 2000 --K 5

# python portfolio_time/portMIP_DRO.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/ --R 20 --T 10001 --interval 20000 --r_start 20 --interval_SAA 100

