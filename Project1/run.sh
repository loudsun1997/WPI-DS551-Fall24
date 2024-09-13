#!/bin/bash
#SBATCH --job-name=age-estimation
#SBATCH --output=output/age-estimation-%j.out
#SBATCH --error=output/age-estimation-%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4  # Added: Request CPUs to support data loading and processing



module load python/3.12.2/mm75czi  
python -m venv rl_p1
source rl_p1/bin/activate 
pip install gymnasium numpy matplotlib nose imp pytest

python mdp_dp_test.py

