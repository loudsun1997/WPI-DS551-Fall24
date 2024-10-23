#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100
#SBATCH -t 01:00:00
#SBATCH --mem 8g
#SBATCH --job-name="P3"

module load miniconda3/24.1.2/lqdppgt
module load cuda

eval "$(conda shell.bash hook)"
# source activate project3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate project33
python main.py --train_dqn