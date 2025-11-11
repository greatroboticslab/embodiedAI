#!/bin/bash
#SBATCH --job-name=minicpm_caption_generation # Job name
#SBATCH --partition=GPU-shared # Partition (queue) name
#SBATCH --gres=gpu:v100-32:1 # Request 1 A100 GPU
#SBATCH -N 1
#SBATCH --time=48:00:00 # Time limit (hh:mm:ss)
#SBATCH --output=../logs/minicpm_caption.out # Standard output
#SBATCH --error=../logs/minicpm_caption.err # Standard error

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate minicpm

set -x

echo "Generating MiniCPM captions..."
python minicpm_caption.py --src_dir $1 --out_dir $2