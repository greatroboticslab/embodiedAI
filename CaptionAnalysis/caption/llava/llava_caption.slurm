#!/bin/bash
#SBATCH --job-name=llava_caption_generation
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1    # Request 1 V100 32GB GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --output=../logs/llava_captions_%j.out
#SBATCH --error=../logs/llava_captions_%j.err

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate llava_caption

set -x

echo "Generating Llava captions..."
python llava_caption.py --src_dir "$1" --out_dir "$2"