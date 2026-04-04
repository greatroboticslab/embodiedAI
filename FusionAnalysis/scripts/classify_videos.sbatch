#!/bin/bash
#SBATCH --job-name=gemini_classify
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=12:00:00
#SBATCH --output=classify_videos_%j.out
#SBATCH --error=classify_videos_%j.err
#SBATCH --account=cis240145p
#SBATCH --qos=low

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llava

cd /ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/scripts

# Run classification (replace API key before submitting)
python -u gemini_video_classify.py \
    --api_key "${GOOGLE_API_KEY}" \
    --model gemini-2.0-flash \
    --label both \
    --sleep 4
