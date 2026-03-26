#!/bin/bash
#SBATCH -J polypseg_unet
#SBATCH -p gpua30q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/home/abdulbasitmohammed01/multi_mode_segmentation/multi_model_segmentation/logs/polypseg_unet-%j.out
#SBATCH --error=/home/abdulbasitmohammed01/multi_mode_segmentation/multi_model_segmentation/logs/polypseg_unet-%j.err

set -euo pipefail
mkdir -p /home/abdulbasitmohammed01/multi_mode_segmentation/multi_model_segmentation/logs
echo "job_start=$(date +%Y%m%d-%H%M%S) job_id=${SLURM_JOB_ID} model=unet"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate polypseg-py311

REPO_DIR="$HOME/multi_mode_segmentation/multi_model_segmentation"
cd "$REPO_DIR"

python scripts/train_segmentation.py \
    --config configs/base.yaml \
    --model-config configs/models/unet.yaml

echo "job_end=$(date +%Y%m%d-%H%M%S) job_id=${SLURM_JOB_ID} model=unet"
