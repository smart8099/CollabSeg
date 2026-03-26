#!/bin/bash
#SBATCH -J collabseg_eval
#SBATCH -p gpua30q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/home/abdulbasitmohammed01/multi_mode_segmentation/multi_model_segmentation/logs/collabseg_eval-%j.out
#SBATCH --error=/home/abdulbasitmohammed01/multi_mode_segmentation/multi_model_segmentation/logs/collabseg_eval-%j.err

set -euo pipefail
mkdir -p /home/abdulbasitmohammed01/multi_mode_segmentation/multi_model_segmentation/logs
echo "job_start=$(date +%Y%m%d-%H%M%S) job_id=${SLURM_JOB_ID} task=collabseg_eval"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate polypseg-py311

REPO_DIR="$HOME/multi_mode_segmentation/multi_model_segmentation"
cd "$REPO_DIR"

mkdir -p outputs/ensemble_eval

python scripts/evaluate_ensemble_batched.py \
    --ensemble-config configs/ensemble/default.yaml \
    --split-csv datasets/agentpolyp_2504/unified_split/manifests/test.csv \
    --batch-size 8 \
    --output-json outputs/ensemble_eval/test_summary_batched.json

echo "job_end=$(date +%Y%m%d-%H%M%S) job_id=${SLURM_JOB_ID} task=collabseg_eval"
