#!/bin/bash
# Run this ONCE before submitting any Slurm jobs.
# Creates the dedicated conda environment: polypseg-py311

set -euo pipefail

ENV_NAME="polypseg-py311"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -y -n "${ENV_NAME}" python=3.11
    echo "[INFO] Environment created."
fi

conda activate "${ENV_NAME}"

echo "[INFO] Installing packages..."
python -m pip install --upgrade pip
python -m pip install "numpy==1.26.4"
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

REPO_DIR="$HOME/multi_mode_segmentation/multi_model_segmentation"
cd "$REPO_DIR"
python -m pip install -r requirements.txt

echo "[INFO] Environment '${ENV_NAME}' is ready."
echo "[INFO] Run 'conda activate ${ENV_NAME}' before any local testing."
