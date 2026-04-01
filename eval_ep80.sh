#!/bin/bash
# Runs test evaluation on all completed ep80 best checkpoints.
# Usage: bash eval_ep80.sh

set -euo pipefail

OUTDIR="outputs"

declare -A MODEL_DIRS=(
    ["unet"]="${OUTDIR}/polyp_segmentation_baseline_unet_ep80"
    ["unetpp"]="${OUTDIR}/polyp_segmentation_baseline_unetpp_ep80"
    ["unetv2"]="${OUTDIR}/polyp_segmentation_baseline_unetv2"
    ["deeplabv3plus"]="${OUTDIR}/polyp_segmentation_baseline_deeplabv3plus_ep80"
)

declare -A MODEL_CFGS=(
    ["unet"]="configs/models/unet.yaml"
    ["unetpp"]="configs/models/unetpp.yaml"
    ["unetv2"]="configs/models/unetv2.yaml"
    ["deeplabv3plus"]="configs/models/deeplabv3plus.yaml"
)

echo "=== ep80 Test Evaluation ==="
for KEY in unet unetpp unetv2 deeplabv3plus; do
    DIR="${MODEL_DIRS[$KEY]}"
    CFG="${MODEL_CFGS[$KEY]}"
    CHECKPOINT="${DIR}/best_model.pt"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "  SKIPPING ${KEY}: checkpoint not found at ${CHECKPOINT}"
        continue
    fi

    echo ""
    echo "--- ${KEY} ---"
    python scripts/evaluate_segmentation.py \
        --config configs/base.yaml \
        --model-config "${CFG}" \
        --checkpoint "${CHECKPOINT}" \
        --split test
done

echo ""
echo "=== Done ==="
