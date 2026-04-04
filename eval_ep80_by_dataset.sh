#!/bin/bash
# Runs per-dataset test evaluation on all completed ep80 best checkpoints.
# Usage: bash eval_ep80_by_dataset.sh

set -euo pipefail

OUTDIR="outputs"
JSON_OUTDIR="${OUTDIR}/dataset_eval_ep80"

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

mkdir -p "${JSON_OUTDIR}"

echo "=== ep80 Per-Dataset Test Evaluation ==="
for KEY in unet unetpp unetv2 deeplabv3plus; do
    DIR="${MODEL_DIRS[$KEY]}"
    CFG="${MODEL_CFGS[$KEY]}"
    CHECKPOINT="${DIR}/best_model.pt"
    OUTPUT_JSON="${JSON_OUTDIR}/${KEY}_test_by_dataset.json"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "  SKIPPING ${KEY}: checkpoint not found at ${CHECKPOINT}"
        continue
    fi

    echo ""
    echo "--- ${KEY} ---"
    python scripts/evaluate_segmentation_by_dataset.py \
        --config configs/base.yaml \
        --model-config "${CFG}" \
        --checkpoint "${CHECKPOINT}" \
        --split test \
        --output-json "${OUTPUT_JSON}"
done

echo ""
echo "=== Done ==="
