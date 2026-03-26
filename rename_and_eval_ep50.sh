#!/bin/bash
# Renames ep50 output folders and runs test evaluation on each best checkpoint.
# Run from: multi_model_segmentation/
# Usage:    bash rename_and_eval_ep50.sh

set -euo pipefail

OUTDIR="outputs"
MODELS=(unet unetpp unetv2 deeplabv3plus)

echo "=== Step 1: Renaming output folders to include _ep50 ==="
for MODEL in "${MODELS[@]}"; do
    SRC="${OUTDIR}/polyp_segmentation_baseline_${MODEL}"
    DST="${OUTDIR}/polyp_segmentation_baseline_${MODEL}_ep50"
    if [ -d "$SRC" ]; then
        mv "$SRC" "$DST"
        echo "  Renamed: $SRC -> $DST"
    elif [ -d "$DST" ]; then
        echo "  Already renamed: $DST (skipping)"
    else
        echo "  WARNING: $SRC not found, skipping."
    fi
done

echo ""
echo "=== Step 2: Running test evaluation for each model ==="
for MODEL in "${MODELS[@]}"; do
    CHECKPOINT="${OUTDIR}/polyp_segmentation_baseline_${MODEL}_ep50/best_model.pt"
    MODEL_CFG="configs/models/${MODEL}.yaml"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "  SKIPPING ${MODEL}: checkpoint not found at ${CHECKPOINT}"
        continue
    fi

    echo ""
    echo "--- Evaluating: ${MODEL} ---"
    python scripts/evaluate_segmentation.py \
        --config configs/base.yaml \
        --model-config "${MODEL_CFG}" \
        --checkpoint "${CHECKPOINT}" \
        --split test
done

echo ""
echo "=== Done. All evaluations complete. ==="
