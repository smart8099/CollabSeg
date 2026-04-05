#!/bin/bash
# Dedicated evaluation script for the unetv2 baseline checkpoint at
# outputs/polyp_segmentation_baseline_unetv2.
#
# Runs both overall and per-dataset test evaluation, then writes JSON
# results to outputs/eval_unetv2/.
#
# Usage:
#   bash eval_unetv2.sh [--split val|test] [--save-predictions]
#
# Options:
#   --split val|test        Which split to evaluate (default: test)
#   --save-predictions      Also dump predicted mask PNGs under
#                           outputs/eval_unetv2/predictions/

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SPLIT="test"
SAVE_PREDS=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --save-predictions)
            SAVE_PREDS=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash eval_unetv2.sh [--split val|test] [--save-predictions]" >&2
            exit 1
            ;;
    esac
done

if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "ERROR: --split must be 'val' or 'test', got '${SPLIT}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR="outputs/polyp_segmentation_baseline_unetv2"
CHECKPOINT="${MODEL_DIR}/best_model.pt"
MODEL_CFG="configs/models/unetv2_custom.yaml"
BASE_CFG="configs/base.yaml"
OUT_DIR="outputs/eval_unetv2"
OVERALL_JSON="${OUT_DIR}/unetv2_${SPLIT}_overall.json"
BY_DATASET_JSON="${OUT_DIR}/unetv2_${SPLIT}_by_dataset.json"
PRED_DIR="${OUT_DIR}/predictions"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "$MODEL_CFG" ]; then
    echo "ERROR: model config not found at ${MODEL_CFG}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "========================================"
echo " UNetV2 Evaluation"
echo "========================================"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Model cfg  : ${MODEL_CFG}"
echo "  Split      : ${SPLIT}"
echo "  Output dir : ${OUT_DIR}"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Overall evaluation
# ---------------------------------------------------------------------------
echo ""
echo "--- Overall metrics ---"
python scripts/evaluate_segmentation.py \
    --config "${BASE_CFG}" \
    --model-config "${MODEL_CFG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    | tee /dev/stderr \
    > "${OVERALL_JSON}"

echo ""
echo "Saved overall metrics -> ${OVERALL_JSON}"

# ---------------------------------------------------------------------------
# 2. Per-dataset evaluation
# ---------------------------------------------------------------------------
echo ""
echo "--- Per-dataset metrics ---"
python scripts/evaluate_segmentation_by_dataset.py \
    --config "${BASE_CFG}" \
    --model-config "${MODEL_CFG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --output-json "${BY_DATASET_JSON}"

echo "Saved per-dataset metrics -> ${BY_DATASET_JSON}"

# ---------------------------------------------------------------------------
# 3. Optional: save predicted masks
# ---------------------------------------------------------------------------
if [[ "$SAVE_PREDS" -eq 1 ]]; then
    echo ""
    echo "--- Saving predicted masks ---"
    python scripts/evaluate_segmentation.py \
        --config "${BASE_CFG}" \
        --model-config "${MODEL_CFG}" \
        --checkpoint "${CHECKPOINT}" \
        --split "${SPLIT}" \
        --save-predictions-dir "${PRED_DIR}"
    echo "Saved predictions -> ${PRED_DIR}"
fi

echo ""
echo "=== Done ==="
