Polyp segmentation training workspace

Model code:
- `src/polypseg/models`

Dataset split:
- `datasets/agentpolyp_2504/unified_split`

Training entrypoint:
- `scripts/train_segmentation.py`
- `scripts/evaluate_segmentation.py`
- `scripts/run_agentic_segmentation.py`
- `scripts/evaluate_ensemble.py`
- `scripts/evaluate_ensemble_batched.py`

Configs:
- `configs/base.yaml`
- `configs/models/unet.yaml`
- `configs/models/unetpp.yaml`
- `configs/models/unetv2.yaml`
- `configs/models/deeplabv3plus.yaml`
- `configs/ensemble/default.yaml`

Example:
```bash
python3 scripts/train_segmentation.py \
  --config configs/base.yaml \
  --model-config configs/models/unet.yaml
```

The training script expects the dataset manifests to point to files under `datasets/agentpolyp_2504/unified_split/files`.

Smoke run:
```bash
python3 scripts/train_segmentation.py \
  --config configs/base.yaml \
  --model-config configs/models/unet.yaml \
  --epochs 1 \
  --smoke-run-batches 2
```

Evaluation:
```bash
python3 scripts/evaluate_segmentation.py \
  --config configs/base.yaml \
  --model-config configs/models/unet.yaml \
  --checkpoint outputs/polyp_segmentation_baseline_unet/best_model.pt \
  --split test \
  --save-predictions-dir outputs/polyp_segmentation_baseline_unet/test_predictions
```

Agentic single-image inference:
```bash
python3 scripts/run_agentic_segmentation.py \
  --ensemble-config configs/ensemble/default.yaml \
  --image path/to/image.png \
  --prompt "segment the polyp near the center" \
  --output-mask outputs/agentic_mask.png
```

Ensemble evaluation:
```bash
python3 scripts/evaluate_ensemble.py \
  --ensemble-config configs/ensemble/default.yaml \
  --split-csv datasets/agentpolyp_2504/unified_split/manifests/test.csv \
  --output-json outputs/ensemble_eval/test_summary.json
```

Faster batched ensemble evaluation:
```bash
python3 scripts/evaluate_ensemble_batched.py \
  --ensemble-config configs/ensemble/default.yaml \
  --split-csv datasets/agentpolyp_2504/unified_split/manifests/test.csv \
  --batch-size 8 \
  --output-json outputs/ensemble_eval/test_summary_batched.json
```

The ensemble layer lives under `src/polypseg/ensemble` and does the following:
- loads multiple trained checkpoints from the registry config
- runs all candidate models on the same image
- scores each mask using confidence, agreement, shape sanity, boundary strength, and prompt hints
- applies a selection policy to choose one result or fuse the top masks
- reports selector performance against standalone models and oracle best-per-sample performance
