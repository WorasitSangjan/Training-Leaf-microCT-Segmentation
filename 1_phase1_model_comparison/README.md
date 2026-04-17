# Phase 1 — Model Comparison

Compare 9 segmentation architectures on leaf microCT data under identical training conditions. The goal is to identify the top 2–3 models to carry forward into Phase 2 fine-tuning.

---

## Workflow

```
0_compute_class_weights.py        ←  compute per-class weights from training data
         │
         ▼
1_train_unet_resnet101.ipynb      ←  single-GPU walkthrough (read this first)
         │
         ▼
2_train_*.py  +  3_submit_*.sh    ←  train all 9 models on 4-GPU SLURM cluster
         │
         ▼
4_evaluate.py                     ←  evaluate each model on a separate test dataset (unseen during training)
```

---

## Models

| File | Architecture | Encoder | Type |
|------|-------------|---------|------|
| `2_train_unet_resnet101.py` | U-Net | ResNet-101 | CNN |
| `2_train_deeplab_efficientnet.py` | DeepLabV3+ | EfficientNet-B4 | CNN |
| `2_train_deeplab_mitb4.py` | DeepLabV3+ | MiT-B4 | Transformer encoder |
| `2_train_fpn_mitb4.py` | FPN | MiT-B4 | Transformer encoder |
| `2_train_fpn_mitb5.py` | FPN | MiT-B5 | Transformer encoder |
| `2_train_segformer.py` | SegFormer | MiT-B4 | Full transformer |
| `2_train_mask2former.py` | Mask2Former | Swin-B | Full transformer |
| `2_train_eomt_vitb.py` | EoMT | DINOv2 ViT-B | Full transformer |
| `2_train_eomt_vitl.py` | EoMT | DINOv2 ViT-L | Full transformer |

All models are trained under the same Phase 1 conditions:
- **Augmentation:** horizontal flip + vertical flip + rotate (±45°)
- **Loss:** weighted cross-entropy (inverse pixel frequency)
- **Optimizer:** AdamW, lr=1e-4, weight decay=5e-3
- **Patch size:** 320 × 320, stride 160
- **Epochs:** 100 with early stopping (patience=15)
- **Input:** single-channel grayscale (microCT)

---

## How to Run

### Option 1 — Single GPU (interactive)
Open `1_train_unet_resnet101.ipynb` in Jupyter. This notebook walks through the full pipeline step by step and can run on a single GPU.

### Option 2 — Multi-GPU on SLURM (NERSC Perlmutter)

Submit a training job:
```bash
sbatch 3_submit_train1.sh
```

Each submit script runs `torchrun` across 4 GPUs on a single node:
```bash
torchrun --standalone --nproc_per_node=4 2_train_unet_resnet101.py
```

> **Before running:** update the `SCRATCH` path in each training script to point to your storage location.

---

## Class Weights

Run once before training to compute per-class weights from the training data:

```bash
python 0_compute_class_weights.py
```

The weights reflect inverse pixel frequency across all datasets. The Phase 1 values used:

| Class | Index | Weight |
|-------|-------|--------|
| Background | 0 | 1.00 |
| Epidermis | 1 | 4.25 |
| Vascular Region | 2 | 3.15 |
| Mesophyll | 3 | 1.55 |
| Air Space | 4 | 4.95 |

---

## Evaluation

There are two evaluation stages:

**Stage 1 — Built-in (runs automatically at end of each training script)**
After training completes, each script automatically loads the best checkpoint and evaluates on the validation split (a held-out portion of the training dataset). Reports per-class IoU and Dice and saves to CSV. Used for early stopping and comparing models under the same data conditions.

**Stage 2 — `4_evaluate.py` (run manually after training)**
A separate evaluation on a completely independent **test dataset** (new leaf species/samples never seen during training). This gives an unbiased measure of how well each model generalises to new data.

```bash
# Evaluate with explicit test configs
python 4_evaluate.py \

  --model unet_resnet101 \
  --checkpoint /path/to/best_model.pth \
  --configs /path/to/configs/ttest1.json /path/to/configs/ttest2.json \
  --per_image

# Auto-discover all test configs (files starting with "t")
python 4_evaluate.py \
  --model eomt_vitl \
  --checkpoint /path/to/best_model.pth \
  --test_configs_dir /path/to/configs \
  --save_predictions /path/to/predictions \
  --per_image
```

Outputs:
- `eval_<model>_overall.csv` — per-class IoU, Dice, Precision, Recall
- `eval_<model>_per_image.csv` — per-image breakdown (with `--per_image`)
- `<save_predictions>/<dataset>/` — color overlay and label PNGs (with `--save_predictions`)

Supported model names: `unet_resnet101`, `deeplab_efficientnet`, `deeplab_mitb4`, `fpn_mitb4`, `fpn_mitb5`, `segformer`, `mask2former`, `eomt_vitb`, `eomt_vitl`
