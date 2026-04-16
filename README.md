# Training-Leaf-microCT-Segmentation

This repository documents the full training pipeline for leaf microCT segmentation — from raw dataset preparation through model comparison and fine-tuning. The best model produced from this pipeline powers the **[Leaf microCT Segmentation Web App](https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation)**.

---

## Overview

Leaf microCT images are segmented into 5 tissue classes:

| Index | Class |
|-------|-------|
| 0 | Background |
| 1 | Epidermis |
| 2 | Vascular Region |
| 3 | Mesophyll |
| 4 | Air Space |

The dataset covers **900+ leaf samples (25+ species)** across broadleaf and pine species. Most samples were scanned at the **X-ray μCT beamline (8.3.2) at the Advanced Light Source (ALS), Lawrence Berkeley National Laboratory (LBNL)**. Each lab produces masks with different pixel value conventions — the config-based dataset system handles this unification automatically.

---

## Pipeline

```
0_data_preparation/         ←  inspect raw data, onboard datasets, validate before training
         │
         ▼
1_phase1_model_comparison/  ←  train & compare 10 architectures under identical conditions
         │
         ▼
2_phase2_fine_tuning/       ←  fine-tune top 2–3 models  [coming soon]
         │
         ▼
3_phase3_specialized/       ←  specialized models per leaf type  [coming soon]
         │
         ▼
  WebApp deployment →  github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation
```

---

## Repository Structure

```
Training-Leaf-microCT-Segmentation/
│
├── 0_data_preparation/
│   ├── configs/                        ← example dataset config files
│   ├── 0_dataset_checklist.ipynb       ← how to onboard a new dataset
│   └── 1_check_image_sizes.py          ← validate all datasets before training
│
├── 1_phase1_model_comparison/
│   ├── 0_compute_class_weights.py      ← compute per-class weights
│   ├── 1_train_unet_resnet101.ipynb    ← single-GPU walkthrough (start here)
│   ├── 2_train_*.py                    ← 10 training scripts (multi-GPU, SLURM)
│   ├── 3_submit_*.sh                   ← SLURM job submission scripts
│   └── 4_evaluate.py                   ← evaluate on independent test dataset
│
├── 2_phase2_fine_tuning/               ← [coming soon]
│
└── 3_phase3_specialized/               ← [coming soon]
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- `segmentation-models-pytorch`
- `transformers` (HuggingFace)
- `albumentations`
- `tifffile`, `Pillow`, `numpy`, `pandas`, `tqdm`

**Multi-GPU training** requires a SLURM cluster with 4× NVIDIA GPUs (tested on NERSC Perlmutter). For single-GPU usage, follow the notebook in `1_phase1_model_comparison/`.

---

## Quick Start

### 1. Prepare your dataset
Follow `0_data_preparation/0_dataset_checklist.ipynb` to create a config file for your data, then run:
```bash
python 0_data_preparation/1_check_image_sizes.py
```

### 2. Compute class weights
```bash
python 1_phase1_model_comparison/0_compute_class_weights.py
```

### 3. Train (single GPU)
Open `1_phase1_model_comparison/1_train_unet_resnet101.ipynb` in Jupyter.

### 4. Train (multi-GPU, SLURM)
```bash
sbatch 1_phase1_model_comparison/3_submit_train1.sh
```

### 5. Evaluate on test dataset
```bash
python 1_phase1_model_comparison/4_evaluate.py \
  --model unet_resnet101 \
  --checkpoint /path/to/best_model.pth \
  --test_configs_dir /path/to/configs
```

---

## Related

- **Web App:** [WebApp-Leaf-microCT-Segmentation](https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation) — deploys the best model from this pipeline as an interactive segmentation tool
