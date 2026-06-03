# Training-Leaf-microCT-Segmentation

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-PyTorch%20%7C%20NERSC-orange.svg)](https://pytorch.org/)
![Science](https://img.shields.io/badge/Science-Plant%20Phenomics-green.svg)
![Research](https://img.shields.io/badge/Research-USDA--ARS-navy.svg)

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
1_phase1_multi-architecture_benchmark/ ← train & compare 9 architectures under identical conditions
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
│   ├── 1_check_image_sizes.py          ← validate all datasets before training
│   └── 2_prepare_datasets.py           ← fix image/mask size mismatches
│
├── 1_phase1_multi-architecture_benchmark/
│   ├── README.md                       ← benchmark workflow and model list
│   ├── 0_compute_class_weights.py      ← calculate class imbalance weights
│   ├── 1_train_multi_architecture.py   ← unified trainer for all 9 models
│   ├── 2_evaluate.py                   ← independent test evaluation
│   ├── 1.1_SLURM_submit_training.sh    ← SLURM launcher for step 1
│   ├── 2.1_run_standard_tests.sh       ← standard evaluation helper
│   └── images/result_phase1.png        ← benchmark summary figure
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

**Multi-GPU training** requires a SLURM cluster with 4× NVIDIA GPUs (tested on NERSC Perlmutter).

---

## Quick Start

### 1. Prepare your dataset
Follow `0_data_preparation/0_dataset_checklist.ipynb` to create a config file for your data, then run:
```bash
python 0_data_preparation/1_check_image_sizes.py
```

If the audit reports image/mask dimension mismatches, dry-run the repair with:
```bash
python 0_data_preparation/2_prepare_datasets.py --configs /path/to/config.json
```

### 2. Compute class weights
When the training dataset changes, calculate class weights before training:

```bash
python 1_phase1_multi-architecture_benchmark/0_compute_class_weights.py
```

This uses the same species-level training split as Phase 1 training, so validation/test masks are not included in the default class-weight calculation.

### 3. Train (multi-GPU, SLURM)
```bash
sbatch -J phase1_eomt_vitl 1_phase1_multi-architecture_benchmark/1.1_SLURM_submit_training.sh eomt_vitl
```

Or launch directly:

```bash
torchrun --standalone --nproc_per_node=4 \
  1_phase1_multi-architecture_benchmark/1_train_multi_architecture.py \
  --model eomt_vitl
```

### 4. Evaluate on test dataset
```bash
python 1_phase1_multi-architecture_benchmark/2_evaluate.py \
  --model eomt_vitl \
  --checkpoint /path/to/best_model.pth \
  --test_configs_dir /path/to/configs
```

---

## Documentation Notes

- Dataset composition and config format details are documented in `0_data_preparation/README.md`.
- Phase 1 model table, leakage-safe split details, and benchmark figure are documented in `1_phase1_multi-architecture_benchmark/README.md`.

---

## Related

- **Leaf CT Hub:** [leafcthub.github.io](https://leafcthub.github.io/) — dataset database and corpus documentation.
- **Segmentation App:** [Hugging Face Space](https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation) — interactive leaf microCT segmentation application.
- **Web App Source:** [WebApp-Leaf-microCT-Segmentation](https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation) — source code for the segmentation application.
