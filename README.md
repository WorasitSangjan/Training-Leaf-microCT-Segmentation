# Training-Leaf-microCT-Segmentation

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-PyTorch%20%7C%20NERSC-orange.svg)](https://pytorch.org/)
![Science](https://img.shields.io/badge/Science-Plant%20Phenomics-green.svg)
![Research](https://img.shields.io/badge/Research-USDA--ARS-navy.svg)

This repository documents a reproducible training pipeline for cross-species leaf microCT segmentation — from raw dataset preparation through Phase 1 model comparison and Phase 2 fine-tuning. The four final Phase 2 models produced from this pipeline power the **[Leaf microCT Segmentation Web App](https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation)**.

---

## Overview

X-ray micro-computed tomography (microCT) provides a non-destructive view of internal leaf anatomy, but automated tissue-level segmentation remains difficult across species with contrasting anatomy, leaf morphology, and image acquisition conditions. This repository provides the training and evaluation code for a harmonized cross-species segmentation framework built from **1,147 annotated images from 52 species across 14 plant families**.

Leaf microCT images are segmented into 5 tissue classes:

| Index | Class |
|-------|-------|
| 0 | Background |
| 1 | Epidermis |
| 2 | Vascular Region |
| 3 | Mesophyll |
| 4 | Air Space |

Dataset-specific labels are reconciled into these five canonical classes with JSON configs, allowing data from different labs and acquisition settings to be evaluated under a shared label system. Most samples were scanned at the **X-ray μCT beamline (8.3.2) at the Advanced Light Source (ALS), Lawrence Berkeley National Laboratory (LBNL)**. Held-out test groups include broadleaf species, conifers, and wheat acquired on a different microCT instrument.

---

## Key Results

Phase 1 evaluated nine architectures spanning convolutional neural networks (CNNs), hybrid Transformer-CNNs, and query-based Transformer paradigms under identical settings. The convolutional U-Net baseline had the lowest overall mIoU (0.715), while the four models selected for Phase 2 all used Transformer-based encoders.

Fine-tuning changed the model ranking. Mask2Former/Swin-B ranked below four models in Phase 1, but achieved the largest fine-tuning gain (+0.044 mIoU) and the highest final performance.

| Final Phase 2 architecture | Overall mIoU |
|----------------------------|-------------:|
| Mask2Former / Swin-B | 0.809 |
| SegFormer-B4 | 0.808 |
| EoMT / DINOv3 ViT-L | 0.801 |
| FPN / MiT-B4 | 0.800 |

The main remaining limitation is segmentation of the Mesophyll/Vascular Region boundary in broadleaf samples. The trained models, web interface, Leaf CT Hub dataset resource, and source code are released as open resources for generalized plant microCT segmentation.

---

## Pipeline

```
0_data_preparation/         ←  inspect raw data, onboard datasets, validate before training
         │
         ▼
1_phase1_multi-architecture_benchmark/ ← train & compare 9 architectures under identical conditions
         │
         ▼
2_phase2_fine_tuning/       ←  train final best-mIoU fine-tuned model families
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
├── 2_phase2_fine_tuning/
│   ├── README.md                       ← fine-tuning workflow and best-result model table
│   ├── 1_train_mask2former.py          ← Mask2Former/Swin-B Tversky fine-tune
│   ├── 1_train_segformer.py            ← SegFormer-B4 Tversky fine-tune
│   ├── 1_train_fpn_mitb4.py            ← FPN/MiT-B4 Mesophyll-weight fine-tune
│   ├── 1_train_eomt_vitl.py            ← EoMT/ViT-L CLAHE pipeline fine-tune
│   ├── 1.1_SLURM_submit_training.sh    ← SLURM launcher for step 1
│   ├── 2_evaluate.py                   ← independent test evaluation
│   └── images/result_phase2.png        ← final fine-tuning summary figure
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

### 5. Train Phase 2 final candidates
Phase 2 scripts train the four final fine-tuned model families that produced the best held-out mIoU results. For example:

```bash
cd 2_phase2_fine_tuning
sbatch -J phase2_mask2former 1.1_SLURM_submit_training.sh mask2former
```

See `2_phase2_fine_tuning/README.md` for the best-result model table, output paths, evaluation notes, result figure, and links to use the final models.

---

## Documentation Notes

- Dataset composition and config format details are documented in `0_data_preparation/README.md`.
- Phase 1 model table, leakage-safe split details, and benchmark figure are documented in `1_phase1_multi-architecture_benchmark/README.md`.
- Phase 2 final fine-tuning recipes, result figure, model-use links, and SLURM entry points are documented in `2_phase2_fine_tuning/README.md`.

---

## Related

- **Leaf CT Hub:** [leafcthub.github.io](https://leafcthub.github.io/) — dataset database and corpus documentation.
- **Segmentation App:** [Hugging Face Space](https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation) — interactive application for using the final leaf microCT segmentation models.
- **Web App Source:** [WebApp-Leaf-microCT-Segmentation](https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation) — source code for the application, including access to the four final fine-tuned Phase 2 models.
