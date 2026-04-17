# Data Preparation

This folder covers everything needed before training: how to onboard a new dataset and how to validate all datasets are compatible with the training pipeline.

---

## Workflow

```
New dataset arrives
       │
       ▼
0_dataset_checklist.ipynb   ←  inspect images/masks, create config file
       │
       ▼
1_check_image_sizes.py      ←  validate all configs, confirm safe PATCH_SIZE
       │
       ▼
   Ready to train
```

---

## Files

### `0_dataset_checklist.ipynb`
Step-by-step guide for onboarding a new dataset from a new lab or scan facility.

Covers:
- Checking image dimensions and channel format (grayscale vs. RGB)
- Inspecting mask pixel values (they differ per lab)
- Mapping lab-specific pixel values to canonical class indices
- Handling `ignore_index` conflicts
- Creating a JSON config file for the new dataset

### `1_check_image_sizes.py`
Run this across all configs before starting training. Reports min/max/mean image dimensions per dataset and flags any images smaller than 512px.

```bash
python 1_check_image_sizes.py
```

Output:
- Per-dataset size summary
- Safe `PATCH_SIZE` recommendation
- List of configs with images below threshold

---

## Config Files (`configs/`)

Each dataset is described by a JSON config file. The training pipeline loads these configs to locate images/masks and remap pixel values to canonical class indices.

### Format

```json
{
  "name": "dataset_name",
  "image_dir": "/path/to/images",
  "mask_dir":  "/path/to/masks",
  "file_list": "/path/to/file_list.json",
  "num_classes": 5,
  "ignore_index": 254,
  "class_names": ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"],
  "mapping": {
    "PIXEL_VALUE": CLASS_INDEX
  }
}
```

| Field | Description |
|-------|-------------|
| `name` | Human-readable dataset identifier |
| `image_dir` / `mask_dir` | Absolute paths to image and mask folders |
| `file_list` | *(Optional)* JSON list of allowed filenames — used to filter a subset |
| `mapping` | Remaps lab-specific pixel values to canonical class indices |
| `ignore_index` | Pixels with this value are excluded from loss and metrics (default: 254) |

### Canonical Class Indices

| Index | Class | Note |
|-------|-------|------|
| 0 | Background | |
| 1 | Epidermis | |
| 2 | Vascular Region | |
| 3 | Mesophyll | |
| 4 | Air Space | |
| 254 | Ignored | unlabelled or ambiguous pixels |

> **Important:** The pixel value in the mask varies between labs and scan facilities. The canonical index never changes. The config `mapping` bridges the two.

### Examples

**Broadleaf** — `configs/devin1_no_bse.json`
```json
{
  "name": "devin1_no_bse",
  "num_classes": 5,
  "mapping": {
    "170": 0,
    "85":  1,
    "180": 2,
    "0":   3,
    "255": 4,
    "152": 254
  }
}
```

**Pine** — `configs/st_pinus_pf1.json`
```json
{
  "name": "st_pinus_pf1",
  "num_classes": 5,
  "has_resin_ducts": true,
  "has_transfusion_tissue": true,
  "mapping": {
    "85":  0,
    "170": 1,
    "103": 2,
    "127": 2,
    "0":   3,
    "255": 4,
    "200": 254
  }
}
```

Note that pine configs can map multiple pixel values to the same class index (e.g. `103` and `127` both map to class 2 — Vascular Region), and use extra metadata fields like `has_resin_ducts` to document dataset-specific anatomy.
