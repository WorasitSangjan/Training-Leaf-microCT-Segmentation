"""
Compute class weights for Leaf CT segmentation training.

Two methods:
  1. Inverse frequency:       weight_c = 1 / freq_c  (normalized)
  2. Median frequency (SegNet): weight_c = median(freq) / freq_c

Run on login node or interactively — no GPU needed.
Usage: python compute_class_weights.py
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIG — match your training scripts exactly
# ============================================================
SCRATCH      = "/pscratch/sd/w/worasit"
IGNORE_INDEX = 254
NUM_CLASSES  = 5
CLASS_NAMES  = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]

BROADLEAF_CONFIGS = [
    f"{SCRATCH}/configs/ab_vcarlsii.json",
    f"{SCRATCH}/configs/ab_vcinnamo.json",
    f"{SCRATCH}/configs/ab_vdavidii.json",
    f"{SCRATCH}/configs/ab_vdavidii2.json",
    f"{SCRATCH}/configs/ab_vdentatum.json",
    f"{SCRATCH}/configs/ab_vdentatum2.json",
    f"{SCRATCH}/configs/ab_vfurcatum.json",
    f"{SCRATCH}/configs/ab_vhartwegii.json",
    f"{SCRATCH}/configs/ab_vjaponicum.json",
    f"{SCRATCH}/configs/ab_vjucundum.json",
    f"{SCRATCH}/configs/ab_vjucundum2.json",
    f"{SCRATCH}/configs/ab_vlantana.json",
    f"{SCRATCH}/configs/ab_vlautum.json",
    f"{SCRATCH}/configs/ab_vlautum3.json",
    f"{SCRATCH}/configs/ab_vpropinquum.json",
    f"{SCRATCH}/configs/ab_vtinus.json",
    f"{SCRATCH}/configs/devin1_no_bse.json",
    f"{SCRATCH}/configs/devin1_with_bse.json",
    f"{SCRATCH}/configs/devin2.json",
    f"{SCRATCH}/configs/devin3.json",
    f"{SCRATCH}/configs/jg_mag.json",
    f"{SCRATCH}/configs/lf_arab.json",
    f"{SCRATCH}/configs/oak_ce.json",
    f"{SCRATCH}/configs/oak_cf.json",
    f"{SCRATCH}/configs/oak_cr.json",
    f"{SCRATCH}/configs/oak_ob.json",
    f"{SCRATCH}/configs/oak_ru.json",
    f"{SCRATCH}/configs/oak_su.json",
    f"{SCRATCH}/configs/olive_d4.json",
    f"{SCRATCH}/configs/olive_d5.json",
    f"{SCRATCH}/configs/olive_r1.json",
    f"{SCRATCH}/configs/olive_w4.json",
    f"{SCRATCH}/configs/olive_w5.json",
]

PINE_CONFIGS = [
    f"{SCRATCH}/configs/st_pinus_lo1.json",
    f"{SCRATCH}/configs/st_pinus_lo2.json",
    f"{SCRATCH}/configs/st_pinus_palus.json",
    f"{SCRATCH}/configs/st_pinus_pb.json",
    f"{SCRATCH}/configs/st_pinus_pc.json",
    f"{SCRATCH}/configs/st_pinus_pd.json",
    f"{SCRATCH}/configs/st_pinus_pe1.json",
    f"{SCRATCH}/configs/st_pinus_pe2.json",
    f"{SCRATCH}/configs/st_pinus_pf1.json",
    f"{SCRATCH}/configs/st_pinus_pf3.json",
    f"{SCRATCH}/configs/st_pinus_pg.json",
    f"{SCRATCH}/configs/st_pinus_ph.json",
    f"{SCRATCH}/configs/st_pinus_pinaster.json",
    f"{SCRATCH}/configs/st_pinus_pinea.json",
    f"{SCRATCH}/configs/st_pinus_pj1.json",
    f"{SCRATCH}/configs/st_pinus_pj2.json",
    f"{SCRATCH}/configs/st_pinus_pk.json",
    f"{SCRATCH}/configs/st_pinus_pm2.json",
    f"{SCRATCH}/configs/st_pinus_pm5.json",
    f"{SCRATCH}/configs/st_pinus_pn1.json",
    f"{SCRATCH}/configs/st_pinus_pn2.json",
    f"{SCRATCH}/configs/st_pinus_ppd2.json",
    f"{SCRATCH}/configs/st_pinus_ppg1.json",
    f"{SCRATCH}/configs/st_pinus_ppg2.json",
    f"{SCRATCH}/configs/st_pinus_pr1.json",
    f"{SCRATCH}/configs/st_pinus_pse1.json",
    f"{SCRATCH}/configs/st_pinus_pth5.json",
    f"{SCRATCH}/configs/st_pinus_tm10.json",
]

# Pine repeated 3x to match training data balance
CONFIG_PATHS = BROADLEAF_CONFIGS + PINE_CONFIGS * 3

# ============================================================
# COUNT PIXELS PER CLASS
# ============================================================
pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
total_valid  = 0

for config_path in tqdm(CONFIG_PATHS, desc="Scanning datasets"):
    if not os.path.exists(config_path):
        continue

    with open(config_path) as f:
        cfg = json.load(f)

    mask_dir  = cfg["mask_dir"]
    mapping   = {int(k): int(v) for k, v in cfg["mapping"].items()}
    ignore    = cfg.get("ignore_index", 254)

    file_list_path = cfg.get("file_list", None)
    if file_list_path and os.path.exists(file_list_path):
        with open(file_list_path) as f:
            allowed = set(json.load(f))
        mask_files = sorted([fn for fn in os.listdir(mask_dir) if fn in allowed])
    else:
        mask_files = sorted([fn for fn in os.listdir(mask_dir) if not fn.startswith(".")])

    for mask_name in mask_files:
        mask = np.array(Image.open(os.path.join(mask_dir, mask_name)))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Remap to class indices
        remapped = np.full(mask.shape, ignore, dtype=np.int32)
        for gray_val, class_idx in mapping.items():
            remapped[mask == gray_val] = class_idx

        # Count valid pixels per class
        valid_mask = remapped[remapped != ignore]
        total_valid += len(valid_mask)
        for c in range(NUM_CLASSES):
            pixel_counts[c] += np.sum(valid_mask == c)

# ============================================================
# COMPUTE WEIGHTS
# ============================================================
freq = pixel_counts / total_valid  # fraction of total valid pixels

# Method 1: Inverse frequency (normalized so min weight = 1)
inv_freq = 1.0 / (freq + 1e-8)
inv_freq_norm = inv_freq / inv_freq.min()

# Method 2: Median frequency balancing (SegNet)
median_freq = np.median(freq)
median_freq_weights = median_freq / (freq + 1e-8)

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 60)
print("CLASS PIXEL STATISTICS")
print("=" * 60)
print(f"{'Class':<20} {'Pixels':>12} {'Frequency':>10}")
print("-" * 60)
for c in range(NUM_CLASSES):
    print(f"{CLASS_NAMES[c]:<20} {pixel_counts[c]:>12,} {freq[c]:>10.4f}")
print(f"{'Total valid':<20} {total_valid:>12,}")

print("\n" + "=" * 60)
print("METHOD 1: Inverse Frequency (normalized, min=1)")
print("=" * 60)
for c in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[c]:<20}: {inv_freq_norm[c]:.4f}")
print(f"\n  → torch.tensor([{', '.join(f'{w:.2f}' for w in inv_freq_norm)}])")

print("\n" + "=" * 60)
print("METHOD 2: Median Frequency Balancing (SegNet)")
print("=" * 60)
for c in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[c]:<20}: {median_freq_weights[c]:.4f}")
print(f"\n  → torch.tensor([{', '.join(f'{w:.2f}' for w in median_freq_weights)}])")

print("\n" + "=" * 60)
print("REFERENCE: V4 tuned weights")
print("=" * 60)
v4 = [0.5, 4.5, 4.0, 2.5, 2.5]
for c in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[c]:<20}: {v4[c]}")
print("=" * 60)
