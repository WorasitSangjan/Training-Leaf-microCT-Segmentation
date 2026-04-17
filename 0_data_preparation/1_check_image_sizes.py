"""
Check image sizes across all dataset configs used in training + test sets.
Reports min/max/mean dimensions per config and flags images smaller than 512px.

Usage:
    python check_image_sizes.py
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

SCRATCH = "/pscratch/sd/w/worasit"

# ============================================================
# Exact configs from train_11_eomt_vitl.py (same for all 4 models)
# ============================================================
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
    f"{SCRATCH}/configs/st_pinus_pm2.json",
    f"{SCRATCH}/configs/st_pinus_pm5.json",
    f"{SCRATCH}/configs/st_pinus_pn1.json",
    f"{SCRATCH}/configs/st_pinus_pn2.json",
    f"{SCRATCH}/configs/st_pinus_ppd2.json",
    f"{SCRATCH}/configs/st_pinus_ppg1.json",
    f"{SCRATCH}/configs/st_pinus_ppg2.json",
    f"{SCRATCH}/configs/st_pinus_pr1.json",
    f"{SCRATCH}/configs/st_pinus_pr2.json",   
    f"{SCRATCH}/configs/st_pinus_pse1.json",
    f"{SCRATCH}/configs/st_pinus_pth5.json",
    f"{SCRATCH}/configs/st_pinus_tm10.json",
]

TEST_CONFIGS = [
    f"{SCRATCH}/configs/tab_vjucundum3.json",
    f"{SCRATCH}/configs/tjg_laca.json",
    f"{SCRATCH}/configs/tolive_r2.json",
    f"{SCRATCH}/configs/tst_pinus_aa.json",
    f"{SCRATCH}/configs/tst_pinus_pk.json",
    f"{SCRATCH}/configs/trh_wheat.json",
]

def check_configs(config_list, section_name):
    print(f"\n{'='*95}")
    print(f" {section_name} ({len(config_list)} configs)")
    print(f"{'='*95}")
    print(f"  {'Config':<28} {'N':>4} {'MinH':>6} {'MinW':>6} {'MaxH':>6} {'MaxW':>6} {'MeanH':>7} {'MeanW':>7} {'<512':>5} {'SmallImages'}")
    print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*30}")

    too_small = []
    all_sizes = []

    for cfg_path in config_list:
        if not os.path.exists(cfg_path):
            print(f"  MISSING: {cfg_path}")
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        name      = cfg.get("name", Path(cfg_path).stem)
        image_dir = cfg.get("image_dir", "")

        if not os.path.exists(image_dir):
            print(f"  {name:<28} image_dir not found")
            continue

        heights, widths, small_imgs = [], [], []
        for fname in sorted(os.listdir(image_dir)):
            fpath = os.path.join(image_dir, fname)
            if fname.startswith(".") or os.path.isdir(fpath):
                continue
            try:
                img = Image.open(fpath)
                w, h = img.size
                heights.append(h)
                widths.append(w)
                if h < 512 or w < 512:
                    small_imgs.append(f"{fname}({h}x{w})")
            except Exception:
                continue

        if not heights:
            print(f"  {name:<28} no images found")
            continue

        min_h  = min(heights)
        min_w  = min(widths)
        max_h  = max(heights)
        max_w  = max(widths)
        mean_h = np.mean(heights)
        mean_w = np.mean(widths)
        flag   = "YES" if small_imgs else ""

        if small_imgs:
            too_small.append((name, min_h, min_w, small_imgs))

        all_sizes.append((min_h, min_w))

        small_str = ", ".join(small_imgs[:2]) + ("..." if len(small_imgs) > 2 else "")
        print(f"  {name:<28} {len(heights):>4} {min_h:>6} {min_w:>6} {max_h:>6} {max_w:>6} "
              f"{mean_h:>7.0f} {mean_w:>7.0f} {flag:>5}  {small_str}")

    if all_sizes:
        overall_min_h = min(s[0] for s in all_sizes)
        overall_min_w = min(s[1] for s in all_sizes)
        print(f"\n  Overall min H={overall_min_h}, min W={overall_min_w}")
        print(f"  Safe PATCH_SIZE = {min(overall_min_h, overall_min_w)}")

    return too_small

# Run checks
small_broadleaf = check_configs(BROADLEAF_CONFIGS, "BROADLEAF TRAINING CONFIGS")
small_pine      = check_configs(PINE_CONFIGS,      "PINE TRAINING CONFIGS")
small_test      = check_configs(TEST_CONFIGS,      "TEST CONFIGS")

# Summary
all_small = small_broadleaf + small_pine + small_test
print(f"\n{'='*95}")
print(f" SUMMARY — Configs with images < 512px: {len(all_small)}")
print(f"{'='*95}")
if all_small:
    for name, h, w, imgs in sorted(all_small, key=lambda x: min(x[1], x[2])):
        print(f"  {name:<30} min H={h}, min W={w}  →  {len(imgs)} small image(s)")
        for img in imgs:
            print(f"    - {img}")
else:
    print("  All images >= 512px — safe to use PATCH_SIZE=512 for all configs")

print(f"\n{'='*95}")
print(" RECOMMENDATION")
print(f"{'='*95}")
print("  - Configs with min dim >= 512  → include in PATCH_SIZE=512 training")
print("  - Configs with min dim 320-511 → move to test set (evaluate with PATCH_SIZE=320)")
print("  - Configs with min dim < 320   → exclude entirely")
