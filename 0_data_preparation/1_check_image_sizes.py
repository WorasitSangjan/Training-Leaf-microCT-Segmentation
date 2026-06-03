"""
Step 1: audit image sizes and image/mask resolution matches for all training/test configs.

This is the single pre-training check script:
  - Reports min/max/mean image dimensions per config
  - Flags images smaller than the selected patch-size threshold
  - Checks paired image/mask dimensions and reports mismatches

Use 2_prepare_datasets.py only after this audit identifies image/mask dimension
mismatches that should be repaired.

Usage:
    python 0_data_preparation/1_check_image_sizes.py
    python 0_data_preparation/1_check_image_sizes.py --min-size 320
    python 0_data_preparation/1_check_image_sizes.py --configs /path/to/a.json /path/to/b.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SCRATCH = "/pscratch/sd/w/worasit"
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

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


def iter_image_files(directory, allowed_names=None):
    """Yield non-hidden image files, optionally filtered by names from file_list."""
    allowed = set(allowed_names or [])
    allowed_stems = {Path(name).stem for name in allowed}
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if fname.startswith(".") or os.path.isdir(fpath):
            continue
        if Path(fname).suffix.lower() not in IMAGE_EXTS:
            continue
        if allowed and fname not in allowed and Path(fname).stem not in allowed_stems:
            continue
        yield fname


def load_file_list(path):
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"    WARN: file_list not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("files") or data.get("images") or data.get("filenames") or []
    return [str(item) for item in data]


def build_mask_index(mask_dir):
    if not mask_dir or not os.path.isdir(mask_dir):
        return {}
    masks = {}
    for fname in iter_image_files(mask_dir):
        if ".orig." in fname:
            continue
        masks[Path(fname).stem] = fname
    return masks


def format_examples(items, limit=2):
    examples = ", ".join(items[:limit])
    return examples + ("..." if len(items) > limit else "")


def check_configs(config_list, section_name, min_size):
    print(f"\n{'=' * 112}")
    print(f" {section_name} ({len(config_list)} configs)")
    print(f"{'=' * 112}")
    print(
        f"  {'Config':<28} {'N':>4} {'MinH':>6} {'MinW':>6} {'MaxH':>6} {'MaxW':>6} "
        f"{'MeanH':>7} {'MeanW':>7} {f'<{min_size}':>6} {'MaskMismatch':>12}  Examples"
    )
    print(
        f"  {'-' * 28} {'-' * 4} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} "
        f"{'-' * 7} {'-' * 7} {'-' * 6} {'-' * 12}  {'-' * 30}"
    )

    too_small = []
    mismatched = []
    all_sizes = []

    for cfg_path in config_list:
        if not os.path.exists(cfg_path):
            print(f"  MISSING: {cfg_path}")
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        name = cfg.get("name", Path(cfg_path).stem)
        image_dir = cfg.get("image_dir", "")
        mask_dir = cfg.get("mask_dir", "")
        allowed_names = load_file_list(cfg.get("file_list"))

        if not os.path.exists(image_dir):
            print(f"  {name:<28} image_dir not found")
            continue

        mask_index = build_mask_index(mask_dir)
        heights, widths, small_imgs, mismatch_examples = [], [], [], []

        for fname in iter_image_files(image_dir, allowed_names):
            img_path = os.path.join(image_dir, fname)
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as exc:
                print(f"    WARN: could not open image {img_path}: {exc}")
                continue

            heights.append(img_h)
            widths.append(img_w)
            if img_h < min_size or img_w < min_size:
                small_imgs.append(f"{fname}({img_h}x{img_w})")

            mask_name = mask_index.get(Path(fname).stem)
            if mask_dir and mask_name:
                mask_path = os.path.join(mask_dir, mask_name)
                try:
                    with Image.open(mask_path) as mask:
                        mask_w, mask_h = mask.size
                except Exception as exc:
                    print(f"    WARN: could not open mask {mask_path}: {exc}")
                    continue
                if (img_w, img_h) != (mask_w, mask_h):
                    mismatch_examples.append(
                        f"{fname}: img {img_w}x{img_h}, mask {mask_w}x{mask_h}"
                    )
            elif mask_dir:
                mismatch_examples.append(f"{fname}: missing mask")

        if not heights:
            print(f"  {name:<28} no images found")
            continue

        min_h = min(heights)
        min_w = min(widths)
        max_h = max(heights)
        max_w = max(widths)
        mean_h = np.mean(heights)
        mean_w = np.mean(widths)

        if small_imgs:
            too_small.append((name, min_h, min_w, small_imgs))
        if mismatch_examples:
            mismatched.append((name, mismatch_examples))

        all_sizes.append((min_h, min_w))
        examples = []
        if small_imgs:
            examples.extend(small_imgs[:2])
        if mismatch_examples:
            examples.extend(mismatch_examples[:2])

        print(
            f"  {name:<28} {len(heights):>4} {min_h:>6} {min_w:>6} {max_h:>6} {max_w:>6} "
            f"{mean_h:>7.0f} {mean_w:>7.0f} {len(small_imgs):>6} {len(mismatch_examples):>12}  "
            f"{format_examples(examples)}"
        )

    if all_sizes:
        overall_min_h = min(s[0] for s in all_sizes)
        overall_min_w = min(s[1] for s in all_sizes)
        print(f"\n  Overall min H={overall_min_h}, min W={overall_min_w}")
        print(f"  Safe PATCH_SIZE = {min(overall_min_h, overall_min_w)}")

    return too_small, mismatched


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-size",
        type=int,
        default=512,
        help="Flag images with either dimension below this size (default: 512)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Optional explicit config path(s). If omitted, the built-in train/test config groups are checked.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.configs:
        sections = [("CUSTOM CONFIGS", args.configs)]
    else:
        sections = [
            ("BROADLEAF TRAINING CONFIGS", BROADLEAF_CONFIGS),
            ("PINE TRAINING CONFIGS", PINE_CONFIGS),
            ("TEST CONFIGS", TEST_CONFIGS),
        ]

    all_small = []
    all_mismatched = []
    for section_name, configs in sections:
        small, mismatched = check_configs(configs, section_name, args.min_size)
        all_small.extend(small)
        all_mismatched.extend(mismatched)

    print(f"\n{'=' * 112}")
    print(f" SUMMARY")
    print(f"{'=' * 112}")
    print(f"  Configs with images < {args.min_size}px: {len(all_small)}")
    if all_small:
        for name, h, w, imgs in sorted(all_small, key=lambda x: min(x[1], x[2])):
            print(f"  {name:<30} min H={h}, min W={w} -> {len(imgs)} small image(s)")
            for img in imgs[:5]:
                print(f"    - {img}")
            if len(imgs) > 5:
                print(f"    ... {len(imgs) - 5} more")
    else:
        print(f"  All images >= {args.min_size}px")

    print(f"\n  Configs with image/mask dimension mismatches: {len(all_mismatched)}")
    if all_mismatched:
        for name, examples in all_mismatched:
            print(f"  {name:<30} {len(examples)} mismatch(es)")
            for example in examples[:5]:
                print(f"    - {example}")
            if len(examples) > 5:
                print(f"    ... {len(examples) - 5} more")
    else:
        print("  All matched image/mask pairs have identical dimensions")

    print(f"\n{'=' * 112}")
    print(" RECOMMENDATION")
    print(f"{'=' * 112}")
    print(f"  - Image-only size issues affect PATCH_SIZE selection; use --min-size to match your training patch.")
    print("  - Image/mask dimension mismatches should be fixed before training.")
    print("  - Use 2_prepare_datasets.py with the same config(s) from this audit to repair mismatched masks.")
    print("  - Missing masks, wrong file names, and semantic label problems still need manual review.")


if __name__ == "__main__":
    main()
