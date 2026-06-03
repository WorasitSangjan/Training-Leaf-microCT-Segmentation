"""
Step 2: fix dataset issues found by the step 1 audit.

This script is intentionally general. It does not know about specific species,
labs, or downstream training tools. It reads the same JSON config format used by
training and fixes image/mask dimension mismatches for any dataset.

Run step 1 first:
    python 0_data_preparation/1_check_image_sizes.py --configs /path/to/config.json

Then dry-run the fix:
    python 0_data_preparation/2_prepare_datasets.py --configs /path/to/config.json

Apply after reviewing the dry-run:
    python 0_data_preparation/2_prepare_datasets.py --configs /path/to/config.json --apply

Restore mask backups:
    python 0_data_preparation/2_prepare_datasets.py --configs /path/to/config.json --restore
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRATCH = "/pscratch/sd/w/worasit"
CONFIG_ROOT = f"{SCRATCH}/configs"
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
PAD_VALUE = 0


def iter_image_files(directory, allowed_names=None):
    allowed = set(allowed_names or [])
    allowed_stems = {Path(name).stem for name in allowed}
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if fname.startswith(".") or os.path.isdir(fpath):
            continue
        if ".orig." in fname:
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
        print(f"  WARN: file_list not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("files") or data.get("images") or data.get("filenames") or []
    return [str(item) for item in data]


def config_paths_from_args(args):
    paths = []
    if args.configs:
        paths.extend(args.configs)
    if args.config_dir:
        for fname in sorted(os.listdir(args.config_dir)):
            if fname.endswith(".json"):
                paths.append(os.path.join(args.config_dir, fname))
    return paths


def backup_path(mask_path):
    base, ext = os.path.splitext(mask_path)
    return f"{base}.orig{ext}"


def build_mask_index(mask_dir):
    if not os.path.isdir(mask_dir):
        return {}
    return {Path(fname).stem: fname for fname in iter_image_files(mask_dir)}


def load_pairs(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    name = cfg.get("name", Path(config_path).stem)
    image_dir = cfg.get("image_dir", "")
    mask_dir = cfg.get("mask_dir", "")
    allowed_names = load_file_list(cfg.get("file_list"))

    if not os.path.isdir(image_dir):
        print(f"  WARN: {name}: image_dir not found: {image_dir}")
        return name, []
    if not os.path.isdir(mask_dir):
        print(f"  WARN: {name}: mask_dir not found: {mask_dir}")
        return name, []

    mask_index = build_mask_index(mask_dir)
    pairs = []
    for img_name in iter_image_files(image_dir, allowed_names):
        mask_name = mask_index.get(Path(img_name).stem)
        if not mask_name:
            print(f"  WARN: {name}: missing mask for {img_name}")
            continue
        pairs.append((
            os.path.join(image_dir, img_name),
            os.path.join(mask_dir, mask_name),
        ))
    return name, pairs


def choose_method(method, img_size, mask_size):
    if method != "auto":
        return method

    img_w, img_h = img_size
    mask_w, mask_h = mask_size
    if mask_w == 0 or mask_h == 0:
        return "resize"

    width_scale = img_w / mask_w
    height_scale = img_h / mask_h
    scales_match = abs(width_scale - height_scale) < 0.02

    if scales_match:
        return "resize"
    return "pad-crop"


def fix_mask_array(mask, img_size, method):
    img_w, img_h = img_size
    if method == "resize":
        return mask.resize((img_w, img_h), Image.NEAREST)

    mask_arr = np.array(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]

    fixed = np.full((img_h, img_w), PAD_VALUE, dtype=mask_arr.dtype)
    paste_h = min(img_h, mask_arr.shape[0])
    paste_w = min(img_w, mask_arr.shape[1])
    fixed[:paste_h, :paste_w] = mask_arr[:paste_h, :paste_w]
    return Image.fromarray(fixed, mode="L")


def fix_mismatches(args):
    config_paths = config_paths_from_args(args)
    if not config_paths:
        print("ERROR: pass --configs or --config-dir")
        sys.exit(1)

    total_mismatches = 0
    total_fixed = 0

    for config_path in config_paths:
        if not os.path.isfile(config_path):
            print(f"WARN: config not found: {config_path}")
            continue

        name, pairs = load_pairs(config_path)
        mismatches = []
        for img_path, mask_path in pairs:
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        mismatches.append((img_path, mask_path, img.size, mask.size))
            except Exception as exc:
                print(f"  WARN: {name}: could not inspect {img_path} / {mask_path}: {exc}")

        if not mismatches:
            print(f"{name}: no image/mask dimension mismatches")
            continue

        total_mismatches += len(mismatches)
        print(f"\n{name}: {len(mismatches)} mismatch(es)")

        for img_path, mask_path, img_size, mask_size in mismatches:
            selected_method = choose_method(args.method, img_size, mask_size)
            backup = backup_path(mask_path)
            summary = (
                f"  {os.path.basename(mask_path):<35} "
                f"mask {mask_size[0]}x{mask_size[1]} -> image {img_size[0]}x{img_size[1]} "
                f"via {selected_method}"
            )

            if os.path.exists(backup) and not args.overwrite_backup:
                print(f"{summary} [SKIP: backup exists]")
                continue

            if not args.apply:
                print(f"{summary} [would fix]")
                continue

            if not os.path.exists(backup) or args.overwrite_backup:
                shutil.copy2(mask_path, backup)

            with Image.open(mask_path) as mask:
                fixed = fix_mask_array(mask, img_size, selected_method)
                fixed.save(mask_path)

            with Image.open(img_path) as img, Image.open(mask_path) as fixed_mask:
                ok = img.size == fixed_mask.size
            print(f"{summary} [{'OK' if ok else 'FAILED'}]")
            total_fixed += int(ok)

    print(f"\nFound {total_mismatches} mismatch(es).")
    if args.apply:
        print(f"Fixed {total_fixed} mask(s). Backups use the suffix .orig before the extension.")
    else:
        print("Dry run only. Re-run with --apply to modify masks.")


def restore_backups(args):
    config_paths = config_paths_from_args(args)
    if not config_paths:
        print("ERROR: pass --configs or --config-dir")
        sys.exit(1)

    restored = 0
    seen_mask_dirs = set()
    for config_path in config_paths:
        if not os.path.isfile(config_path):
            print(f"WARN: config not found: {config_path}")
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        mask_dir = cfg.get("mask_dir", "")
        if not os.path.isdir(mask_dir) or mask_dir in seen_mask_dirs:
            continue
        seen_mask_dirs.add(mask_dir)

        for fname in sorted(os.listdir(mask_dir)):
            if ".orig." not in fname:
                continue
            src = os.path.join(mask_dir, fname)
            dst = os.path.join(mask_dir, fname.replace(".orig", "", 1))
            if not args.apply:
                print(f"  {src} -> {dst} [would restore]")
                restored += 1
                continue
            shutil.copy2(src, dst)
            os.remove(src)
            print(f"  {src} -> {dst} [restored]")
            restored += 1

    print(f"\n{'Restored' if args.apply else 'Would restore'} {restored} backup(s).")
    if not args.apply:
        print("Dry run only. Re-run with --apply --restore to restore backups.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="General step 2 fixer for image/mask dimension mismatches."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Config JSON path(s) to repair. Use the same configs audited by step 1.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help=f"Optional directory of config JSON files (default example root: {CONFIG_ROOT})",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "resize", "pad-crop"],
        default="auto",
        help=(
            "How to make masks match image dimensions. "
            "auto resizes proportional resolution differences and pad/crops non-proportional differences."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Actually modify files; default is dry-run")
    parser.add_argument("--restore", action="store_true", help="Restore masks from .orig backups")
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Replace existing .orig backups when applying fixes",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.restore:
        restore_backups(args)
    else:
        fix_mismatches(args)


if __name__ == "__main__":
    main()
