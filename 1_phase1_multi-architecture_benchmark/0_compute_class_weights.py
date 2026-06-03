"""
Compute segmentation class weights from dataset config files.

Use this before Phase 1 training when the dataset changes. By default, it
matches 1_train_multi_architecture.py: same config lists, same species-level
train/val split, and the same 3x pine repeat used for training.

Examples:
  python 0_compute_class_weights.py
  python 0_compute_class_weights.py --config-dir /pscratch/sd/w/worasit/configs
"""

import argparse
import ast
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

SCRATCH = "/pscratch/sd/w/worasit"
CONFIG_ROOT = f"{SCRATCH}/configs"
TRAIN_SCRIPT = Path(__file__).with_name("1_train_multi_architecture.py")
SEED = 42
VAL_FRAC = 0.2
PINE_REPEAT = 3
NUM_CLASSES = 5
IGNORE_INDEX = 254
CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


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
    if not paths:
        paths = phase1_train_configs(args.train_script, args.seed, args.val_frac, args.pine_repeat)
    return paths


def eval_config_node(node):
    if isinstance(node, ast.List):
        return [eval_config_node(item) for item in node.elts]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for item in node.values:
            if isinstance(item, ast.Constant):
                parts.append(str(item.value))
            elif isinstance(item, ast.FormattedValue) and isinstance(item.value, ast.Name):
                if item.value.id == "SCRATCH":
                    parts.append(SCRATCH)
                else:
                    raise ValueError(f"Unsupported formatted value: {item.value.id}")
            else:
                raise ValueError(f"Unsupported f-string node: {ast.dump(item)}")
        return "".join(parts)
    raise ValueError(f"Unsupported config list node: {ast.dump(node)}")


def read_training_config_lists(train_script):
    tree = ast.parse(Path(train_script).read_text())
    lists = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"BROADLEAF_CONFIGS", "PINE_CONFIGS"}:
                lists[target.id] = eval_config_node(node.value)
    missing = {"BROADLEAF_CONFIGS", "PINE_CONFIGS"} - set(lists)
    if missing:
        raise SystemExit(f"ERROR: could not read {', '.join(sorted(missing))} from {train_script}")
    return lists["BROADLEAF_CONFIGS"], lists["PINE_CONFIGS"]


def phase1_train_configs(train_script, seed, val_frac, pine_repeat):
    broadleaf_configs, pine_configs = read_training_config_lists(train_script)
    broadleaf = broadleaf_configs[:]
    pine = pine_configs[:]
    random.seed(seed)
    random.shuffle(broadleaf)
    random.shuffle(pine)

    broadleaf_val_n = int(len(broadleaf) * val_frac)
    pine_val_n = int(len(pine) * val_frac)
    broadleaf_train = broadleaf[broadleaf_val_n:]
    pine_train = pine[pine_val_n:]
    configs = broadleaf_train + pine_train * pine_repeat

    print("Using Phase 1 training split from 1_train_multi_architecture.py")
    print(f"  Broadleaf train configs: {len(broadleaf_train)}")
    print(f"  Pine train configs:      {len(pine_train)} x {pine_repeat}")
    print(f"  Total weighted configs:  {len(configs)}")
    return configs


def count_config_pixels(config_path, pixel_counts):
    with open(config_path) as f:
        cfg = json.load(f)

    name = cfg.get("name", Path(config_path).stem)
    mask_dir = cfg.get("mask_dir", "")
    if not os.path.isdir(mask_dir):
        print(f"  WARN: {name}: mask_dir not found: {mask_dir}")
        return 0

    mapping = {int(k): int(v) for k, v in cfg["mapping"].items()}
    ignore_index = cfg.get("ignore_index", IGNORE_INDEX)
    allowed_names = load_file_list(cfg.get("file_list"))
    total_valid = 0

    for mask_name in iter_image_files(mask_dir, allowed_names):
        mask = np.array(Image.open(os.path.join(mask_dir, mask_name)))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        remapped = np.full(mask.shape, ignore_index, dtype=np.int32)
        for gray_val, class_idx in mapping.items():
            remapped[mask == gray_val] = class_idx

        valid = remapped[remapped != ignore_index]
        total_valid += len(valid)
        for class_idx in range(NUM_CLASSES):
            pixel_counts[class_idx] += np.sum(valid == class_idx)

    return total_valid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", help="Custom config JSON path(s); overrides default Phase 1 split")
    parser.add_argument("--config-dir", default=None,
                        help=f"Custom directory of config JSON files; overrides default Phase 1 split, e.g. {CONFIG_ROOT}")
    parser.add_argument("--train-script", default=str(TRAIN_SCRIPT),
                        help="Phase 1 training script to read config lists from")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Species split seed; must match training")
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC,
                        help="Validation fraction; must match training")
    parser.add_argument("--pine-repeat", type=int, default=PINE_REPEAT,
                        help="Pine training repeat factor; must match training")
    parser.add_argument(
        "--method",
        choices=["inverse", "median"],
        default="inverse",
        help="Weighting method to print first",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_paths = config_paths_from_args(args)
    if not config_paths:
        raise SystemExit("ERROR: pass --configs or --config-dir")

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_valid = 0

    for config_path in tqdm(config_paths, desc="Scanning masks"):
        if not os.path.isfile(config_path):
            print(f"  WARN: config not found: {config_path}")
            continue
        total_valid += count_config_pixels(config_path, pixel_counts)

    if total_valid == 0:
        raise SystemExit("ERROR: no valid pixels found")

    freq = pixel_counts / total_valid
    present = pixel_counts > 0
    if not np.all(present):
        missing = ", ".join(CLASS_NAMES[idx] for idx in np.where(~present)[0])
        print(f"\nWARN: no pixels found for class(es): {missing}. Their suggested weight is set to 0.")

    inv_freq_norm = np.zeros(NUM_CLASSES, dtype=np.float64)
    inv_freq = 1.0 / freq[present]
    inv_freq_norm[present] = inv_freq / inv_freq.min()

    median_weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    median_freq = np.median(freq[present])
    median_weights[present] = median_freq / freq[present]

    primary = inv_freq_norm if args.method == "inverse" else median_weights

    print("\n" + "=" * 72)
    print("CLASS PIXEL STATISTICS")
    print("=" * 72)
    print(f"{'Class':<20} {'Pixels':>14} {'Frequency':>12}")
    print("-" * 72)
    for class_idx in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[class_idx]:<20} {pixel_counts[class_idx]:>14,} {freq[class_idx]:>12.6f}")
    print(f"{'Total valid':<20} {total_valid:>14,}")

    print("\n" + "=" * 72)
    print("INVERSE FREQUENCY WEIGHTS (normalized, min=1)")
    print("=" * 72)
    print(" ".join(f"{weight:.4f}" for weight in inv_freq_norm))

    print("\n" + "=" * 72)
    print("MEDIAN FREQUENCY WEIGHTS")
    print("=" * 72)
    print(" ".join(f"{weight:.4f}" for weight in median_weights))

    print("\n" + "=" * 72)
    print("TRAINING ARGUMENT")
    print("=" * 72)
    print("--class-weights " + " ".join(f"{weight:.4f}" for weight in primary))


if __name__ == "__main__":
    main()
