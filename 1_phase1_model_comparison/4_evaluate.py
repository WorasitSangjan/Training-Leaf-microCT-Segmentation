"""
Standalone Evaluation + Inference Script — Leaf CT Segmentation
================================================================
Evaluates a trained model checkpoint on a held-out test set and
optionally saves predicted segmentation mask images.

Model options:
  CNN       : unet_resnet101, deeplab_efficientnet, deeplab_mitb4
  Transformer: fpn_mitb4, fpn_mitb5, segformer, mask2former
  DINOv2    : eomt_vitb, eomt_vitl

Usage:

  # Evaluate with explicit config files
  python evaluate.py \
    --model unet_resnet101 \
    --checkpoint /pscratch/sd/w/worasit/outputs/models_UNet_ResNet101/best_model.pth \
    --configs /pscratch/sd/w/worasit/configs/tab_vjucundum3.json \
              /pscratch/sd/w/worasit/configs/tjg_laca.json \
              /pscratch/sd/w/worasit/configs/tolive_r2.json \
    --per_image

  # Auto-discover all t*.json configs in a directory
  python evaluate.py \
    --model eomt_vitl \
    --checkpoint /pscratch/sd/w/worasit/outputs/Pharse/Phase_9_v5/models_EoMT_ViTL/best_model.pth \
    --test_configs_dir /pscratch/sd/w/worasit/configs \
    --save_predictions /pscratch/sd/w/worasit/predictions/eomt_vitl \
    --per_image

Outputs (saved next to checkpoint unless --output_dir is set):
  eval_<model>_overall.csv   — per-class IoU/Dice/Precision/Recall + mean
  eval_<model>_per_image.csv — per-image breakdown (with --per_image)
  <save_predictions>/<dataset>/<name>_pred.png   — color overlay
  <save_predictions>/<dataset>/<name>_label.png  — grayscale class index
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image

os.environ['TORCH_HOME']     = '/pscratch/sd/w/worasit/torch_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME']        = '/pscratch/sd/w/worasit/.cache/huggingface'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
NUM_CLASSES  = 5
IGNORE_INDEX = 254
CLASS_NAMES  = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]
PATCH_SIZE   = 320
STRIDE       = 80    # denser than training (160) for smoother inference via Gaussian averaging
BATCH_SIZE   = 8

CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0 Background — black
    [255, 100, 100],   # 1 Epidermis  — red
    [100, 200, 100],   # 2 Vascular   — green
    [100, 100, 255],   # 3 Mesophyll  — blue
    [255, 230, 50 ],   # 4 Air Space  — yellow
], dtype=np.uint8)


# ============================================================
# DATASET
# ============================================================
class LeafDataset(Dataset):
    """Loads full images from a JSON config file (same format as training)."""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        self.name         = cfg["name"]
        self.image_dir    = cfg["image_dir"]
        self.mask_dir     = cfg["mask_dir"]
        self.mapping      = {int(k): int(v) for k, v in cfg["mapping"].items()}
        self.ignore_index = cfg.get("ignore_index", IGNORE_INDEX)

        file_list_path = cfg.get("file_list", None)
        if file_list_path:
            with open(file_list_path) as f:
                allowed = set(json.load(f))
            masks  = sorted([fn for fn in os.listdir(self.mask_dir)  if fn in allowed])
            stems  = {os.path.splitext(fn)[0] for fn in masks}
            images = sorted([fn for fn in os.listdir(self.image_dir)
                             if os.path.splitext(fn)[0] in stems and not fn.startswith(".")])
        else:
            images = sorted([fn for fn in os.listdir(self.image_dir) if not fn.startswith(".")])
            masks  = sorted([fn for fn in os.listdir(self.mask_dir)  if not fn.startswith(".")])

        assert len(images) == len(masks), \
            f"{self.name}: {len(images)} images vs {len(masks)} masks"
        print(f"  {self.name}: {len(images)} images")

        self._img_cache  = []
        self._mask_cache = []
        self._names      = []
        self._shapes     = []   # (H, W) original size — needed for patch stitching

        for img_name, mask_name in zip(images, masks):
            img  = np.array(Image.open(os.path.join(self.image_dir, img_name)))
            mask = np.array(Image.open(os.path.join(self.mask_dir,  mask_name)))

            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if img.ndim == 3:
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            H_img, W_img   = img.shape
            H_mask, W_mask = mask.shape
            if H_img != H_mask or W_img != W_mask:
                new_mask = np.full((H_img, W_img), self.ignore_index, dtype=mask.dtype)
                ph = min(H_img, H_mask)
                pw = min(W_img, W_mask)
                new_mask[:ph, :pw] = mask[:ph, :pw]
                mask = new_mask

            self._img_cache.append(img)
            self._mask_cache.append(mask.astype(np.uint8))
            self._names.append(img_name)
            self._shapes.append((H_img, W_img))

    def remap_mask(self, mask_np):
        remapped = np.full(mask_np.shape, self.ignore_index, dtype=np.int64)
        for gray_val, class_idx in self.mapping.items():
            remapped[mask_np == gray_val] = class_idx
        return remapped

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        img  = self._img_cache[idx].copy()
        mask = self._mask_cache[idx].copy()
        mask = self.remap_mask(mask)

        # astype(float32) forces native byte order — needed for big-endian TIFFs
        img_t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        valid_pixels = img_t[img_t > 0]
        if len(valid_pixels) > 0:
            mean, std = valid_pixels.mean(), valid_pixels.std()
            if std > 1e-5:
                img_t = (img_t - mean) / std

        return img_t, torch.from_numpy(np.asarray(mask, dtype=np.int64))


class PatchDataset(Dataset):
    """
    Tiles full images into overlapping patches for sliding-window inference.
    Returns (img_patch, mask_patch, global_img_idx, top, left, orig_H, orig_W).
    """

    def __init__(self, datasets, patch_size=320, stride=160):
        self.datasets    = datasets
        self.patch       = int(patch_size)
        self.stride      = int(stride)
        self.patch_index = []   # (ds_idx, local_img_idx, global_img_idx, top, left)

        self._names      = []
        self._shapes     = []
        offset = 0
        for ds_idx, ds in enumerate(self.datasets):
            self._names.extend(ds._names)
            self._shapes.extend(ds._shapes)
            for local_i, (H, W) in enumerate(ds._shapes):
                global_i = offset + local_i
                for t in self._grid(H, self.patch, self.stride):
                    for l in self._grid(W, self.patch, self.stride):
                        self.patch_index.append((ds_idx, local_i, global_i, t, l))
            offset += len(ds)

        print(f"  Total patches: {len(self.patch_index)}")

    @staticmethod
    def _grid(size, patch, stride):
        positions = list(range(0, max(1, size - patch + 1), stride))
        last = max(0, size - patch)
        if not positions or positions[-1] != last:
            positions.append(last)
        return positions

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        ds_idx, local_i, global_i, t, l = self.patch_index[idx]
        img, mask = self.datasets[ds_idx][local_i]
        _, H, W   = img.shape

        img_p  = img[:, t:min(t + self.patch, H), l:min(l + self.patch, W)]
        mask_p = mask[t:min(t + self.patch, H), l:min(l + self.patch, W)]

        pad_h = self.patch - img_p.shape[1]
        pad_w = self.patch - img_p.shape[2]
        if pad_h > 0 or pad_w > 0:
            img_p  = F.pad(img_p,  (0, pad_w, 0, pad_h), value=0.0)
            mask_p = F.pad(mask_p, (0, pad_w, 0, pad_h), value=IGNORE_INDEX)

        return (img_p, mask_p,
                torch.tensor(global_i, dtype=torch.long),
                torch.tensor(t,        dtype=torch.long),
                torch.tensor(l,        dtype=torch.long),
                torch.tensor(H,        dtype=torch.long),
                torch.tensor(W,        dtype=torch.long))


# ============================================================
# MODELS
# ============================================================
class EoMT(nn.Module):
    """
    End-to-end Mask Transformer using DINOv2 backbone.
    Supports ViT-B (embed_dim=768, heads=8) and ViT-L (embed_dim=1024, heads=16).
    """
    def __init__(self, num_classes=5, num_queries=100, backbone_name="vitl"):
        super().__init__()
        from transformers import AutoModel
        self.conv_in = nn.Conv2d(1, 3, kernel_size=1)

        hf_ids = {
            "vitl": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "vitb": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        }
        hf_id = hf_ids[backbone_name]
        print(f"Loading DINOv3 {backbone_name.upper()}...")
        self.backbone    = AutoModel.from_pretrained(hf_id, local_files_only=True)
        embed_dim        = self.backbone.config.hidden_size
        self._embed_dim  = embed_dim
        self._patch_size = self.backbone.config.patch_size
        num_heads        = 16 if backbone_name == "vitl" else 8

        self.q          = nn.Embedding(num_queries, embed_dim)
        self.query_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads,
                                                dropout=0.0, batch_first=True)
        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        self.mask_head  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim))
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2), nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim, 2, 2))

    def forward(self, images):
        B, C, H, W = images.shape
        if C == 1:
            images = self.conv_in(images)
        hidden  = self.backbone(images).last_hidden_state
        spatial = hidden[:, 1:, :]   # drop CLS token
        q_tokens = self.q.weight[None].expand(B, -1, -1)
        q_out, _ = self.query_attn(q_tokens, spatial, spatial)
        class_logits = self.class_head(q_out)
        gh = H // self._patch_size
        gw = W // self._patch_size
        spatial = spatial[:, :gh * gw, :]
        sp_grid = spatial.transpose(1, 2).reshape(B, self._embed_dim, gh, gw)
        sp_up   = self.upscale(sp_grid)
        masks_l = torch.einsum("bqd,bdhw->bqhw", self.mask_head(q_out), sp_up)
        sem     = torch.einsum("bqc,bqhw->bchw", class_logits[..., :-1], masks_l.sigmoid())
        return F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)


class SegFormerWrapper(nn.Module):
    """HuggingFace SegFormer-B4 adapted for 1-channel input."""
    def __init__(self, num_classes,
                 pretrained="nvidia/segformer-b4-finetuned-ade-512-512"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        try:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                pretrained, num_labels=num_classes,
                ignore_mismatched_sizes=True, local_files_only=True)
        except Exception:
            cfg = SegformerConfig.from_pretrained(pretrained, local_files_only=True)
            cfg.num_labels = num_classes
            self.segformer = SegformerForSemanticSegmentation(cfg)
        # Adapt first patch embedding conv: 3-ch → 1-ch
        enc = self.segformer.segformer.encoder
        old = enc.patch_embeddings[0].proj
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding)
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        new.bias.data   = old.bias.data.clone()
        enc.patch_embeddings[0].proj = new

    def forward(self, x):
        B, C, H, W = x.shape
        logits = self.segformer(pixel_values=x).logits   # (B, C, H/4, W/4)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)


class Mask2FormerWrapper(nn.Module):
    """HuggingFace Mask2Former/Swin-B — grayscale input via channel expansion."""
    def __init__(self, num_classes,
                 pretrained="facebook/mask2former-swin-base-IN21k-ade-semantic"):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained, num_labels=num_classes,
            ignore_mismatched_sizes=True, local_files_only=True)
        self.num_classes = num_classes

    def forward(self, x):
        x_rgb = x.expand(-1, 3, -1, -1)
        B, C, H, W = x.shape
        out         = self.model(pixel_values=x_rgb)
        class_probs = out.class_queries_logits[..., :self.num_classes].softmax(-1)
        masks       = F.interpolate(out.masks_queries_logits, size=(H, W),
                                    mode='bilinear', align_corners=False)
        return torch.einsum('bqc,bqhw->bchw', class_probs, masks.sigmoid())


def build_model(model_type, num_classes):
    mt = model_type.lower()

    # ---- DINOv2 / EoMT ----
    if mt == "eomt_vitl":
        return EoMT(num_classes=num_classes, backbone_name="vitl")
    if mt == "eomt_vitb":
        return EoMT(num_classes=num_classes, backbone_name="vitb")

    # ---- HuggingFace transformers ----
    if mt == "segformer":
        return SegFormerWrapper(num_classes=num_classes)
    if mt == "mask2former":
        return Mask2FormerWrapper(num_classes=num_classes)

    # ---- segmentation_models_pytorch ----
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("segmentation_models_pytorch is required for this model")

    smp_configs = {
        "unet_resnet101":      ("Unet",        "resnet101"),
        "deeplab_efficientnet":("DeepLabV3Plus","efficientnet-b4"),
        "deeplab_mitb4":       ("DeepLabV3Plus","mit_b4"),
        "fpn_mitb4":           ("FPN",          "mit_b4"),
        "fpn_mitb5":           ("FPN",          "mit_b5"),
    }
    if mt not in smp_configs:
        raise ValueError(
            f"Unknown model: '{model_type}'. "
            f"Choose from: {', '.join(list(smp_configs) + ['eomt_vitl','eomt_vitb','segformer','mask2former'])}"
        )
    arch_name, encoder = smp_configs[mt]
    arch = getattr(smp, arch_name)
    return arch(encoder_name=encoder, encoder_weights=None,
                in_channels=1, classes=num_classes)


# ============================================================
# HELPERS
# ============================================================
def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-7)
    iou       = tp / (tp + fp + fn + 1e-7)
    return precision, recall, dice, iou


def gaussian_kernel(size):
    ax     = torch.linspace(-1, 1, size)
    gauss  = torch.exp(-ax**2 / 0.5)
    kernel = torch.outer(gauss, gauss)
    return kernel / kernel.max()


def save_prediction_images(logit_sum, weight, orig_H, orig_W,
                           img_name, dataset_name, save_dir, threshold=0.0):
    """Save color overlay and grayscale label PNG for one image."""
    logit_avg   = logit_sum[:, :orig_H, :orig_W] / weight[:orig_H, :orig_W].clamp(min=1e-6)
    probs       = torch.softmax(logit_avg, dim=0)
    pred_cls    = torch.argmax(probs, dim=0).byte()
    if threshold > 0.0:
        pred_cls[probs.max(dim=0).values < threshold] = 0
    pred_np = pred_cls.numpy()

    stem      = os.path.splitext(img_name)[0]
    out_subdir = os.path.join(save_dir, dataset_name)
    os.makedirs(out_subdir, exist_ok=True)
    Image.fromarray(CLASS_COLORS[pred_np]).save(
        os.path.join(out_subdir, f"{stem}_pred.png"))
    Image.fromarray((pred_np.astype(np.uint8) * 50)).save(
        os.path.join(out_subdir, f"{stem}_label.png"))


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate leaf segmentation models (all 9 architectures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument("--model", required=True,
                        choices=["unet_resnet101", "deeplab_efficientnet", "deeplab_mitb4",
                                 "fpn_mitb4", "fpn_mitb5", "segformer", "mask2former",
                                 "eomt_vitb", "eomt_vitl"])
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--configs", nargs="+",
                            help="Explicit list of JSON config files")
    data_group.add_argument("--test_configs_dir",
                            help="Auto-discover all <prefix>*.json files in this directory")
    parser.add_argument("--prefix", default="t",
                        help="Filename prefix for --test_configs_dir (default: 't')")

    parser.add_argument("--save_predictions", default=None,
                        help="Directory to save _pred.png and _label.png per image")
    parser.add_argument("--per_image",   action="store_true",
                        help="Also save per-image IoU/Dice to CSV")
    parser.add_argument("--output_dir",  default=None,
                        help="Where to save CSV results (default: checkpoint directory)")

    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--patch_size",  type=int, default=PATCH_SIZE)
    parser.add_argument("--stride",      type=int, default=STRIDE,
                        help="Inference stride (default: 160, same as training). "
                             "Use 80 for denser overlap and smoother edges.")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE,
                        help="Per-GPU batch size (auto-scaled by GPU count)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--no_tta",      action="store_true",
                        help="Disable test-time augmentation (TTA is ON by default)")
    parser.add_argument("--threshold",   type=float, default=0.0,
                        help="Confidence threshold — pixels below this are set to background. "
                             "Default 0.0 = disabled. Try 0.3-0.5 to reduce noise.")

    args = parser.parse_args()

    # ----- Device + multi-GPU -----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
    print(f"\nDevice: {device}  ({num_gpus} GPU(s))")
    if device.type == "cuda":
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ----- Output dir -----
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)

    # ----- Resolve config files -----
    if args.configs:
        config_paths = args.configs
    else:
        cfg_dir = args.test_configs_dir
        if not os.path.isdir(cfg_dir):
            raise FileNotFoundError(f"--test_configs_dir not found: {cfg_dir}")
        config_paths = sorted([
            os.path.join(cfg_dir, fn)
            for fn in os.listdir(cfg_dir)
            if fn.startswith(args.prefix) and fn.endswith(".json")
        ])
        if not config_paths:
            raise ValueError(f"No '{args.prefix}*.json' files in {cfg_dir}")

    missing = [p for p in config_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Config files not found: {missing}")

    # ----- Load datasets -----
    print(f"\nLoading {len(config_paths)} test dataset(s)...")
    datasets     = [LeafDataset(p) for p in config_paths]
    total_images = sum(len(d) for d in datasets)
    print(f"Total test images: {total_images}")

    patch_ds = PatchDataset(datasets, patch_size=args.patch_size, stride=args.stride)

    effective_batch   = args.batch_size * num_gpus
    effective_workers = max(args.num_workers, num_gpus * 4)
    print(f"Batch: {effective_batch} ({args.batch_size}×{num_gpus} GPU)  "
          f"Workers: {effective_workers}  "
          f"TTA: {'OFF' if args.no_tta else 'ON (4-pass flip)'}")

    loader = DataLoader(patch_ds, batch_size=effective_batch, shuffle=False,
                        num_workers=effective_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(effective_workers > 0))

    # ----- Load model -----
    print(f"\nModel:      {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model      = build_model(args.model, args.num_classes)

    # Strip "module." prefix if checkpoint was saved under DDP
    state = checkpoint['model_state_dict']
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state)

    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel across {num_gpus} GPUs")
    model.eval()

    saved_epoch = checkpoint.get('epoch', '?')
    saved_miou  = checkpoint.get('best_miou', checkpoint.get('val_loss', float('nan')))
    print(f"Checkpoint epoch: {saved_epoch}  |  saved metric: {saved_miou}")

    # ----- Accumulators -----
    n       = args.num_classes
    tp      = torch.zeros(n)
    fp      = torch.zeros(n)
    fn      = torch.zeros(n)
    present = torch.zeros(n)

    per_img_tp = {}
    per_img_fp = {}
    per_img_fn = {}
    pred_accum = {}   # global_i -> FloatTensor (n, H, W)
    pred_count = {}   # global_i -> FloatTensor (H, W)

    do_save = args.save_predictions is not None
    if do_save:
        os.makedirs(args.save_predictions, exist_ok=True)
        gk_cache = {}   # patch_size -> Gaussian kernel

    # ----- Inference loop -----
    print(f"\nEvaluating {len(patch_ds)} patches...")
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs, masks, img_ids, tops, lefts, orig_Hs, orig_Ws = batch
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16) if device.type == "cuda" \
                    else torch.inference_mode():
                if not args.no_tta:
                    flips = [([], []), ([3], [3]), ([2], [2]), ([2, 3], [2, 3])]
                    logits = sum(
                        torch.flip(model(torch.flip(imgs, dims=fd)), dims=bd)
                        for fd, bd in flips
                    ) / len(flips)
                else:
                    logits = model(imgs)

            probs    = torch.softmax(logits.float(), dim=1)
            pred_cls = torch.argmax(probs, dim=1)
            if args.threshold > 0.0:
                pred_cls[probs.max(dim=1).values < args.threshold] = 0
            valid = (masks != IGNORE_INDEX)

            # Global metrics
            for c in range(n):
                pred_c = (pred_cls == c) & valid
                true_c = (masks    == c) & valid
                if true_c.any():
                    tp[c]      += (pred_c & true_c).sum().float().cpu()
                    fp[c]      += (pred_c & ~true_c).sum().float().cpu()
                    fn[c]      += (~pred_c & true_c).sum().float().cpu()
                    present[c] += 1

            # Per-image metrics + prediction stitching
            if args.per_image or do_save:
                logits_cpu = logits.float().cpu()
                for b in range(imgs.shape[0]):
                    gi  = img_ids[b].item()
                    t   = tops[b].item()
                    l   = lefts[b].item()
                    H   = orig_Hs[b].item()
                    W   = orig_Ws[b].item()
                    p_h = min(args.patch_size, H - t)
                    p_w = min(args.patch_size, W - l)

                    if args.per_image:
                        if gi not in per_img_tp:
                            per_img_tp[gi] = torch.zeros(n)
                            per_img_fp[gi] = torch.zeros(n)
                            per_img_fn[gi] = torch.zeros(n)
                        v = valid[b]
                        for c in range(n):
                            pc = (pred_cls[b] == c) & v
                            tc = (masks[b]    == c) & v
                            per_img_tp[gi][c] += (pc & tc).sum().float().cpu()
                            per_img_fp[gi][c] += (pc & ~tc).sum().float().cpu()
                            per_img_fn[gi][c] += (~pc & tc).sum().float().cpu()

                    if do_save:
                        if gi not in pred_accum:
                            pred_accum[gi] = torch.zeros(n, H, W)
                            pred_count[gi] = torch.zeros(H, W)
                        ps = args.patch_size
                        if ps not in gk_cache:
                            gk_cache[ps] = gaussian_kernel(ps)
                        gk = gk_cache[ps][:p_h, :p_w]
                        pred_accum[gi][:, t:t+p_h, l:l+p_w] += \
                            logits_cpu[b, :, :p_h, :p_w] * gk.unsqueeze(0)
                        pred_count[gi][t:t+p_h, l:l+p_w] += gk

    # ----- Save prediction images -----
    if do_save:
        ds_lookup = {}
        gi = 0
        for ds in datasets:
            for name in ds._names:
                ds_lookup[gi] = (ds.name, name)
                gi += 1
        print(f"\nSaving predictions to: {args.save_predictions}")
        for global_i in tqdm(pred_accum, desc="Saving"):
            ds_name, img_name = ds_lookup[global_i]
            orig_H, orig_W    = patch_ds._shapes[global_i]
            save_prediction_images(pred_accum[global_i], pred_count[global_i],
                                   orig_H, orig_W, img_name, ds_name,
                                   args.save_predictions, args.threshold)

    # ----- Overall metrics table -----
    precision, recall, dice, iou = compute_metrics(tp, fp, fn)
    rows = []
    for c in range(n):
        cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
        if present[c] > 0:
            rows.append({'Class':     cname,
                         'IoU':       round(iou[c].item(),       4),
                         'Dice':      round(dice[c].item(),      4),
                         'Precision': round(precision[c].item(), 4),
                         'Recall':    round(recall[c].item(),    4)})
        else:
            rows.append({'Class': cname,
                         'IoU': 'N/A', 'Dice': 'N/A',
                         'Precision': 'N/A', 'Recall': 'N/A'})

    df           = pd.DataFrame(rows)
    numeric_iou  = [r['IoU']  for r in rows if r['IoU']  != 'N/A']
    numeric_dice = [r['Dice'] for r in rows if r['Dice'] != 'N/A']

    print("\n" + "=" * 65)
    print(f"Model: {args.model}   Epoch: {saved_epoch}")
    print("=" * 65)
    print(df.to_string(index=False))
    print("-" * 65)
    print(f"Mean IoU  : {np.mean(numeric_iou):.4f}")
    print(f"Mean Dice : {np.mean(numeric_dice):.4f}")
    print("=" * 65)

    overall_csv = os.path.join(out_dir, f"eval_{args.model}_overall.csv")
    df.to_csv(overall_csv, index=False)
    print(f"\nOverall CSV : {overall_csv}")

    # ----- Per-image CSV -----
    if args.per_image and per_img_tp:
        ds_lookup_all = {}
        gi = 0
        for ds in datasets:
            for name in ds._names:
                ds_lookup_all[gi] = (ds.name, name)
                gi += 1

        per_img_rows = []
        for gi in sorted(per_img_tp.keys()):
            ds_name, img_name = ds_lookup_all.get(gi, ("?", str(gi)))
            p, r, d, iou_     = compute_metrics(per_img_tp[gi], per_img_fp[gi], per_img_fn[gi])
            row = {'Dataset': ds_name, 'Image': img_name}
            for c in range(n):
                cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
                row[f'IoU_{cname}']  = round(iou_[c].item(), 4)
                row[f'Dice_{cname}'] = round(d[c].item(),    4)
            per_img_rows.append(row)

        per_img_df  = pd.DataFrame(per_img_rows)
        per_img_csv = os.path.join(out_dir, f"eval_{args.model}_per_image.csv")
        per_img_df.to_csv(per_img_csv, index=False)
        print(f"Per-image CSV: {per_img_csv}")


if __name__ == "__main__":
    main()
