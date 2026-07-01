"""
Leaf CT Segmentation — EoMT + DINOv3 ViT-L/16  [Phase 2]
====================================================================
Training script for the Phase 2 EoMT candidate that produced the best
held-out mIoU among the fine-tuned EoMT / ViT-L runs.

Best-result recipe:
  - Backbone: DINOv3 ViT-L/16
  - Loss: weighted cross-entropy
  - Class weights: [1.00, 4.15, 3.10, 1.50, 4.60]
  - Scheduler: warmup followed by cosine annealing
  - Species-level split with pine x3 repeat
  - Augmentation: CLAHE-compatible uint8 augment first, then normalize
  - Patch size/stride: 320 / 160

SLURM: torchrun --standalone --nproc_per_node=4 1_train_eomt_vitl.py
"""

import os
import gc
import json
import math
import random
import numpy as np
import pandas as pd
from PIL import Image

os.environ['TORCH_HOME']     = '/pscratch/sd/w/worasit/torch_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME']        = '/pscratch/sd/w/worasit/.cache/huggingface'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModel
from tqdm import tqdm

import albumentations as A
import segmentation_models_pytorch as smp

# ============================================================
# DDP SETUP
# ============================================================
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device  = torch.device(f'cuda:{local_rank}')
is_main = (local_rank == 0)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if is_main:
    print(f"World size: {world_size}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ============================================================
# PATHS & CONFIG
# ============================================================
SCRATCH = "/pscratch/sd/w/worasit"

BROADLEAF_CONFIGS = [
    f"{SCRATCH}/configs/almond_control.json",
    f"{SCRATCH}/configs/almond_drought.json",
    f"{SCRATCH}/configs/arabidopsis1.json",
    f"{SCRATCH}/configs/arabidopsis2.json",
    f"{SCRATCH}/configs/arabidopsis3.json",
    f"{SCRATCH}/configs/grape_111.json",
    f"{SCRATCH}/configs/grape_112.json",
    f"{SCRATCH}/configs/grape_113.json",
    f"{SCRATCH}/configs/grape_76.json",
    f"{SCRATCH}/configs/grape_control.json",
    f"{SCRATCH}/configs/grape_drought.json",
    f"{SCRATCH}/configs/grape_f.json",
    f"{SCRATCH}/configs/grape_r.json",
    f"{SCRATCH}/configs/grape_w.json",
    f"{SCRATCH}/configs/lantana_camara.json",
    f"{SCRATCH}/configs/magnolia_grandiflora.json",
    f"{SCRATCH}/configs/oak_quce.json",
    f"{SCRATCH}/configs/oak_qucf.json",
    f"{SCRATCH}/configs/oak_qucr.json",
    f"{SCRATCH}/configs/oak_quob.json",
    f"{SCRATCH}/configs/oak_quru.json",
    f"{SCRATCH}/configs/oak_qusu.json",
    f"{SCRATCH}/configs/olive_d1.json",
    f"{SCRATCH}/configs/olive_d2.json",
    f"{SCRATCH}/configs/olive_d3.json",
    f"{SCRATCH}/configs/olive_d4.json",
    f"{SCRATCH}/configs/olive_d5.json",
    f"{SCRATCH}/configs/olive_rw1.json",
    f"{SCRATCH}/configs/olive_rw3.json",
    f"{SCRATCH}/configs/olive_rw4.json",
    f"{SCRATCH}/configs/olive_ww1.json",
    f"{SCRATCH}/configs/olive_ww2.json",
    f"{SCRATCH}/configs/olive_ww3.json",
    f"{SCRATCH}/configs/olive_ww4.json",
    f"{SCRATCH}/configs/olive_ww5.json",
    f"{SCRATCH}/configs/pistachio.json",
    f"{SCRATCH}/configs/pistachio_control.json",
    f"{SCRATCH}/configs/pistachio_drought.json",
    f"{SCRATCH}/configs/tomato.json",
    f"{SCRATCH}/configs/v_carlsii.json",
    f"{SCRATCH}/configs/v_cinnamo.json",
    f"{SCRATCH}/configs/v_davidii.json",
    f"{SCRATCH}/configs/v_davidii2.json",
    f"{SCRATCH}/configs/v_dentatum.json",
    f"{SCRATCH}/configs/v_dentatum2.json",
    f"{SCRATCH}/configs/v_furcatum.json",
    f"{SCRATCH}/configs/v_hartwegii.json",
    f"{SCRATCH}/configs/v_japonicum.json",
    f"{SCRATCH}/configs/v_jucundum.json",
    f"{SCRATCH}/configs/v_jucundum2.json",
    f"{SCRATCH}/configs/v_lantana.json",
    f"{SCRATCH}/configs/v_lautum.json",
    f"{SCRATCH}/configs/v_lautum3.json",
    f"{SCRATCH}/configs/v_propinquum.json",
    f"{SCRATCH}/configs/v_tinus.json",
    f"{SCRATCH}/configs/walnut.json",
]

PINE_CONFIGS = [
    f"{SCRATCH}/configs/pine_larix_occidentails1.json",
    f"{SCRATCH}/configs/pine_larix_occidentails2.json",
    f"{SCRATCH}/configs/pine_pinus_banksiana.json",
    f"{SCRATCH}/configs/pine_pinus_contorta2.json",
    f"{SCRATCH}/configs/pine_pinus_elliotii1.json",
    f"{SCRATCH}/configs/pine_pinus_elliotii2.json",
    f"{SCRATCH}/configs/pine_pinus_flexilis1.json",
    f"{SCRATCH}/configs/pine_pinus_flexilis3.json",
    f"{SCRATCH}/configs/pine_pinus_glabra.json",
    f"{SCRATCH}/configs/pine_pinus_halepensis.json",
    f"{SCRATCH}/configs/pine_pinus_jeffreyi1.json",
    f"{SCRATCH}/configs/pine_pinus_jeffreyi2.json",
    f"{SCRATCH}/configs/pine_pinus_kwangtungensis.json",
    f"{SCRATCH}/configs/pine_pinus_monticola2.json",
    f"{SCRATCH}/configs/pine_pinus_monticola5.json",
    f"{SCRATCH}/configs/pine_pinus_nigra1.json",
    f"{SCRATCH}/configs/pine_pinus_nigra2.json",
    f"{SCRATCH}/configs/pine_pinus_palustris.json",
    f"{SCRATCH}/configs/pine_pinus_pinaster.json",
    f"{SCRATCH}/configs/pine_pinus_pinea.json",
    f"{SCRATCH}/configs/pine_pinus_ponderosa.json",
    f"{SCRATCH}/configs/pine_pinus_pungens1.json",
    f"{SCRATCH}/configs/pine_pinus_pungens2.json",
    f"{SCRATCH}/configs/pine_pinus_rigida1.json",
    f"{SCRATCH}/configs/pine_pinus_rigida2.json",
    f"{SCRATCH}/configs/pine_pinus_sabiniana.json",
    f"{SCRATCH}/configs/pine_pinus_serotina.json",
    f"{SCRATCH}/configs/pine_pinus_thunbergii.json",
    f"{SCRATCH}/configs/pine_pinus_virginiana.json",
    f"{SCRATCH}/configs/pine_tsuga_mertensiana.json",
]

OUTPUT_DIR = f"{SCRATCH}/outputs"
MODEL_DIR  = f"{OUTPUT_DIR}/models_EoMT_ViTL_V17"
LOG_DIR    = f"{OUTPUT_DIR}/logs_EoMT_ViTL_V17"

if is_main:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

PATCH_SIZE    = 320
STRIDE        = 160
BATCH_SIZE    = 16
NUM_EPOCHS    = 300
LR            = 1e-4
VAL_FRAC      = 0.2
PATIENCE      = 30
NUM_CLASSES   = 5
IGNORE_INDEX  = 254
WARMUP_EPOCHS = 10

if is_main:
    print(f"Model dir: {MODEL_DIR}")
    print(f"Log dir:   {LOG_DIR}")

# ============================================================
# DATASET
# ============================================================
class LeafDataset(Dataset):
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)
        self.name      = self.cfg["name"]
        self.image_dir = self.cfg["image_dir"]
        self.mask_dir  = self.cfg["mask_dir"]
        self.mapping   = {int(k): int(v) for k, v in self.cfg["mapping"].items()}
        self.num_classes  = int(self.cfg["num_classes"])
        self.ignore_index = self.cfg.get("ignore_index", 254)

        file_list_path = self.cfg.get("file_list", None)
        if file_list_path:
            with open(file_list_path) as f:
                allowed = set(json.load(f))
            self.masks  = sorted([f for f in os.listdir(self.mask_dir) if f in allowed])
            stems       = {os.path.splitext(f)[0] for f in self.masks}
            self.images = sorted([f for f in os.listdir(self.image_dir)
                                  if os.path.splitext(f)[0] in stems and not f.startswith(".")])
        else:
            self.images = sorted([f for f in os.listdir(self.image_dir) if not f.startswith(".")])
            self.masks  = sorted([f for f in os.listdir(self.mask_dir)  if not f.startswith(".")])

        assert len(self.images) == len(self.masks), \
            f"{self.name}: {len(self.images)} images vs {len(self.masks)} masks"
        if is_main:
            print(f"Loaded: {self.name} — {len(self.images)} images")

        self._img_cache  = []
        self._mask_cache = []
        for img_name, mask_name in zip(self.images, self.masks):
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
                paste_h  = min(H_img, H_mask)
                paste_w  = min(W_img, W_mask)
                new_mask[:paste_h, :paste_w] = mask[:paste_h, :paste_w]
                mask = new_mask

            self._img_cache.append(img)
            self._mask_cache.append(mask.astype(np.uint8))

    def remap_mask(self, mask_np):
        remapped = np.full(mask_np.shape, self.ignore_index, dtype=np.int64)
        for gray_val, class_idx in self.mapping.items():
            remapped[mask_np == gray_val] = class_idx
        return remapped

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = self._img_cache[idx].copy()
        mask = self._mask_cache[idx].copy()

        mask = self.remap_mask(mask)

        # V17: return RAW uint8 tensor (no normalization).
        # PatchDataset will apply uint8 augmentations (CLAHE, RandomGamma) then normalize.
        img_t = torch.from_numpy(img).unsqueeze(0)  # uint8 tensor, shape (1, H, W)

        return img_t, torch.from_numpy(mask).long()


class PatchDataset(Dataset):
    def __init__(self, base_dataset, patch_size=256, stride=128,
                 drop_background_only=True, augment=False):
        self.base   = base_dataset
        self.patch  = int(patch_size)
        self.stride = int(stride)
        self.drop_background_only = drop_background_only
        self.augment = augment
        self.patch_index = []

        # V17: V11 stack restored INCLUDING CLAHE + RandomGamma.
        # These run on uint8 because the pipeline now augments BEFORE normalization.
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=50, p=0.5),
            A.ElasticTransform(alpha=80, sigma=8, p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),               # V17 NEW
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),     # V17 NEW (restored from V11)
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(16, 32),
                            hole_width_range=(16, 32),
                            p=0.3),
        ])
        self._build_index()

    def _grid_positions(self, H, W):
        tops  = list(range(0, max(1, H - self.patch + 1), self.stride))
        lefts = list(range(0, max(1, W - self.patch + 1), self.stride))
        if not tops  or tops[-1]  != max(0, H - self.patch):
            tops.append(max(0, H - self.patch))
        if not lefts or lefts[-1] != max(0, W - self.patch):
            lefts.append(max(0, W - self.patch))
        return tops, lefts

    def _build_index(self):
        self.patch_index = []
        for img_i in range(len(self.base)):
            img, mask = self.base[img_i]
            _, H, W = img.shape
            tops, lefts = self._grid_positions(H, W)
            for t in tops:
                for l in lefts:
                    mask_p = mask[t:min(t + self.patch, H), l:min(l + self.patch, W)]
                    uniq   = set(mask_p.flatten().tolist())
                    if self.drop_background_only:
                        if len(uniq) == 1 and 0 in uniq:
                            continue
                    if 1 in uniq or 2 in uniq or 4 in uniq:
                        self.patch_index.extend([(img_i, t, l)] * 4)
                    else:
                        self.patch_index.append((img_i, t, l))
        if is_main:
            print(f"Total valid patches: {len(self.patch_index)}")

    def shuffle_patches(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.patch_index)

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        img_i, t, l = self.patch_index[idx]
        img, mask   = self.base[img_i]

        annotated_coords = torch.nonzero(mask > 0)
        if len(annotated_coords) > 0:
            min_y    = annotated_coords[:, 0].min().item()
            max_y    = annotated_coords[:, 0].max().item()
            padding  = 100
            blindfold = torch.ones_like(mask, dtype=torch.bool)
            blindfold[max(0, min_y - padding):min(mask.shape[0], max_y + padding), :] = False
            mask[(mask == 0) & blindfold] = IGNORE_INDEX

        _, H, W = img.shape
        img_p  = img[:, t:min(t + self.patch, H), l:min(l + self.patch, W)]   # uint8
        mask_p = mask[t:min(t + self.patch, H), l:min(l + self.patch, W)]

        pad_h = self.patch - img_p.shape[1]
        pad_w = self.patch - img_p.shape[2]
        if pad_h > 0 or pad_w > 0:
            img_p  = F.pad(img_p,  (0, pad_w, 0, pad_h), value=0)             # uint8 pad with 0
            pad_val = getattr(self.base.dataset if hasattr(self.base, 'dataset') else self.base,
                              'ignore_index', 254)
            mask_p = F.pad(mask_p, (0, pad_w, 0, pad_h), value=pad_val)

        # V17: AUGMENT on uint8 (CLAHE/RandomGamma require uint8).
        if self.augment:
            img_np  = img_p.squeeze(0).numpy()                                  # (H, W) uint8
            mask_np = mask_p.numpy().astype(np.int32)
            aug_out = self.aug(image=img_np[..., np.newaxis], mask=mask_np)
            img_p   = torch.from_numpy(aug_out['image'].squeeze(-1)).unsqueeze(0)  # uint8
            mask_p  = torch.from_numpy(aug_out['mask']).long()

        # V17: Normalize AFTER augmentation, on the final uint8 patch.
        img_p_f = img_p.float()
        valid_pixels = img_p_f[img_p_f > 0]
        if len(valid_pixels) > 0:
            mean = valid_pixels.mean()
            std  = valid_pixels.std()
            if std > 1e-5:
                img_p_f = (img_p_f - mean) / std

        return img_p_f, mask_p


# ============================================================
# DATA LOADING — species-level split FIRST, oversample train only
# ============================================================
bl_shuffled   = BROADLEAF_CONFIGS[:]
pine_shuffled = PINE_CONFIGS[:]
random.seed(SEED)
random.shuffle(bl_shuffled)
random.shuffle(pine_shuffled)

bl_val_n   = int(len(bl_shuffled)   * VAL_FRAC)   # ~11 BL val species
pine_val_n = int(len(pine_shuffled) * VAL_FRAC)   # ~6 pine val species

bl_train   = bl_shuffled[bl_val_n:]     # ~45 BL train species
bl_val     = bl_shuffled[:bl_val_n]     # ~11 BL val species
pine_train = pine_shuffled[pine_val_n:] # ~24 pine train species
pine_val   = pine_shuffled[:pine_val_n] # ~6  pine val species

# V17: NO BL x2 (proven unhelpful in V14/V15/V16). Pine x3 only (matches V12 ratio).
# Oversampling applied ONLY to training — val is clean.
train_configs = bl_train + pine_train * 3   # ~45 + ~72 = ~117 configs (V12 ratio)
val_configs   = bl_val + pine_val           # ~17 configs, no duplicates

if is_main:
    print(f"\nSpecies split: BL {len(bl_train)} train + {len(bl_val)} val | "
          f"Pine {len(pine_train)} train + {len(pine_val)} val")
    print(f"Train configs: {len(bl_train)}BL + {len(pine_train)}pinex3 = {len(train_configs)}")
    print(f"Val   configs: {len(bl_val)}BL + {len(pine_val)}pine = {len(val_configs)} (no duplication)")

train_datasets = [LeafDataset(p) for p in train_configs if os.path.exists(p)]
val_datasets   = [LeafDataset(p) for p in val_configs   if os.path.exists(p)]

train_base = ConcatDataset(train_datasets)
val_base   = ConcatDataset(val_datasets)

if is_main:
    print(f"\nTotal train images: {len(train_base)}")
    print(f"Total val   images: {len(val_base)}")

train_patch_ds = PatchDataset(train_base, PATCH_SIZE, STRIDE,
                               drop_background_only=True, augment=True)
val_patch_ds   = PatchDataset(val_base,   PATCH_SIZE, STRIDE,
                               drop_background_only=False, augment=False)

if is_main:
    print(f"Train patches: {len(train_patch_ds)}")
    print(f"Val   patches: {len(val_patch_ds)}")

train_sampler = DistributedSampler(train_patch_ds, shuffle=False, seed=SEED)
val_sampler   = DistributedSampler(val_patch_ds,   shuffle=False, seed=SEED)

train_loader = DataLoader(train_patch_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=8, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_patch_ds,   batch_size=BATCH_SIZE, sampler=val_sampler,
                          num_workers=8, pin_memory=True, persistent_workers=True)

# ============================================================
# MODEL — EoMT with DINOv3 ViT-L/16 (unchanged from V12)
# ============================================================
class EoMT_ViTL(nn.Module):
    """
    Encoder-only Mask Transformer using DINOv3 ViT-L/16 backbone.
    ViT-L: hidden_size=1024, 24 layers
    Backbone: facebook/dinov3-vitl16-pretrain-lvd1689m
    """
    def __init__(self, num_classes=5, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_q = num_queries

        self.conv_in = nn.Conv2d(1, 3, kernel_size=1)

        print("Loading DINOv3 ViT-L/16...")
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            local_files_only=True
        )
        embed_dim        = self.backbone.config.hidden_size  # 1024
        patch_size       = self.backbone.config.patch_size   # 16
        self._embed_dim  = embed_dim
        self._patch_size = patch_size

        self.q = nn.Embedding(num_queries, embed_dim)

        self.query_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=16, dropout=0.0, batch_first=True
        )
        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        self.mask_head  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2)
        )

    def forward(self, images):
        B, C, H, W = images.shape
        if C == 1:
            images = self.conv_in(images)

        outputs = self.backbone(images)
        hidden  = outputs.last_hidden_state
        spatial = hidden[:, 1:, :]

        q_tokens = self.q.weight[None, :, :].expand(B, -1, -1)
        q_out, _ = self.query_attn(q_tokens, spatial, spatial)

        class_logits = self.class_head(q_out)

        grid_h   = H // self._patch_size
        grid_w   = W // self._patch_size
        expected = grid_h * grid_w
        if spatial.shape[1] > expected:
            spatial = spatial[:, :expected, :]
        spatial_grid = spatial.transpose(1, 2).reshape(B, self._embed_dim, grid_h, grid_w)
        spatial_up   = self.upscale(spatial_grid)

        masks_logits    = torch.einsum("bqd, bdhw -> bqhw", self.mask_head(q_out), spatial_up)
        mask_probs      = masks_logits.sigmoid()
        class_probs     = class_logits[..., :-1]
        semantic_logits = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
        semantic_logits = F.interpolate(semantic_logits, size=(H, W),
                                        mode="bilinear", align_corners=False)
        return semantic_logits


def get_model_state_dict(model):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


model = EoMT_ViTL(num_classes=NUM_CLASSES)
model = model.to(device)

class_weights = torch.tensor([1.00, 4.15, 3.10, 1.50, 4.60]).to(device)
loss_jaccard  = smp.losses.JaccardLoss(mode="multiclass", classes=NUM_CLASSES)


def criterion(pred, mask):
    """0.5 * weighted_CE + 0.5 * Jaccard"""
    b, c, h, w = pred.shape
    pred_flat  = pred.permute(0, 2, 3, 1).reshape(-1, c)
    mask_flat  = mask.view(-1)
    valid_idx  = (mask_flat != IGNORE_INDEX)
    if valid_idx.sum() > 0:
        pred_valid    = pred_flat[valid_idx]
        mask_valid    = mask_flat[valid_idx]
        pixel_weights = class_weights[mask_valid]
        jaccard_loss  = loss_jaccard(pred_valid, mask_valid)
        ce_loss       = F.cross_entropy(pred_valid, mask_valid, reduction='none')
        weighted_ce   = (ce_loss * pixel_weights).mean()
    else:
        jaccard_loss = torch.tensor(0.0, device=device)
        weighted_ce  = torch.tensor(0.0, device=device)
    return 0.5 * weighted_ce + 0.5 * jaccard_loss


optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(),   'lr': LR * 0.1},
    {'params': model.query_attn.parameters(), 'lr': LR},
    {'params': model.class_head.parameters(), 'lr': LR},
    {'params': model.mask_head.parameters(),  'lr': LR},
    {'params': model.upscale.parameters(),    'lr': LR},
    {'params': model.q.parameters(),          'lr': LR},
    {'params': model.conv_in.parameters(),    'lr': LR},
], weight_decay=5e-3)

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                         milestones=[WARMUP_EPOCHS])

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

if is_main:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.1f}M")

# ============================================================
# TRAINING
# ============================================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_CHECKPOINT = os.path.join(MODEL_DIR, "last_checkpoint.pth")

best_val_loss     = float('inf')
epochs_no_improve = 0
history           = []

TRAIN_LOG_PATH    = os.path.join(LOG_DIR, "training_log.txt")
BEST_SUMMARY_PATH = os.path.join(LOG_DIR, "best_summary.txt")
HISTORY_CSV_PATH  = os.path.join(LOG_DIR, "training_history.csv")

if is_main and os.path.exists(HISTORY_CSV_PATH):
    history = pd.read_csv(HISTORY_CSV_PATH).to_dict('records')

if is_main:
    print(f"\nTraining started")
    print(f"Best model      -> {BEST_MODEL_PATH}")
    print(f"Last checkpoint -> {LAST_CHECKPOINT}")
    print("=" * 70)
    print(f"[CONFIG V17] LR={LR} (diff backbone x0.1) wd=5e-3 patch={PATCH_SIZE} stride={STRIDE} "
          f"patience={PATIENCE} warmup={WARMUP_EPOCHS} "
          f"class_weights=[1.00,4.15,3.10,1.50,4.60] "
          f"aug=flip+rotate50+elastic+noise+bc+RandomGamma+CLAHE+blur+coarsedropout "
          f"PIPELINE:uint8-augment-then-normalize "
          f"loss=0.5*CE+0.5*Jaccard "
          f"train={len(train_configs)}configs({len(bl_train)}BL+{len(pine_train)}pinex3) "
          f"val={len(val_configs)}configs(no duplication)")
    print("=" * 70)

checkpoint_path = None
if os.path.exists(LAST_CHECKPOINT):
    checkpoint_path = LAST_CHECKPOINT
    if is_main:
        print("Found last checkpoint — resuming")
elif os.path.exists(BEST_MODEL_PATH):
    checkpoint_path = BEST_MODEL_PATH
    if is_main:
        print("Found best model — using as starting point")

if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception as e:
        if is_main:
            print(f"  Optimizer state incompatible ({e}), using fresh optimizer")
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            if is_main:
                print(f"  Scheduler state incompatible ({e}), using fresh scheduler")
    start_epoch       = checkpoint['epoch']
    best_val_loss     = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
    if is_main:
        print(f"Resumed from epoch {checkpoint['epoch']}, best_val_loss={best_val_loss:.4f}, no_improve={epochs_no_improve}")
else:
    start_epoch   = 0
    best_val_loss = float('inf')
    if is_main:
        print("Starting fresh")

for epoch in range(start_epoch, NUM_EPOCHS):
    train_patch_ds.shuffle_patches(seed=epoch)
    train_sampler.set_epoch(epoch)

    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Train]",
                     leave=False, disable=not is_main):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=torch.bfloat16):
            pred = model(x)
            loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss_t = torch.tensor(train_loss / len(train_loader), device=device)
    dist.all_reduce(train_loss_t, op=dist.ReduceOp.AVG)
    avg_train = train_loss_t.item()

    model.eval()
    val_loss, valid_batches = 0.0, 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val]",
                         leave=False, disable=not is_main):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
                loss = criterion(pred, y)
            lv = loss.item()
            if not math.isnan(lv) and not math.isinf(lv):
                val_loss      += lv
                valid_batches += 1

    val_loss_t = torch.tensor(val_loss / max(valid_batches, 1), device=device)
    dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
    avg_val = val_loss_t.item()

    scheduler.step()

    if is_main:
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss':           avg_train,
            'val_loss':             avg_val,
            'best_val_loss':        best_val_loss,
            'epochs_no_improve':    epochs_no_improve,
            'num_classes':          NUM_CLASSES,
            'patch_size':           PATCH_SIZE,
            'train_configs':        train_configs,
            'val_configs':          val_configs,
        }, LAST_CHECKPOINT)

        saved_str = ""
        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            epochs_no_improve = 0
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     get_model_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss':           avg_train,
                'val_loss':             avg_val,
                'num_classes':          NUM_CLASSES,
                'patch_size':           PATCH_SIZE,
                'train_configs':        train_configs,
                'val_configs':          val_configs,
            }, BEST_MODEL_PATH)
            saved_str = " <- saved best"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}{saved_str}")

        history.append({'epoch': epoch+1, 'train_loss': avg_train,
                        'val_loss': avg_val, 'lr': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(HISTORY_CSV_PATH, index=False)

        with open(TRAIN_LOG_PATH, "w") as f:
            for row in history:
                f.write(f"Epoch {int(row['epoch']):3d}/{NUM_EPOCHS} | Train: {row['train_loss']:.4f} | "
                        f"Val: {row['val_loss']:.4f} | LR: {row['lr']:.2e}\n")

        if saved_str:
            with open(BEST_SUMMARY_PATH, "w") as f:
                f.write(f"Best epoch  : {epoch+1}\n")
                f.write(f"Val loss    : {avg_val:.4f}\n")
                f.write(f"Train loss  : {avg_train:.4f}\n")
                f.write(f"LR          : {optimizer.param_groups[0]['lr']:.2e}\n")
                f.write(f"Checkpoint  : {BEST_MODEL_PATH}\n")

    stop_tensor = torch.tensor(int(epochs_no_improve >= PATIENCE), device=device)
    dist.broadcast(stop_tensor, src=0)
    if stop_tensor.item():
        if is_main:
            print(f"\nEarly stopping at epoch {epoch+1}")
        break

    state_tensor = torch.tensor([best_val_loss, float(epochs_no_improve)], device=device)
    dist.broadcast(state_tensor, src=0)
    best_val_loss     = state_tensor[0].item()
    epochs_no_improve = int(state_tensor[1].item())

    if (epoch + 1) % 20 == 0:
        torch.cuda.empty_cache()
        gc.collect()

if is_main:
    pd.DataFrame(history).to_csv(os.path.join(LOG_DIR, "training_history.csv"), index=False)
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")

dist.barrier()

# ============================================================
# EVALUATION (rank 0 only)
# ============================================================
if is_main:
    print("\n" + "=" * 70)
    print("Evaluating best model on validation set...")

    val_loader_eval = DataLoader(val_patch_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=False)

    CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model_eval = EoMT_ViTL(num_classes=NUM_CLASSES)
    model_eval.load_state_dict(checkpoint['model_state_dict'])
    model_eval = model_eval.to(device)
    model_eval.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4f})")

    tp      = torch.zeros(NUM_CLASSES)
    fp      = torch.zeros(NUM_CLASSES)
    fn      = torch.zeros(NUM_CLASSES)
    present = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for x, y in tqdm(val_loader_eval, desc="Evaluating"):
            x = x.to(device); y = y.to(device)
            pred_cls = torch.argmax(model_eval(x), dim=1)
            valid    = (y != IGNORE_INDEX)
            for c in range(NUM_CLASSES):
                pc = (pred_cls == c) & valid
                tc = (y == c) & valid
                if tc.sum() > 0:
                    tp[c]      += (pc & tc).sum().float().cpu()
                    fp[c]      += (pc & ~tc).sum().float().cpu()
                    fn[c]      += (~pc & tc).sum().float().cpu()
                    present[c] += 1

    iou  = tp / (tp + fp + fn + 1e-7)
    dice = 2*tp / (2*tp + fp + fn + 1e-7)
    rows = [{'Class': CLASS_NAMES[c],
             'IoU':  round(iou[c].item(), 4)  if present[c] > 0 else 'N/A',
             'Dice': round(dice[c].item(), 4) if present[c] > 0 else 'N/A'}
            for c in range(NUM_CLASSES)]

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    valid_iou = [r['IoU'] for r in rows if r['IoU'] != 'N/A']
    print(f"\nMean IoU (val species): {np.mean(valid_iou):.4f}")
    df.to_csv(os.path.join(LOG_DIR, "validation_metrics.csv"), index=False)

dist.destroy_process_group()
