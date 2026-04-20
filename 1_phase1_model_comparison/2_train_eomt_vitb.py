"""
Leaf CT Segmentation — EoMT + DINOv3 ViT-B/16
===============================================
SLURM batch script version. Run via submit_train9.sh
- DistributedDataParallel: uses all 4 GPUs with proper scaling
- Resumes automatically from last_checkpoint.pth
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, DistributedSampler
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

# Reproducibility
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

# Pine configs — st_pinus_aa excluded (longitudinal cut, different from all other pine)
# Repeated 3x to balance against broadleaf data
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

CONFIG_PATHS = BROADLEAF_CONFIGS + PINE_CONFIGS * 3

OUTPUT_DIR = f"{SCRATCH}/outputs"
MODEL_DIR  = f"{OUTPUT_DIR}/models_EoMT"
LOG_DIR    = f"{OUTPUT_DIR}/logs_EoMT"

if is_main:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

PATCH_SIZE    = 320
STRIDE        = 160
BATCH_SIZE    = 32
NUM_EPOCHS    = 300
LR            = 1e-4
VAL_FRAC      = 0.2
PATIENCE      = 20
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

        # Pre-load all images and masks into RAM to avoid repeated disk reads
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

        img_t = torch.from_numpy(img).float().unsqueeze(0)
        valid_pixels = img_t[img_t > 0]
        if len(valid_pixels) > 0:
            mean, std = valid_pixels.mean(), valid_pixels.std()
            if std > 1e-5:
                img_t = (img_t - mean) / std

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

        # Phase 1: flip + rotate only (literature standard)
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
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
        img_p  = img[:, t:min(t + self.patch, H), l:min(l + self.patch, W)]
        mask_p = mask[t:min(t + self.patch, H), l:min(l + self.patch, W)]

        pad_h = self.patch - img_p.shape[1]
        pad_w = self.patch - img_p.shape[2]
        if pad_h > 0 or pad_w > 0:
            img_p  = F.pad(img_p,  (0, pad_w, 0, pad_h), value=0.0)
            pad_val = getattr(self.base.dataset if hasattr(self.base, 'dataset') else self.base,
                              'ignore_index', 254)
            mask_p = F.pad(mask_p, (0, pad_w, 0, pad_h), value=pad_val)

        if self.augment:
            img_np  = img_p.squeeze(0).numpy()
            mask_np = mask_p.numpy().astype(np.int32)
            aug_out = self.aug(image=img_np[..., np.newaxis], mask=mask_np)
            img_p   = torch.from_numpy(aug_out['image'].squeeze(-1)).unsqueeze(0)
            mask_p  = torch.from_numpy(aug_out['mask']).long()

        return img_p, mask_p


# ============================================================
# DATA LOADING
# ============================================================
all_datasets = [LeafDataset(p) for p in CONFIG_PATHS if os.path.exists(p)]
base_ds = ConcatDataset(all_datasets)
if is_main:
    print(f"\nTotal images: {len(base_ds)}")

total    = len(base_ds)
val_size = int(total * VAL_FRAC)
trn_size = total - val_size

train_base, val_base = random_split(
    base_ds, [trn_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_patch_ds = PatchDataset(train_base, PATCH_SIZE, STRIDE,
                               drop_background_only=True, augment=True)
val_patch_ds   = PatchDataset(val_base,   PATCH_SIZE, STRIDE,
                               drop_background_only=False, augment=False)

if is_main:
    print(f"Train images: {len(train_base)} | Train patches: {len(train_patch_ds)}")
    print(f"Val images:   {len(val_base)}   | Val patches:   {len(val_patch_ds)}")

train_sampler = DistributedSampler(train_patch_ds, shuffle=False, seed=SEED)
val_sampler   = DistributedSampler(val_patch_ds,   shuffle=False, seed=SEED)

train_loader = DataLoader(train_patch_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=8, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_patch_ds,   batch_size=BATCH_SIZE, sampler=val_sampler,
                          num_workers=8, pin_memory=True, persistent_workers=True)

# ============================================================
# MODEL — EoMT with DINOv3 ViT-B/16
# ============================================================
class EoMT_Wrapper(nn.Module):
    """
    Encoder-only Mask Transformer using DINOv3 ViT-B/16 backbone.
    Based on: 'Your ViT is Secretly an Image Segmentation Model' (arxiv 2503.19108)
    Backbone: facebook/dinov3-vitb16-pretrain-lvd1689m (DINOv3, arxiv 2508.10104)
    """
    def __init__(self, num_classes=5, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_q = num_queries

        # 1-channel CT -> 3-channel adapter
        self.conv_in = nn.Conv2d(1, 3, kernel_size=1)

        print("Loading DINOv3 ViT-B/16...")
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            local_files_only=True
        )
        embed_dim        = self.backbone.config.hidden_size  # 768
        patch_size       = self.backbone.config.patch_size   # 16
        self._embed_dim  = embed_dim
        self._patch_size = patch_size

        self.q = nn.Embedding(num_queries, embed_dim)

        self.query_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, dropout=0.0, batch_first=True
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
        hidden  = outputs.last_hidden_state       # [B, 1+num_patches, 768]
        spatial = hidden[:, 1:, :]               # skip CLS token

        q_tokens = self.q.weight[None, :, :].expand(B, -1, -1)
        q_out, _ = self.query_attn(q_tokens, spatial, spatial)

        class_logits = self.class_head(q_out)    # [B, 100, num_classes+1]

        grid_h   = H // self._patch_size
        grid_w   = W // self._patch_size
        expected = grid_h * grid_w
        if spatial.shape[1] > expected:
            spatial = spatial[:, :expected, :]
        spatial_grid = spatial.transpose(1, 2).reshape(B, self._embed_dim, grid_h, grid_w)
        spatial_up   = self.upscale(spatial_grid)  # [B, 768, H/4, W/4]

        masks_logits    = torch.einsum("bqd, bdhw -> bqhw", self.mask_head(q_out), spatial_up)
        mask_probs      = masks_logits.sigmoid()
        class_probs     = class_logits[..., :-1]
        semantic_logits = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
        semantic_logits = F.interpolate(semantic_logits, size=(H, W),
                                        mode="bilinear", align_corners=False)
        return semantic_logits


def get_model_state_dict(model):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


model = EoMT_Wrapper(num_classes=NUM_CLASSES)
model = model.to(device)

# Class weights: inverse pixel frequency from training set
class_weights = torch.tensor([1.0, 4.25, 3.15, 1.55, 4.95]).to(device)

def criterion(pred, mask):
    b, c, h, w = pred.shape
    pred_flat  = pred.permute(0, 2, 3, 1).reshape(-1, c)
    mask_flat  = mask.view(-1)
    valid_idx  = (mask_flat != IGNORE_INDEX)
    if valid_idx.sum() > 0:
        pred_valid    = pred_flat[valid_idx]
        mask_valid    = mask_flat[valid_idx]
        pixel_weights = class_weights[mask_valid]
        ce_loss       = F.cross_entropy(pred_valid, mask_valid, reduction='none')
        return (ce_loss * pixel_weights).mean()
    return torch.tensor(0.0, device=device)


# Differential LR: backbone 10x slower than new heads
# Optimizer is created BEFORE DDP so param groups reference correctly
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

# Wrap in DDP AFTER creating optimizer
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

# Load existing history so log is complete after resume (no duplicates)
if is_main and os.path.exists(HISTORY_CSV_PATH):
    history = pd.read_csv(HISTORY_CSV_PATH).to_dict('records')

if is_main:
    print(f"\nTraining started")
    print(f"Best model      -> {BEST_MODEL_PATH}")
    print(f"Last checkpoint -> {LAST_CHECKPOINT}")
    print("=" * 70)
    print(f"[CONFIG] model=EoMT/ViT-B LR={LR} wd=5e-3 patch={PATCH_SIZE} stride={STRIDE} "
          f"patience={PATIENCE} class_weights=[1.0,4.25,3.15,1.55,4.95] "
          f"aug=phase1(flip+rotate) loss=CE bg_only=True")
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

    # ===== TRAIN =====
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

    # ===== VALIDATE =====
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

    # ===== SAVE & LOG (rank 0 only) =====
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
            'configs':              CONFIG_PATHS,
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
                'configs':              CONFIG_PATHS,
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

        # Overwrite log from full history — no duplicates on resume
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

    # Sync early stopping across ranks
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
    print("Evaluating best model...")

    val_loader_eval = DataLoader(val_patch_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=False)

    CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model_eval = EoMT_Wrapper(num_classes=NUM_CLASSES)
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
            x        = x.to(device)
            y        = y.to(device)
            pred     = model_eval(x)
            pred_cls = torch.argmax(pred, dim=1)
            valid    = (y != IGNORE_INDEX)
            for c in range(NUM_CLASSES):
                pred_c = (pred_cls == c) & valid
                true_c = (y == c)        & valid
                if true_c.sum() > 0:
                    tp[c]      += (pred_c & true_c).sum().float().cpu()
                    fp[c]      += (pred_c & ~true_c).sum().float().cpu()
                    fn[c]      += (~pred_c & true_c).sum().float().cpu()
                    present[c] += 1

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-7)
    iou       = tp / (tp + fp + fn + 1e-7)

    rows = []
    for c in range(NUM_CLASSES):
        if present[c] > 0:
            rows.append({'Class': CLASS_NAMES[c],
                         'IoU':       round(iou[c].item(), 4),
                         'Dice':      round(dice[c].item(), 4),
                         'Precision': round(precision[c].item(), 4),
                         'Recall':    round(recall[c].item(), 4)})
        else:
            rows.append({'Class': CLASS_NAMES[c], 'IoU': 'N/A',
                         'Dice': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A'})

    modeldata = pd.DataFrame(rows)
    print("\n" + modeldata.to_string(index=False))

    numeric_iou = [r['IoU'] for r in rows if r['IoU'] != 'N/A']
    print(f"\nMean IoU  (present classes): {np.mean(numeric_iou):.4f}")
    print(f"Mean Dice (present classes): {np.mean([r['Dice'] for r in rows if r['Dice'] != 'N/A']):.4f}")

    metrics_csv = os.path.join(LOG_DIR, "validation_metrics.csv")
    modeldata.to_csv(metrics_csv, index=False)
    print(f"Metrics saved: {metrics_csv}")

dist.destroy_process_group()
