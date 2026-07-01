"""
Leaf CT Segmentation — FPN / MiT-B4  [Phase 2]
====================================================================
Training script for the Phase 2 FPN candidate that produced the best
held-out mIoU among the fine-tuned FPN / MiT-B4 runs.

Best-result recipe:
  - Backbone: FPN with MiT-B4 encoder
  - Loss: 0.4 * weighted CE + 0.4 * per-class Tversky + 0.2 * Lovasz
  - Tversky alpha/beta: 0.7 / 0.3
  - Class weights: [1.00, 5.00, 4.00, 2.50, 4.60]
  - EMA decay: 0.999
  - Patch oversampling: classes 1, 2, and 4 at 4x
  - Augmentation: uint8 augment first, then normalize
  - Patch size/stride: 320 / 160

SLURM: torchrun --standalone --nproc_per_node=4 1_train_fpn_mitb4.py
"""

import os, gc, json, math, random
import numpy as np
import pandas as pd
from PIL import Image

os.environ['TORCH_HOME']           = '/pscratch/sd/w/worasit/torch_cache'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME']              = '/pscratch/sd/w/worasit/.cache/huggingface'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler
from torch.amp import autocast
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
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

if is_main:
    print(f"World size: {world_size}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ============================================================
# PATHS & CONFIG
# ============================================================
SCRATCH = "/pscratch/sd/w/worasit"

BROADLEAF_CONFIGS = [
    f"{SCRATCH}/configs/almond_control.json",        f"{SCRATCH}/configs/almond_drought.json",
    f"{SCRATCH}/configs/arabidopsis1.json",          f"{SCRATCH}/configs/arabidopsis2.json",
    f"{SCRATCH}/configs/arabidopsis3.json",
    f"{SCRATCH}/configs/grape_111.json",             f"{SCRATCH}/configs/grape_112.json",
    f"{SCRATCH}/configs/grape_113.json",             f"{SCRATCH}/configs/grape_76.json",
    f"{SCRATCH}/configs/grape_control.json",         f"{SCRATCH}/configs/grape_drought.json",
    f"{SCRATCH}/configs/grape_f.json",               f"{SCRATCH}/configs/grape_r.json",
    f"{SCRATCH}/configs/grape_w.json",
    f"{SCRATCH}/configs/lantana_camara.json",
    f"{SCRATCH}/configs/magnolia_grandiflora.json",
    f"{SCRATCH}/configs/oak_quce.json",              f"{SCRATCH}/configs/oak_qucf.json",
    f"{SCRATCH}/configs/oak_qucr.json",              f"{SCRATCH}/configs/oak_quob.json",
    f"{SCRATCH}/configs/oak_quru.json",              f"{SCRATCH}/configs/oak_qusu.json",
    f"{SCRATCH}/configs/olive_d1.json",              f"{SCRATCH}/configs/olive_d2.json",
    f"{SCRATCH}/configs/olive_d3.json",              f"{SCRATCH}/configs/olive_d4.json",
    f"{SCRATCH}/configs/olive_d5.json",              f"{SCRATCH}/configs/olive_rw1.json",
    f"{SCRATCH}/configs/olive_rw3.json",             f"{SCRATCH}/configs/olive_rw4.json",
    f"{SCRATCH}/configs/olive_ww1.json",             f"{SCRATCH}/configs/olive_ww2.json",
    f"{SCRATCH}/configs/olive_ww3.json",             f"{SCRATCH}/configs/olive_ww4.json",
    f"{SCRATCH}/configs/olive_ww5.json",
    f"{SCRATCH}/configs/pistachio.json",             f"{SCRATCH}/configs/pistachio_control.json",
    f"{SCRATCH}/configs/pistachio_drought.json",
    f"{SCRATCH}/configs/tomato.json",
    f"{SCRATCH}/configs/v_carlsii.json",             f"{SCRATCH}/configs/v_cinnamo.json",
    f"{SCRATCH}/configs/v_davidii.json",             f"{SCRATCH}/configs/v_davidii2.json",
    f"{SCRATCH}/configs/v_dentatum.json",            f"{SCRATCH}/configs/v_dentatum2.json",
    f"{SCRATCH}/configs/v_furcatum.json",            f"{SCRATCH}/configs/v_hartwegii.json",
    f"{SCRATCH}/configs/v_japonicum.json",           f"{SCRATCH}/configs/v_jucundum.json",
    f"{SCRATCH}/configs/v_jucundum2.json",           f"{SCRATCH}/configs/v_lantana.json",
    f"{SCRATCH}/configs/v_lautum.json",              f"{SCRATCH}/configs/v_lautum3.json",
    f"{SCRATCH}/configs/v_propinquum.json",          f"{SCRATCH}/configs/v_tinus.json",
    f"{SCRATCH}/configs/walnut.json",
]
PINE_CONFIGS = [
    f"{SCRATCH}/configs/pine_larix_occidentails1.json",
    f"{SCRATCH}/configs/pine_larix_occidentails2.json",
    f"{SCRATCH}/configs/pine_pinus_banksiana.json",  f"{SCRATCH}/configs/pine_pinus_contorta2.json",
    f"{SCRATCH}/configs/pine_pinus_elliotii1.json",  f"{SCRATCH}/configs/pine_pinus_elliotii2.json",
    f"{SCRATCH}/configs/pine_pinus_flexilis1.json",  f"{SCRATCH}/configs/pine_pinus_flexilis3.json",
    f"{SCRATCH}/configs/pine_pinus_glabra.json",     f"{SCRATCH}/configs/pine_pinus_halepensis.json",
    f"{SCRATCH}/configs/pine_pinus_jeffreyi1.json",  f"{SCRATCH}/configs/pine_pinus_jeffreyi2.json",
    f"{SCRATCH}/configs/pine_pinus_kwangtungensis.json",
    f"{SCRATCH}/configs/pine_pinus_monticola2.json", f"{SCRATCH}/configs/pine_pinus_monticola5.json",
    f"{SCRATCH}/configs/pine_pinus_nigra1.json",     f"{SCRATCH}/configs/pine_pinus_nigra2.json",
    f"{SCRATCH}/configs/pine_pinus_palustris.json",  f"{SCRATCH}/configs/pine_pinus_pinaster.json",
    f"{SCRATCH}/configs/pine_pinus_pinea.json",      f"{SCRATCH}/configs/pine_pinus_ponderosa.json",
    f"{SCRATCH}/configs/pine_pinus_pungens1.json",   f"{SCRATCH}/configs/pine_pinus_pungens2.json",
    f"{SCRATCH}/configs/pine_pinus_rigida1.json",    f"{SCRATCH}/configs/pine_pinus_rigida2.json",
    f"{SCRATCH}/configs/pine_pinus_sabiniana.json",  f"{SCRATCH}/configs/pine_pinus_serotina.json",
    f"{SCRATCH}/configs/pine_pinus_thunbergii.json", f"{SCRATCH}/configs/pine_pinus_virginiana.json",
    f"{SCRATCH}/configs/pine_tsuga_mertensiana.json",
]

OUTPUT_DIR = f"{SCRATCH}/outputs"
MODEL_DIR  = f"{OUTPUT_DIR}/models_FPN_MiTB4_V4"
LOG_DIR    = f"{OUTPUT_DIR}/logs_FPN_MiTB4_V4"
if is_main:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

PATCH_SIZE   = 320
STRIDE       = 160
BATCH_SIZE   = 16
NUM_EPOCHS   = 100
LR           = 1e-4
VAL_FRAC     = 0.2
PATIENCE     = 15
NUM_CLASSES  = 5
IGNORE_INDEX = 254

# V2 settings (kept in V3)
EMA_DECAY    = 0.999
CE_W         = 0.4
TVERSKY_W    = 0.4       # was 0.4 Dice in V2; same magnitude
LOVASZ_W     = 0.2

# V3 NEW — Tversky parameters
TVERSKY_ALPHA = 0.7      # FP penalty weight (Dice equivalent: 0.5)
TVERSKY_BETA  = 0.3      # FN penalty weight (Dice equivalent: 0.5)


# ============================================================
# DATASET (uint8 pipeline -- V17/V19 style with CLAHE-able aug)
# ============================================================
class LeafDataset(Dataset):
    def __init__(self, config_path):
        with open(config_path) as f:
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
        if is_main: print(f"Loaded: {self.name} — {len(self.images)} images")

        self._img_cache, self._mask_cache = [], []
        for img_name, mask_name in zip(self.images, self.masks):
            img  = np.array(Image.open(os.path.join(self.image_dir, img_name)))
            mask = np.array(Image.open(os.path.join(self.mask_dir,  mask_name)))
            if img.ndim == 3 and img.shape[2] == 4: img = img[:, :, :3]
            if img.ndim == 3: img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            if mask.ndim == 3: mask = mask[:, :, 0]
            H_img, W_img = img.shape; H_mask, W_mask = mask.shape
            if H_img != H_mask or W_img != W_mask:
                new_mask = np.full((H_img, W_img), self.ignore_index, dtype=mask.dtype)
                ph, pw = min(H_img, H_mask), min(W_img, W_mask)
                new_mask[:ph, :pw] = mask[:ph, :pw]; mask = new_mask
            self._img_cache.append(img)
            self._mask_cache.append(mask.astype(np.uint8))

    def remap_mask(self, mask_np):
        remapped = np.full(mask_np.shape, self.ignore_index, dtype=np.int64)
        for gray_val, class_idx in self.mapping.items():
            remapped[mask_np == gray_val] = class_idx
        return remapped

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img  = self._img_cache[idx].copy()
        mask = self.remap_mask(self._mask_cache[idx].copy())
        # V2: return RAW uint8 (PatchDataset augments uint8 then normalizes)
        img_t = torch.from_numpy(img).unsqueeze(0)
        return img_t, torch.from_numpy(mask).long()


class PatchDataset(Dataset):
    def __init__(self, base_dataset, patch_size=256, stride=128,
                 drop_background_only=True, augment=False):
        self.base   = base_dataset
        self.patch  = int(patch_size)
        self.stride = int(stride)
        self.drop_background_only = drop_background_only
        self.augment = augment
        # V17/V19 full aug stack on uint8
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=50, p=0.5),
            A.ElasticTransform(alpha=80, sigma=8, p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
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
        if not tops  or tops[-1]  != max(0, H - self.patch): tops.append(max(0, H - self.patch))
        if not lefts or lefts[-1] != max(0, W - self.patch): lefts.append(max(0, W - self.patch))
        return tops, lefts

    def _build_index(self):
        self.patch_index = []
        for img_i in range(len(self.base)):
            img, mask = self.base[img_i]; _, H, W = img.shape
            tops, lefts = self._grid_positions(H, W)
            for t in tops:
                for l in lefts:
                    mask_p = mask[t:min(t+self.patch, H), l:min(l+self.patch, W)]
                    uniq   = set(mask_p.flatten().tolist())
                    if self.drop_background_only and len(uniq) == 1 and 0 in uniq: continue
                    if 1 in uniq or 2 in uniq or 4 in uniq:
                        self.patch_index.extend([(img_i, t, l)] * 4)
                    else:
                        self.patch_index.append((img_i, t, l))
        if is_main: print(f"Total valid patches: {len(self.patch_index)}")

    def shuffle_patches(self, seed=None):
        if seed is not None: random.seed(seed)
        random.shuffle(self.patch_index)

    def __len__(self): return len(self.patch_index)

    def __getitem__(self, idx):
        img_i, t, l = self.patch_index[idx]
        img, mask   = self.base[img_i]    # uint8 (1, H, W)
        annotated_coords = torch.nonzero(mask > 0)
        if len(annotated_coords) > 0:
            min_y = annotated_coords[:, 0].min().item()
            max_y = annotated_coords[:, 0].max().item()
            blindfold = torch.ones_like(mask, dtype=torch.bool)
            blindfold[max(0, min_y-100):min(mask.shape[0], max_y+100), :] = False
            mask[(mask == 0) & blindfold] = IGNORE_INDEX
        _, H, W = img.shape
        img_p  = img[:, t:min(t+self.patch, H), l:min(l+self.patch, W)]
        mask_p = mask[t:min(t+self.patch, H), l:min(l+self.patch, W)]
        pad_h  = self.patch - img_p.shape[1]; pad_w = self.patch - img_p.shape[2]
        if pad_h > 0 or pad_w > 0:
            img_p  = F.pad(img_p,  (0, pad_w, 0, pad_h), value=0)
            pad_val = getattr(self.base.dataset if hasattr(self.base, 'dataset') else self.base,
                              'ignore_index', 254)
            mask_p = F.pad(mask_p, (0, pad_w, 0, pad_h), value=pad_val)

        if self.augment:
            img_np  = img_p.squeeze(0).numpy()
            mask_np = mask_p.numpy().astype(np.int32)
            aug_out = self.aug(image=img_np[..., np.newaxis], mask=mask_np)
            img_p   = torch.from_numpy(aug_out['image'].squeeze(-1)).unsqueeze(0)
            mask_p  = torch.from_numpy(aug_out['mask']).long()

        # Normalize AFTER augmentation, on the final uint8 patch
        img_p_f = img_p.float()
        valid_pixels = img_p_f[img_p_f > 0]
        if len(valid_pixels) > 0:
            mean = valid_pixels.mean()
            std  = valid_pixels.std()
            if std > 1e-5:
                img_p_f = (img_p_f - mean) / std
        return img_p_f, mask_p


# ============================================================
# DATA LOADING — species-level split (clean)
# ============================================================
bl_shuffled   = BROADLEAF_CONFIGS[:]
pine_shuffled = PINE_CONFIGS[:]
random.seed(SEED)
random.shuffle(bl_shuffled)
random.shuffle(pine_shuffled)

bl_val_n   = int(len(bl_shuffled)   * VAL_FRAC)
pine_val_n = int(len(pine_shuffled) * VAL_FRAC)
bl_train   = bl_shuffled[bl_val_n:]
bl_val     = bl_shuffled[:bl_val_n]
pine_train = pine_shuffled[pine_val_n:]
pine_val   = pine_shuffled[:pine_val_n]

train_configs = bl_train + pine_train * 3
val_configs   = bl_val + pine_val

if is_main:
    print(f"\nSpecies split: BL {len(bl_train)}+{len(bl_val)} | Pine {len(pine_train)}+{len(pine_val)}")
    print(f"Train configs: {len(train_configs)} | Val configs: {len(val_configs)}")

train_base = ConcatDataset([LeafDataset(p) for p in train_configs if os.path.exists(p)])
val_base   = ConcatDataset([LeafDataset(p) for p in val_configs   if os.path.exists(p)])

train_patch_ds = PatchDataset(train_base, PATCH_SIZE, STRIDE, drop_background_only=True,  augment=True)
val_patch_ds   = PatchDataset(val_base,   PATCH_SIZE, STRIDE, drop_background_only=False, augment=False)
if is_main:
    print(f"Train patches: {len(train_patch_ds)} | Val patches: {len(val_patch_ds)}")

train_sampler = DistributedSampler(train_patch_ds, shuffle=False, seed=SEED)
val_sampler   = DistributedSampler(val_patch_ds,   shuffle=False, seed=SEED)
train_loader  = DataLoader(train_patch_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                           num_workers=8, pin_memory=True, persistent_workers=True)
val_loader    = DataLoader(val_patch_ds,   batch_size=BATCH_SIZE, sampler=val_sampler,
                           num_workers=8, pin_memory=True, persistent_workers=True)


# ============================================================
# FPN + MiT-B4 (matches Phase 1 clean's fpn_mitb4)
# ============================================================
def build_fpn_mitb4(num_classes):
    return smp.FPN(encoder_name="mit_b4", encoder_weights="imagenet",
                   in_channels=1, classes=num_classes)


def get_model_state_dict(m):
    return m.module.state_dict() if hasattr(m, 'module') else m.state_dict()


# ============================================================
# EMA helper (mirrors EoMT V19 / M2F V19)
# ============================================================
class ModelEMA:
    def __init__(self, model_module, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model_module.state_dict().items()}

    @torch.no_grad()
    def update(self, model_module):
        for k, v in model_module.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self): return self.shadow

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow: self.shadow[k].copy_(v)
            else: self.shadow[k] = v.clone()


def _swap_in_weights(model_module, new_sd):
    backup = {k: v.detach().clone() for k, v in model_module.state_dict().items()}
    model_module.load_state_dict(new_sd)
    return backup


# ============================================================
# MODEL + LOSS
# ============================================================
model = build_fpn_mitb4(num_classes=NUM_CLASSES).to(device)

# V2 FPN-MiT-B4 class weights (full Vas boost, matches M2F V19):
class_weights = torch.tensor([1.00, 5.00, 4.00, 2.50, 4.60]).to(device)
loss_lovasz   = smp.losses.LovaszLoss(mode="multiclass", from_logits=True)


def criterion(pred, mask):
    """V4: 0.4 * weighted_CE + 0.4 * per-class Tversky(0.7, 0.3) + 0.2 * Lovasz (V3 loss kept)"""
    b, c, h, w = pred.shape

    # --- 1. Weighted CE ---
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, c)
    mask_flat = mask.view(-1)
    valid_idx = (mask_flat != IGNORE_INDEX)
    if valid_idx.sum() == 0:
        return torch.tensor(0.0, device=device)
    pred_valid    = pred_flat[valid_idx]
    mask_valid    = mask_flat[valid_idx]
    pixel_weights = class_weights[mask_valid]
    ce_per_pixel  = F.cross_entropy(pred_valid, mask_valid, reduction='none')
    ce_loss       = (ce_per_pixel * pixel_weights).mean()

    # --- 2. V3 NEW: per-class Tversky (replaces V2 per-class Dice) ---
    # Tversky = TP / (TP + alpha*FP + beta*FN)
    # alpha = beta = 0.5 -> Dice; alpha > beta -> penalises FP harder
    pred_soft  = F.softmax(pred, dim=1)
    valid_mask = (mask != IGNORE_INDEX).float()
    tversky_terms = []
    for cls in range(NUM_CLASSES):
        p = pred_soft[:, cls] * valid_mask
        y = (mask == cls).float() * valid_mask
        tp = (p * y).sum()
        fp = (p * (1.0 - y) * valid_mask).sum()
        fn = ((1.0 - p) * y).sum()
        tversky_terms.append(
            (tp + 1e-7) /
            (tp + TVERSKY_ALPHA * fp + TVERSKY_BETA * fn + 1e-7)
        )
    tversky_loss = 1.0 - torch.stack(tversky_terms).mean()

    # --- 3. Lovasz on spatial logits (kept from V2) ---
    safe_mask    = mask.masked_fill(mask == IGNORE_INDEX, 0)
    lovasz_loss  = loss_lovasz(pred, safe_mask)
    valid_frac   = valid_idx.float().mean()
    lovasz_loss  = lovasz_loss * valid_frac

    return CE_W * ce_loss + TVERSKY_W * tversky_loss + LOVASZ_W * lovasz_loss


optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
model     = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
ema       = ModelEMA(model.module, decay=EMA_DECAY)

if is_main:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n/1e6:.1f}M")
    print(f"EMA decay     : {EMA_DECAY}")
    print(f"Class weights : {class_weights.tolist()}  (V4 NEW: Mes 2.0 -> 2.5 to address Mes under-prediction)")
    print(f"Loss weights  : CE={CE_W}  Tversky={TVERSKY_W} (alpha={TVERSKY_ALPHA}, beta={TVERSKY_BETA})  Lovasz={LOVASZ_W}")
    print("=" * 70)
    print(f"[CONFIG FPN-MiT-B4 V4] LR={LR} wd=5e-3 patch={PATCH_SIZE} stride={STRIDE} "
          f"patience={PATIENCE} class_weights=[1.00,5.00,4.00,2.50,4.60] (V4 NEW: Mes 2.0->2.5) "
          f"oversample=Epi+Vas+Air(4x) "
          f"loss={CE_W}*CE+{TVERSKY_W}*Tversky(a={TVERSKY_ALPHA},b={TVERSKY_BETA})+{LOVASZ_W}*Lovasz (V3 kept) EMA={EMA_DECAY} "
          f"scheduler=ReduceLROnPlateau "
          f"aug=flip+rotate50+elastic+noise+bc+RandomGamma+CLAHE+blur+coarsedropout "
          f"PIPELINE:uint8-augment-then-normalize batch={BATCH_SIZE} "
          f"train={len(train_configs)} val={len(val_configs)} (V3 + Mes weight 2.0->2.5 single-variable change)")
    print("=" * 70)

# ============================================================
# TRAINING
# ============================================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_CHECKPOINT = os.path.join(MODEL_DIR, "last_checkpoint.pth")
HISTORY_CSV     = os.path.join(LOG_DIR, "training_history.csv")
TRAIN_LOG_PATH  = os.path.join(LOG_DIR, "training_log.txt")
BEST_SUMMARY    = os.path.join(LOG_DIR, "best_summary.txt")

best_val_loss = float('inf'); epochs_no_improve = 0; history = []
if is_main and os.path.exists(HISTORY_CSV):
    history = pd.read_csv(HISTORY_CSV).to_dict('records')

checkpoint_path = LAST_CHECKPOINT if os.path.exists(LAST_CHECKPOINT) else \
                  BEST_MODEL_PATH  if os.path.exists(BEST_MODEL_PATH)  else None
if checkpoint_path:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'live_state_dict' in ckpt:
        model.module.load_state_dict(ckpt['live_state_dict'])
    else:
        model.module.load_state_dict(ckpt['model_state_dict'])
    if 'ema_state_dict' in ckpt:
        ema.load_state_dict(ckpt['ema_state_dict'])
        if is_main: print("  Loaded EMA state from checkpoint")
    try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except Exception as e:
        if is_main: print(f"  Optimizer incompatible ({e}), fresh")
    if 'scheduler_state_dict' in ckpt:
        try: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except: pass
    start_epoch       = ckpt['epoch']
    best_val_loss     = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))
    epochs_no_improve = ckpt.get('epochs_no_improve', 0)
    if is_main: print(f"Resumed from epoch {ckpt['epoch']}, best={best_val_loss:.4f}")
else:
    start_epoch = 0
    if is_main: print("Starting fresh")

for epoch in range(start_epoch, NUM_EPOCHS):
    train_patch_ds.shuffle_patches(seed=epoch)
    train_sampler.set_epoch(epoch)

    model.train(); train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Train]",
                     leave=False, disable=not is_main):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=torch.bfloat16):
            loss = criterion(model(x), y)
        loss.backward(); optimizer.step()
        ema.update(model.module)
        train_loss += loss.item()

    t = torch.tensor(train_loss / len(train_loader), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG); avg_train = t.item()

    # Validate with EMA weights
    backup_sd = _swap_in_weights(model.module, ema.state_dict())

    model.eval(); val_loss = 0.0; vb = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val-EMA]",
                         leave=False, disable=not is_main):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast('cuda', dtype=torch.bfloat16):
                lv = criterion(model(x), y).item()
            if not math.isnan(lv) and not math.isinf(lv):
                val_loss += lv; vb += 1

    model.module.load_state_dict(backup_sd); del backup_sd

    vt = torch.tensor(val_loss / max(vb, 1), device=device)
    dist.all_reduce(vt, op=dist.ReduceOp.AVG); avg_val = vt.item()
    scheduler.step(avg_val)

    if is_main:
        torch.save({'epoch': epoch, 'model_state_dict': get_model_state_dict(model),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train, 'val_loss': avg_val,
                    'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve,
                    'num_classes': NUM_CLASSES, 'patch_size': PATCH_SIZE,
                    'train_configs': train_configs, 'val_configs': val_configs},
                   LAST_CHECKPOINT)
        saved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val; epochs_no_improve = 0
            # Best ckpt's `model_state_dict` IS the EMA weights -> 2_evaluate.py compatible
            torch.save({'epoch': epoch+1,
                        'model_state_dict': ema.state_dict(),
                        'live_state_dict':  get_model_state_dict(model),
                        'ema_state_dict':   ema.state_dict(),
                        'train_loss': avg_train, 'val_loss': avg_val,
                        'num_classes': NUM_CLASSES, 'patch_size': PATCH_SIZE,
                        'train_configs': train_configs, 'val_configs': val_configs},
                       BEST_MODEL_PATH)
            saved = " <- saved best (EMA)"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train: {avg_train:.4f} | Val(EMA): {avg_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}{saved}")
        history.append({'epoch': epoch+1, 'train_loss': avg_train,
                        'val_loss': avg_val, 'lr': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        with open(TRAIN_LOG_PATH, "w") as f:
            for row in history:
                f.write(f"Epoch {int(row['epoch']):3d}/{NUM_EPOCHS} | Train: {row['train_loss']:.4f} | "
                        f"Val(EMA): {row['val_loss']:.4f} | LR: {row['lr']:.2e}\n")
        if saved:
            with open(BEST_SUMMARY, "w") as f:
                f.write(f"Best epoch: {epoch+1}\nVal loss: {avg_val:.4f}  (EMA weights)\n"
                        f"Train loss: {avg_train:.4f}\nLR: {optimizer.param_groups[0]['lr']:.2e}\n"
                        f"Checkpoint: {BEST_MODEL_PATH}\n")

    stop = torch.tensor(int(epochs_no_improve >= PATIENCE), device=device)
    dist.broadcast(stop, src=0)
    if stop.item():
        if is_main: print(f"\nEarly stopping at epoch {epoch+1}")
        break
    st = torch.tensor([best_val_loss, float(epochs_no_improve)], device=device)
    dist.broadcast(st, src=0)
    best_val_loss = st[0].item(); epochs_no_improve = int(st[1].item())
    if (epoch + 1) % 20 == 0:
        torch.cuda.empty_cache(); gc.collect()

if is_main:
    print("=" * 70)
    print(f"Training complete! FPN-MiT-B4 V3 best val loss (EMA): {best_val_loss:.4f}")

dist.barrier()

# ============================================================
# EVALUATION on val set (EMA weights, rank 0)
# ============================================================
if is_main:
    print("\nEvaluating best model (EMA weights) on val set...")
    CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]
    ckpt       = torch.load(BEST_MODEL_PATH, map_location=device)
    model_eval = build_fpn_mitb4(num_classes=NUM_CLASSES)
    model_eval.load_state_dict(ckpt['model_state_dict'])
    model_eval = model_eval.to(device).eval()

    val_loader_eval = DataLoader(val_patch_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=False)
    tp = torch.zeros(NUM_CLASSES); fp = torch.zeros(NUM_CLASSES)
    fn = torch.zeros(NUM_CLASSES); present = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for x, y in tqdm(val_loader_eval, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            pred_cls = torch.argmax(model_eval(x), dim=1)
            valid    = (y != IGNORE_INDEX)
            for c in range(NUM_CLASSES):
                pc = (pred_cls == c) & valid; tc = (y == c) & valid
                if tc.sum() > 0:
                    tp[c] += (pc & tc).sum().float().cpu()
                    fp[c] += (pc & ~tc).sum().float().cpu()
                    fn[c] += (~pc & tc).sum().float().cpu()
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
    print(f"\nFPN-MiT-B4 V3 Val Mean IoU (EMA): {np.mean(valid_iou):.4f}")
    df.to_csv(os.path.join(LOG_DIR, "validation_metrics.csv"), index=False)

dist.destroy_process_group()
