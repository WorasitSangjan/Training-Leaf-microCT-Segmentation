"""
Leaf CT Segmentation — Phase 1 CLEAN Screening (parameterized)
====================================================================
Blind architecture screening under IDENTICAL standard settings for all models.
"Assume we know nothing" — no domain-specific tricks (no CLAHE, no per-model
tuning). Just a fair baseline comparison to select the best architectures for
Phase 2 tuning.

Standard blind settings (IDENTICAL for all 9 architectures):
  - Augmentation : HFlip + VFlip + Rotate(45) only (literature standard, NO CLAHE)
  - Loss         : weighted CE only (most basic baseline)
  - Class weights: [1.00, 4.15, 3.10, 1.50, 4.60] (new-dataset frequency)
  - LR           : 1e-4 UNIFORM AdamW (no differential — truly identical)
  - Scheduler    : ReduceLROnPlateau (factor=0.5, patience=7)
  - Epochs/Patience: 100 / 15
  - Patch/Stride : 320 / 160
  - Split        : species-level (clean, SEED=42, same as Phase 2)
  - Dataset      : new expanded (56 BL + 30 pine x3)

Difference from original (leaky) Phase 1:
  + Clean species-level split (no data leakage)
  + New expanded dataset (consistent with Phase 2)
  Everything else is standard/blind (no hindsight).

Usage:
  torchrun --nproc_per_node=4 1_train_multi_architecture.py --model eomt_vitl
  Valid --model: unet_resnet101, deeplab_efficientnet, deeplab_mitb4,
                 fpn_mitb4, fpn_mitb5, segformer, mask2former,
                 eomt_vitb, eomt_vitl
"""

import os, gc, json, math, random, argparse
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
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    choices=["unet_resnet101", "deeplab_efficientnet", "deeplab_mitb4",
                             "fpn_mitb4", "fpn_mitb5", "segformer", "mask2former",
                             "eomt_vitb", "eomt_vitl"])
parser.add_argument("--class-weights", nargs=5, type=float,
                    default=[1.00, 4.15, 3.10, 1.50, 4.60],
                    metavar=("BG", "EPI", "VASC", "MES", "AIR"),
                    help="Five class weights from 0_compute_class_weights.py")
args = parser.parse_args()
MODEL_TYPE = args.model

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
    print(f"World size: {world_size}  |  Model: {MODEL_TYPE}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ============================================================
# PATHS & CONFIG (new dataset, blind settings)
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
MODEL_DIR  = f"{OUTPUT_DIR}/phase1_clean_{MODEL_TYPE}"
LOG_DIR    = f"{OUTPUT_DIR}/phase1_clean_{MODEL_TYPE}_logs"
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

# ============================================================
# DATASET (float pipeline — no CLAHE in blind screening, so flip/rotate on float is fine)
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
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        valid = img_t[img_t > 0]
        if len(valid) > 0:
            mean, std = valid.mean(), valid.std()
            if std > 1e-5: img_t = (img_t - mean) / std
        return img_t, torch.from_numpy(mask).long()


class PatchDataset(Dataset):
    def __init__(self, base_dataset, patch_size=256, stride=128,
                 drop_background_only=True, augment=False):
        self.base   = base_dataset
        self.patch  = int(patch_size)
        self.stride = int(stride)
        self.drop_background_only = drop_background_only
        self.augment = augment
        # BLIND screening: flip + rotate only (literature standard, NO CLAHE)
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
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
        img, mask   = self.base[img_i]
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
# DATA LOADING — species-level split (clean, same SEED as Phase 2)
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
# MODEL WRAPPERS
# ============================================================
class SegFormerWrapper(nn.Module):
    def __init__(self, num_classes, pretrained="nvidia/segformer-b4-finetuned-ade-512-512"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.conv_in = nn.Conv2d(1, 3, kernel_size=1)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=num_classes,
            ignore_mismatched_sizes=True, local_files_only=True)

    def forward(self, x):
        B, C, H, W = x.shape
        if C == 1: x = self.conv_in(x)
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)


class Mask2FormerWrapper(nn.Module):
    def __init__(self, num_classes, pretrained="facebook/mask2former-swin-base-IN21k-ade-semantic"):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation
        self.num_classes = num_classes
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained, num_labels=num_classes,
            ignore_mismatched_sizes=True, local_files_only=True, use_safetensors=True)

    def forward(self, x):
        H, W  = x.shape[2], x.shape[3]
        x_rgb = x.expand(-1, 3, -1, -1)
        out   = self.model(pixel_values=x_rgb)
        class_probs = out.class_queries_logits[..., :self.num_classes].softmax(-1)
        masks       = F.interpolate(out.masks_queries_logits, size=(H, W),
                                    mode='bilinear', align_corners=False)
        return torch.einsum('bqc,bqhw->bchw', class_probs, masks.sigmoid())


class EoMT(nn.Module):
    def __init__(self, num_classes=5, num_queries=100, backbone_name="vitl"):
        super().__init__()
        from transformers import AutoModel
        self.conv_in = nn.Conv2d(1, 3, kernel_size=1)
        hf_ids = {"vitl": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                  "vitb": "facebook/dinov3-vitb16-pretrain-lvd1689m"}
        self.backbone    = AutoModel.from_pretrained(hf_ids[backbone_name], local_files_only=True)
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
        if C == 1: images = self.conv_in(images)
        hidden  = self.backbone(images).last_hidden_state
        spatial = hidden[:, 1:, :]
        q_tokens = self.q.weight[None].expand(B, -1, -1)
        q_out, _ = self.query_attn(q_tokens, spatial, spatial)
        class_logits = self.class_head(q_out)
        gh = H // self._patch_size; gw = W // self._patch_size
        spatial = spatial[:, :gh * gw, :]
        sp_grid = spatial.transpose(1, 2).reshape(B, self._embed_dim, gh, gw)
        sp_up   = self.upscale(sp_grid)
        masks_l = torch.einsum("bqd,bdhw->bqhw", self.mask_head(q_out), sp_up)
        sem     = torch.einsum("bqc,bqhw->bchw", class_logits[..., :-1], masks_l.sigmoid())
        return F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)


def build_model(model_type, num_classes):
    smp_configs = {
        "unet_resnet101":       ("Unet",         "resnet101"),
        "deeplab_efficientnet": ("DeepLabV3Plus", "efficientnet-b4"),
        "deeplab_mitb4":        ("DeepLabV3Plus", "mit_b4"),
        "fpn_mitb4":            ("FPN",           "mit_b4"),
        "fpn_mitb5":            ("FPN",           "mit_b5"),
    }
    if model_type in smp_configs:
        arch_name, encoder = smp_configs[model_type]
        arch = getattr(smp, arch_name)
        return arch(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=1, classes=num_classes)
    if model_type == "segformer":
        return SegFormerWrapper(num_classes=num_classes)
    if model_type == "mask2former":
        return Mask2FormerWrapper(num_classes=num_classes)
    if model_type == "eomt_vitl":
        return EoMT(num_classes=num_classes, backbone_name="vitl")
    if model_type == "eomt_vitb":
        return EoMT(num_classes=num_classes, backbone_name="vitb")
    raise ValueError(f"Unknown model: {model_type}")


def get_model_state_dict(m):
    return m.module.state_dict() if hasattr(m, 'module') else m.state_dict()

model = build_model(MODEL_TYPE, NUM_CLASSES).to(device)

# BLIND: weighted CE only, uniform 1e-4 AdamW, ReduceLROnPlateau (identical for all)
class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
if is_main:
    print(f"[CONFIG] class_weights={args.class_weights}")

def criterion(pred, mask):
    b, c, h, w = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, c)
    mask_flat = mask.view(-1)
    valid_idx = (mask_flat != IGNORE_INDEX)
    if valid_idx.sum() == 0:
        return torch.tensor(0.0, device=device)
    pred_valid    = pred_flat[valid_idx]
    mask_valid    = mask_flat[valid_idx]
    pixel_weights = class_weights[mask_valid]
    ce_per_pixel  = F.cross_entropy(pred_valid, mask_valid, reduction='none')
    return (ce_per_pixel * pixel_weights).mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
model     = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

if is_main:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n/1e6:.1f}M")
    print("=" * 70)
    print(f"[PHASE1-CLEAN {MODEL_TYPE}] BLIND screening: LR=1e-4 uniform, CE-only, "
          f"aug=flip+rotate45 (no CLAHE), ROP, 100ep/pat15, clean species split, new dataset")
    print("=" * 70)

# ============================================================
# TRAINING
# ============================================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_CHECKPOINT = os.path.join(MODEL_DIR, "last_checkpoint.pth")
HISTORY_CSV     = os.path.join(LOG_DIR, "training_history.csv")

best_val_loss = float('inf'); epochs_no_improve = 0; history = []
if is_main and os.path.exists(HISTORY_CSV):
    history = pd.read_csv(HISTORY_CSV).to_dict('records')

checkpoint_path = LAST_CHECKPOINT if os.path.exists(LAST_CHECKPOINT) else \
                  BEST_MODEL_PATH  if os.path.exists(BEST_MODEL_PATH)  else None
if checkpoint_path:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.module.load_state_dict(ckpt['model_state_dict'])
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
        train_loss += loss.item()

    t = torch.tensor(train_loss / len(train_loader), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG); avg_train = t.item()

    model.eval(); val_loss = 0.0; vb = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val]",
                         leave=False, disable=not is_main):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast('cuda', dtype=torch.bfloat16):
                lv = criterion(model(x), y).item()
            if not math.isnan(lv) and not math.isinf(lv):
                val_loss += lv; vb += 1

    vt = torch.tensor(val_loss / max(vb, 1), device=device)
    dist.all_reduce(vt, op=dist.ReduceOp.AVG); avg_val = vt.item()
    scheduler.step(avg_val)

    if is_main:
        torch.save({'epoch': epoch, 'model_state_dict': get_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train, 'val_loss': avg_val,
                    'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve,
                    'num_classes': NUM_CLASSES, 'patch_size': PATCH_SIZE, 'model_type': MODEL_TYPE},
                   LAST_CHECKPOINT)
        saved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val; epochs_no_improve = 0
            torch.save({'epoch': epoch+1, 'model_state_dict': get_model_state_dict(model),
                        'train_loss': avg_train, 'val_loss': avg_val,
                        'num_classes': NUM_CLASSES, 'patch_size': PATCH_SIZE,
                        'model_type': MODEL_TYPE}, BEST_MODEL_PATH)
            saved = " <- saved best"
        else:
            epochs_no_improve += 1
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}{saved}")
        history.append({'epoch': epoch+1, 'train_loss': avg_train,
                        'val_loss': avg_val, 'lr': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

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
    print(f"Training complete! {MODEL_TYPE} best val loss: {best_val_loss:.4f}")

dist.barrier()

# ============================================================
# EVALUATION on val set (rank 0)
# ============================================================
if is_main:
    print("\nEvaluating best model on val set...")
    CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]
    ckpt       = torch.load(BEST_MODEL_PATH, map_location=device)
    model_eval = build_model(MODEL_TYPE, NUM_CLASSES)
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
    rows = [{'Class': CLASS_NAMES[c], 'IoU': round(iou[c].item(), 4) if present[c] > 0 else 'N/A'}
            for c in range(NUM_CLASSES)]
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    valid_iou = [r['IoU'] for r in rows if r['IoU'] != 'N/A']
    print(f"\n{MODEL_TYPE} Val Mean IoU: {np.mean(valid_iou):.4f}")
    df.to_csv(os.path.join(LOG_DIR, "validation_metrics.csv"), index=False)

dist.destroy_process_group()
