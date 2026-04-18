"""
Leaf CT Segmentation — DeepLabV3+ + MiT-B4  [Phase 1]
===================================================
Phase 1: standard settings for fair architecture comparison.
- Aug: flip + rotate only (literature standard)
- Loss: weighted CE (inverse frequency class weights)
- LR: 1e-4 (standard AdamW default for pretrained encoders)
SLURM: torchrun --nproc_per_node=4 train_4_deeplab_mitb4.py
"""

import os, gc, json, math, random
import numpy as np
import pandas as pd
from PIL import Image

os.environ['TORCH_HOME']     = '/pscratch/sd/w/worasit/torch_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, DistributedSampler
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
    f"{SCRATCH}/configs/ab_vcarlsii.json",    f"{SCRATCH}/configs/ab_vcinnamo.json",
    f"{SCRATCH}/configs/ab_vdavidii.json",    f"{SCRATCH}/configs/ab_vdavidii2.json",
    f"{SCRATCH}/configs/ab_vdentatum.json",   f"{SCRATCH}/configs/ab_vdentatum2.json",
    f"{SCRATCH}/configs/ab_vfurcatum.json",   f"{SCRATCH}/configs/ab_vhartwegii.json",
    f"{SCRATCH}/configs/ab_vjaponicum.json",  f"{SCRATCH}/configs/ab_vjucundum.json",
    f"{SCRATCH}/configs/ab_vjucundum2.json",  f"{SCRATCH}/configs/ab_vlantana.json",
    f"{SCRATCH}/configs/ab_vlautum.json",     f"{SCRATCH}/configs/ab_vlautum3.json",
    f"{SCRATCH}/configs/ab_vpropinquum.json", f"{SCRATCH}/configs/ab_vtinus.json",
    f"{SCRATCH}/configs/devin1_no_bse.json",  f"{SCRATCH}/configs/devin1_with_bse.json",
    f"{SCRATCH}/configs/devin2.json",         f"{SCRATCH}/configs/devin3.json",
    f"{SCRATCH}/configs/jg_mag.json",         f"{SCRATCH}/configs/lf_arab.json",
    f"{SCRATCH}/configs/oak_ce.json",         f"{SCRATCH}/configs/oak_cf.json",
    f"{SCRATCH}/configs/oak_cr.json",         f"{SCRATCH}/configs/oak_ob.json",
    f"{SCRATCH}/configs/oak_ru.json",         f"{SCRATCH}/configs/oak_su.json",
    f"{SCRATCH}/configs/olive_d4.json",       f"{SCRATCH}/configs/olive_d5.json",
    f"{SCRATCH}/configs/olive_r1.json",       f"{SCRATCH}/configs/olive_w4.json",
    f"{SCRATCH}/configs/olive_w5.json",
]
PINE_CONFIGS = [
    f"{SCRATCH}/configs/st_pinus_lo1.json",     f"{SCRATCH}/configs/st_pinus_lo2.json",
    f"{SCRATCH}/configs/st_pinus_palus.json",   f"{SCRATCH}/configs/st_pinus_pb.json",
    f"{SCRATCH}/configs/st_pinus_pc.json",      f"{SCRATCH}/configs/st_pinus_pd.json",
    f"{SCRATCH}/configs/st_pinus_pe1.json",     f"{SCRATCH}/configs/st_pinus_pe2.json",
    f"{SCRATCH}/configs/st_pinus_pf1.json",     f"{SCRATCH}/configs/st_pinus_pf3.json",
    f"{SCRATCH}/configs/st_pinus_pg.json",      f"{SCRATCH}/configs/st_pinus_ph.json",
    f"{SCRATCH}/configs/st_pinus_pinaster.json",f"{SCRATCH}/configs/st_pinus_pinea.json",
    f"{SCRATCH}/configs/st_pinus_pj1.json",     f"{SCRATCH}/configs/st_pinus_pj2.json",
    f"{SCRATCH}/configs/st_pinus_pk.json",      f"{SCRATCH}/configs/st_pinus_pm2.json",
    f"{SCRATCH}/configs/st_pinus_pm5.json",     f"{SCRATCH}/configs/st_pinus_pn1.json",
    f"{SCRATCH}/configs/st_pinus_pn2.json",     f"{SCRATCH}/configs/st_pinus_ppd2.json",
    f"{SCRATCH}/configs/st_pinus_ppg1.json",    f"{SCRATCH}/configs/st_pinus_ppg2.json",
    f"{SCRATCH}/configs/st_pinus_pr1.json",     f"{SCRATCH}/configs/st_pinus_pse1.json",
    f"{SCRATCH}/configs/st_pinus_pth5.json",    f"{SCRATCH}/configs/st_pinus_tm10.json",
]
CONFIG_PATHS = BROADLEAF_CONFIGS + PINE_CONFIGS * 3

OUTPUT_DIR = f"{SCRATCH}/outputs"
MODEL_DIR  = f"{OUTPUT_DIR}/models_DeepLab_MitB4"
LOG_DIR    = f"{OUTPUT_DIR}/logs_DeepLab_MitB4"

if is_main:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

PATCH_SIZE   = 320
STRIDE       = 160
BATCH_SIZE   = 32
NUM_EPOCHS   = 100
LR           = 1e-4
VAL_FRAC     = 0.2
PATIENCE     = 15
NUM_CLASSES  = 5
IGNORE_INDEX = 254

# ============================================================
# DATASET
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
        if is_main:
            print(f"Loaded: {self.name} — {len(self.images)} images")

        self._img_cache, self._mask_cache = [], []
        for img_name, mask_name in zip(self.images, self.masks):
            img  = np.array(Image.open(os.path.join(self.image_dir, img_name)))
            mask = np.array(Image.open(os.path.join(self.mask_dir,  mask_name)))
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if img.ndim == 3:
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            H_img, W_img = img.shape
            H_mask, W_mask = mask.shape
            if H_img != H_mask or W_img != W_mask:
                new_mask = np.full((H_img, W_img), self.ignore_index, dtype=mask.dtype)
                ph, pw = min(H_img, H_mask), min(W_img, W_mask)
                new_mask[:ph, :pw] = mask[:ph, :pw]
                mask = new_mask
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
        if not tops  or tops[-1]  != max(0, H - self.patch): tops.append(max(0, H - self.patch))
        if not lefts or lefts[-1] != max(0, W - self.patch): lefts.append(max(0, W - self.patch))
        return tops, lefts

    def _build_index(self):
        self.patch_index = []
        for img_i in range(len(self.base)):
            img, mask = self.base[img_i]
            _, H, W = img.shape
            tops, lefts = self._grid_positions(H, W)
            for t in tops:
                for l in lefts:
                    mask_p = mask[t:min(t+self.patch, H), l:min(l+self.patch, W)]
                    uniq   = set(mask_p.flatten().tolist())
                    if self.drop_background_only and len(uniq) == 1 and 0 in uniq:
                        continue
                    if 1 in uniq or 2 in uniq or 4 in uniq:
                        self.patch_index.extend([(img_i, t, l)] * 4)
                    else:
                        self.patch_index.append((img_i, t, l))
        if is_main:
            print(f"Total valid patches: {len(self.patch_index)}")

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

        pad_h = self.patch - img_p.shape[1]
        pad_w = self.patch - img_p.shape[2]
        if pad_h > 0 or pad_w > 0:
            img_p  = F.pad(img_p,  (0, pad_w, 0, pad_h), value=0.0)
            pad_val = getattr(self.base.dataset if hasattr(self.base, 'dataset') else self.base, 'ignore_index', 254)
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
base_ds  = ConcatDataset(all_datasets)
val_size = int(len(base_ds) * VAL_FRAC)
trn_size = len(base_ds) - val_size
train_base, val_base = random_split(base_ds, [trn_size, val_size],
                                    generator=torch.Generator().manual_seed(SEED))

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
# MODEL
# ============================================================
def get_model_state_dict(m):
    return m.module.state_dict() if hasattr(m, 'module') else m.state_dict()

model = smp.DeepLabV3Plus(encoder_name="mit_b4", encoder_weights="imagenet",
                 in_channels=1, classes=NUM_CLASSES)
model = model.to(device)

# Class weights: inverse pixel frequency from training set
class_weights = torch.tensor([1.0, 4.25, 3.15, 1.55, 4.95]).to(device)

def criterion(pred, mask):
    b, c, h, w = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, c)
    mask_flat = mask.view(-1)
    valid_idx = (mask_flat != IGNORE_INDEX)
    if valid_idx.sum() > 0:
        pred_valid    = pred_flat[valid_idx]
        mask_valid    = mask_flat[valid_idx]
        pixel_weights = class_weights[mask_valid]
        ce_loss       = F.cross_entropy(pred_valid, mask_valid, reduction='none')
        return (ce_loss * pixel_weights).mean()
    return torch.tensor(0.0, device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
model     = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

if is_main:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n/1e6:.1f}M")

# ============================================================
# TRAINING
# ============================================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_CHECKPOINT = os.path.join(MODEL_DIR, "last_checkpoint.pth")
TRAIN_LOG_PATH  = os.path.join(LOG_DIR, "training_log.txt")
BEST_SUMMARY    = os.path.join(LOG_DIR, "best_summary.txt")
HISTORY_CSV     = os.path.join(LOG_DIR, "training_history.csv")

best_val_loss = float('inf'); epochs_no_improve = 0; history = []
if is_main and os.path.exists(HISTORY_CSV):
    history = pd.read_csv(HISTORY_CSV).to_dict('records')

if is_main:
    print("=" * 70)
    print(f"[CONFIG] model=DeepLabV3+/MiT-B4 LR={LR} wd=5e-3 patch={PATCH_SIZE} stride={STRIDE} "
          f"patience={PATIENCE} class_weights=[1.0,4.25,3.15,1.55,4.95] "
          f"aug=phase1(flip+rotate) loss=CE bg_only=True")
    print("=" * 70)

checkpoint_path = LAST_CHECKPOINT if os.path.exists(LAST_CHECKPOINT) else \
                  BEST_MODEL_PATH  if os.path.exists(BEST_MODEL_PATH)  else None

if checkpoint_path:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.module.load_state_dict(ckpt['model_state_dict'])
    try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except Exception as e:
        if is_main: print(f"  Optimizer incompatible ({e}), fresh optimizer")
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

    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Train]", leave=False, disable=not is_main):
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
        for x, y in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val]", leave=False, disable=not is_main):
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
                    'num_classes': NUM_CLASSES, 'patch_size': PATCH_SIZE, 'configs': CONFIG_PATHS},
                   LAST_CHECKPOINT)

        saved_str = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val; epochs_no_improve = 0
            torch.save({'epoch': epoch+1, 'model_state_dict': get_model_state_dict(model),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': avg_train, 'val_loss': avg_val,
                        'num_classes': NUM_CLASSES, 'patch_size': PATCH_SIZE, 'configs': CONFIG_PATHS},
                       BEST_MODEL_PATH)
            saved_str = " <- saved best"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}{saved_str}")

        history.append({'epoch': epoch+1, 'train_loss': avg_train,
                        'val_loss': avg_val, 'lr': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
        with open(TRAIN_LOG_PATH, "w") as f:
            for row in history:
                f.write(f"Epoch {int(row['epoch']):3d}/{NUM_EPOCHS} | Train: {row['train_loss']:.4f} | "
                        f"Val: {row['val_loss']:.4f} | LR: {row['lr']:.2e}\n")
        if saved_str:
            with open(BEST_SUMMARY, "w") as f:
                f.write(f"Best epoch: {epoch+1}\nVal loss: {avg_val:.4f}\n"
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
    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")

dist.barrier()

# ============================================================
# EVALUATION (rank 0 only)
# ============================================================
if is_main:
    print("\n" + "=" * 70 + "\nEvaluating best model...")
    CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]

    ckpt       = torch.load(BEST_MODEL_PATH, map_location=device)
    model_eval = smp.DeepLabV3Plus(encoder_name="mit_b4", encoder_weights=None, in_channels=1, classes=NUM_CLASSES)
    model_eval.load_state_dict(ckpt['model_state_dict'])
    model_eval = model_eval.to(device).eval()
    print(f"Loaded epoch {ckpt['epoch']} (val loss: {ckpt['val_loss']:.4f})")

    val_loader_eval = DataLoader(val_patch_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=False)
    tp = torch.zeros(NUM_CLASSES); fp = torch.zeros(NUM_CLASSES)
    fn = torch.zeros(NUM_CLASSES); present = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for x, y in tqdm(val_loader_eval, desc="Evaluating"):
            x, y     = x.to(device), y.to(device)
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
             'IoU':   round(iou[c].item(), 4) if present[c] > 0 else 'N/A',
             'Dice':  round(dice[c].item(), 4) if present[c] > 0 else 'N/A'}
            for c in range(NUM_CLASSES)]

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    valid_iou = [r['IoU'] for r in rows if r['IoU'] != 'N/A']
    print(f"\nMean IoU: {np.mean(valid_iou):.4f}")
    df.to_csv(os.path.join(LOG_DIR, "validation_metrics.csv"), index=False)

dist.destroy_process_group()
