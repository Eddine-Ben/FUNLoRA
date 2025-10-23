import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import cv2
import numpy as np
from tqdm import tqdm

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import (
    create_logger, set_random_seed, setup_cuda_optimizations, cleanup_memory,
    monitor_system_resources, download_file, can_use_torch_compile
)
from build_model_fixed import (
    create_model, add_lora_to_backbone, enable_mit_gradient_checkpointing,
    param_groups, is_head_param, is_lora_param, set_requires_grad_named, group_frozen_backbone_modules
)
from weights import MODEL_CONFIGS, PRETRAIN_URLS, variant_to_backbone_name, filter_backbone_from_mmseg_ckpt


# -------------------------
# CLI / Config
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='SegFormer Cityscapes Fine-tuning with LoRA and Fisher-guided Unfreezing'
    )

    # Model configuration
    parser.add_argument('--model-variant', choices=['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B5_FAST', 'B5_HQ', 'B5_2X2'],
                        default='B3', help='SegFormer variant to train')
    parser.add_argument('--dataset-root', required=True, help='Cityscapes dataset root directory')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--num-workers-train', type=int, default=2, help='Number of training data workers')
    parser.add_argument('--num-workers-val', type=int, default=2, help='Number of validation data workers')

    # Learning rates
    parser.add_argument('--head-lr', type=float, default=4e-5, help='Learning rate for head layers')
    parser.add_argument('--backbone-lr', type=float, default=0.0, help='Learning rate for backbone (frozen phase)')
    parser.add_argument('--backbone-lr-thawed', type=float, default=1e-5, help='Learning rate for backbone (unfrozen)')
    parser.add_argument('--lora-lr', type=float, default=5e-5, help='Learning rate for LoRA adapters')

    # LoRA configuration
    parser.add_argument('--use-lora', action='store_true', default=True, help='Enable LoRA adapters')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout rate')

    # Fisher-guided unfreezing
    parser.add_argument('--use-fun-unfreeze', action='store_true', default=True,
                        help='Enable Fisher-guided unfreezing')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=10,
                        help='Epochs to keep backbone frozen (head warmup)')
    parser.add_argument('--fisher-compute-interval', type=int, default=3,
                        help='Interval (epochs) for Fisher information computation')
    parser.add_argument('--fisher-batches', type=int, default=12,
                        help='Number of batches for Fisher estimation')
    parser.add_argument('--max-groups-per-unfreeze', type=int, default=1,
                        help='Maximum groups to unfreeze per Fisher cycle')

    # Memory and optimization
    parser.add_argument('--grad-accum-steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Enable automatic mixed precision')
    parser.add_argument('--use-compile', action='store_true', default=True, help='Enable torch.compile')
    parser.add_argument('--resize-short', type=int, default=None,
                        help='Resize shorter side to this value (memory optimization)')
    parser.add_argument('--crop-size', type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help='Crop size (H W) for memory optimization')
    parser.add_argument('--center-crop', action='store_true', help='Use center crop for validation')

    # Checkpointing and logging
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Checkpoint saving interval')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint for resuming')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for logs and checkpoints')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--persistent-workers', action='store_true', help='Use persistent workers')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor for DataLoader')

    return parser.parse_args()


def create_config_from_args(args):
    return {
        'model_variant': args.model_variant,
        'dataset_root': args.dataset_root,
        'batch_size': args.batch_size,
        'num_workers_train': args.num_workers_train,
        'num_workers_val': args.num_workers_val,
        'persistent_workers': args.persistent_workers,
        'prefetch_factor': args.prefetch_factor,
        'num_epochs': args.num_epochs,
        'checkpoint_interval': args.checkpoint_interval,
        'grad_accum_steps': args.grad_accum_steps,
        'use_amp': args.use_amp,
        'use_compile': args.use_compile,
        'freeze_backbone_epochs': args.freeze_backbone_epochs,
        'head_lr': args.head_lr,
        'backbone_lr': args.backbone_lr,
        'backbone_lr_thawed': args.backbone_lr_thawed,
        'lora_lr': args.lora_lr,
        'use_lora': args.use_lora,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'use_fun_unfreeze': args.use_fun_unfreeze,
        'fisher_compute_interval': args.fisher_compute_interval,
        'fisher_batches': args.fisher_batches,
        'max_groups_per_unfreeze': args.max_groups_per_unfreeze,
        'resize_short': args.resize_short,
        'crop_size': tuple(args.crop_size) if args.crop_size else None,
        'center_crop': args.center_crop,
        'output_dir': args.output_dir,
        'seed': args.seed,
        'resume': args.resume,
    }


# -------------------------
# Constants
# -------------------------
CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 19
IGNORE_IDX = 255
CS_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


# -------------------------
# Cityscapes dataset (resize + crop capable)
# -------------------------
class CityscapesDataset(Dataset):
    def __init__(self, root, split="train", resize_short=None, crop_size=None, center_crop=False):
        self.root = Path(root)
        self.img_dir = self.root / "leftImg8bit" / split
        self.mask_dir = self.root / "gtFine" / split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"❌ Images folder not found: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"❌ Masks folder not found: {self.mask_dir}")

        self.resize_short = int(resize_short) if resize_short else None
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.center_crop = bool(center_crop)

        self.samples = []
        self.mean = th.tensor(CITYSCAPES_MEAN).view(3, 1, 1)
        self.std = th.tensor(CITYSCAPES_STD).view(3, 1, 1)

        for city in sorted(os.listdir(self.img_dir)):
            img_city_dir = self.img_dir / city
            mask_city_dir = self.mask_dir / city
            if not img_city_dir.is_dir():
                continue
            for fname in sorted(os.listdir(img_city_dir)):
                if fname.endswith("_leftImg8bit.png"):
                    img_path = img_city_dir / fname
                    mask_name = fname.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
                    mask_path = mask_city_dir / mask_name
                    if mask_path.exists():
                        self.samples.append((str(img_path), str(mask_path)))
        print(f"🗂️ {split}: {len(self.samples)} samples found")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resize_short(img, mask, short_side):
        if short_side is None or short_side <= 0:
            return img, mask
        h, w = img.shape[:2]
        s = short_side / float(min(h, w))
        new_w, new_h = int(round(w * s)), int(round(h * s))
        if (new_w, new_h) != (w, h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return img, mask

    @staticmethod
    def _ensure_min_size(img, mask, th_h, th_w):
        H, W = img.shape[:2]
        if H >= th_h and W >= th_w:
            return img, mask
        scale = max(th_h / float(H), th_w / float(W))
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def _crop(self, img, mask, size, center=False):
        if size is None:
            return img, mask
        ch, cw = size
        H, W = img.shape[:2]
        if H < ch or W < cw:
            img, mask = self._ensure_min_size(img, mask, ch, cw)
            H, W = img.shape[:2]
        if center:
            top = (H - ch) // 2
            left = (W - cw) // 2
        else:
            top = np.random.randint(0, H - ch + 1)
            left = np.random.randint(0, W - cw + 1)
        img = img[top:top + ch, left:left + cw]
        mask = mask[top:top + ch, left:left + cw]
        return img, mask

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        try:
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            assert img is not None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            assert mask is not None
        except Exception:
            img = np.zeros((1024, 2048, 3), dtype=np.uint8)
            mask = np.full((1024, 2048), IGNORE_IDX, dtype=np.uint8)

        img, mask = self._resize_short(img, mask, self.resize_short)
        img, mask = self._crop(img, mask, self.crop_size, center=self.center_crop)

        img = th.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img = (img - self.mean) / self.std
        mask = th.from_numpy(mask).long()
        return img, mask


def _worker_init_fn(_):
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


# -------------------------
# Trainer with classic FT + optional FUN-LoRA
# -------------------------
class Trainer:
    def __init__(self, model, cfg, device, logger):
        self.m = model.to(device)
        self.cfg = cfg
        self.d = device
        self.log = logger

        cw = th.tensor(
            [0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023,
             0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507],
            dtype=th.float32, device=device
        )
        self.crit = nn.CrossEntropyLoss(weight=cw, ignore_index=IGNORE_IDX)

        self.use_amp = bool(cfg['use_amp'])
        if self.use_amp:
            try:
                self.autocast = th.autocast(device_type="cuda", dtype=th.float16)
                self.scaler = th.amp.GradScaler("cuda")
            except Exception:
                self.autocast = th.cuda.amp.autocast()
                self.scaler = th.cuda.amp.GradScaler(enabled=True)
        else:
            self.autocast = nullcontext()
            self.scaler = None

        self._apply_initial_freeze()

        self.opt = optim.AdamW(
            param_groups(self.m, cfg['head_lr'], cfg['lora_lr'], cfg['backbone_lr']),
            betas=(0.9, 0.999), eps=1e-8
        )
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="max", factor=0.9, patience=20, verbose=True, min_lr=1e-7, eps=1e-8
        )

        self._compiled = False
        try:
            if cfg.get("use_compile", True) and can_use_torch_compile():
                self.m = th.compile(self.m, backend="inductor")
                self._compiled = True
                print("⚙️  torch.compile enabled (inductor)")
            else:
                print("ℹ️  torch.compile disabled (Windows or Triton not available).")
        except Exception as e:
            print(f"ℹ️  torch.compile unavailable ({e}); continuing without.")

        self.best_iou = 0.0
        self.best_state = None
        self.ep = 0

    def _apply_initial_freeze(self):
        set_requires_grad_named(self.m, lambda n: True, False)
        set_requires_grad_named(self.m, is_head_param, True)
        if self.cfg['use_lora']:
            set_requires_grad_named(self.m, is_lora_param, True)
        self.log.info("Phase-1: head (and LoRA if enabled) trainable; backbone frozen.")

    def _move_logits_to_mask_size(self, logits, masks):
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def _forward_loss(self, imgs, masks):
        with self.autocast:
            logits = self.m(imgs)
            logits = self._move_logits_to_mask_size(logits, masks)
            loss = self.crit(logits, masks.long())
        preds = logits.argmax(1)
        return loss, preds

    @staticmethod
    def _batch_metrics(preds, masks, num_classes=NUM_CLASSES):
        valid = masks != IGNORE_IDX
        miou_agg = []
        for c in range(num_classes):
            p = (preds == c) & valid
            t = (masks == c) & valid
            inter = int((p & t).sum().item())
            union = int((p | t).sum().item())
            miou_agg.append(inter / union if union > 0 else 0.0)
        pix_acc = float((preds[valid] == masks[valid]).float().mean().item())
        return float(np.mean(miou_agg)), pix_acc

    def _maybe_switch_to_phase2(self, epoch_idx):
        if epoch_idx == self.cfg['freeze_backbone_epochs']:
            print("🔓 Switching to full E2E: backbone unfrozen (classic schedule).")
            set_requires_grad_named(self.m, lambda n: not is_head_param(n) and not is_lora_param(n), True)
            self.opt = optim.AdamW(
                param_groups(self.m, self.cfg['head_lr'], self.cfg['lora_lr'], self.cfg['backbone_lr_thawed']),
                betas=(0.9, 0.999), eps=1e-8
            )
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, mode="max", factor=0.9, patience=20, verbose=True, min_lr=1e-7, eps=1e-8
            )

    def _compute_fisher(self, loader, max_batches=8):
        self.m.eval()
        groups = group_frozen_backbone_modules(self.m)
        if not groups:
            return {}
        tmp_params = []
        for g in groups.values():
            for p in g['params']:
                if not p.requires_grad:
                    p.requires_grad = True
                    tmp_params.append(p)
        tmp_opt = optim.SGD(tmp_params, lr=1.0) if tmp_params else None
        scores = {name: 0.0 for name in groups.keys()}
        batches = 0
        for imgs, masks in loader:
            imgs = imgs.to(self.d, non_blocking=True).to(memory_format=th.channels_last)
            masks = masks.to(self.d, non_blocking=True)
            if tmp_opt:
                tmp_opt.zero_grad(set_to_none=True)
            loss, _ = self._forward_loss(imgs, masks)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if tmp_opt:
                    self.scaler.unscale_(tmp_opt)
            else:
                loss.backward()
            for name, g in groups.items():
                s = 0.0
                for p in g['params']:
                    if p.grad is None:
                        continue
                    g2 = th.clamp(p.grad, min=-1e3, max=1e3).pow(2).sum().item()
                    s += g2
                scores[name] += s
            if tmp_opt:
                tmp_opt.zero_grad(set_to_none=True)
            batches += 1
            if batches >= max_batches:
                break
        for p in tmp_params:
            p.requires_grad = False
            p.grad = None
        self.m.train()
        return scores

    def _fun_unfreeze_step(self, train_loader):
        if not self.cfg['use_fun_unfreeze']:
            return
        if self.ep < self.cfg['freeze_backbone_epochs']:
            return
        if (self.ep % self.cfg['fisher_compute_interval']) != 0:
            return
        print("🧠 Fisher pass on frozen backbone groups …")
        scores = self._compute_fisher(train_loader, max_batches=self.cfg['fisher_batches'])
        if not scores:
            print("ℹ️  No frozen groups found.")
            return
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:self.cfg['max_groups_per_unfreeze']]
        to_unfreeze = {n for n, _ in top}
        unfrozen = 0
        for name, module in self.m.named_modules():
            if name in to_unfreeze:
                for p in module.named_parameters(recurse=False):
                    p[1].requires_grad = True
                    unfrozen += p[1].numel()
        self.opt = optim.AdamW(
            param_groups(self.m, self.cfg['head_lr'], self.cfg['lora_lr'], self.cfg['backbone_lr_thawed']),
            betas=(0.9, 0.999), eps=1e-8
        )
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="max", factor=0.9, patience=20, verbose=True, min_lr=1e-7, eps=1e-8
        )
        print(f"🔓 FUN: unfroze {len(to_unfreeze)} group(s) ≈ {unfrozen:,} params.")

    def train_epoch(self, loader):
        self.m.train()
        loss_sum = 0.0
        tot_corr = 0
        tot_pix = 0
        batch_ious = []
        accum = max(1, self.cfg['grad_accum_steps'])
        micro = 0
        self.opt.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"🚂 Train {self.ep + 1}", dynamic_ncols=True)
        for imgs, masks in pbar:
            imgs = imgs.to(self.d, non_blocking=True).to(memory_format=th.channels_last)
            masks = masks.to(self.d, non_blocking=True)
            try:
                loss, preds = self._forward_loss(imgs, masks)
                loss = loss / accum
            except RuntimeError as e:
                msg = str(e).lower()
                print(f"⚠️ Forward error: {e} • skipping batch")
                if "out of memory" in msg or "cuda error" in msg:
                    try:
                        del loss
                    except Exception:
                        pass
                    cleanup_memory()
                continue

            try:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro += 1
                if micro == accum:
                    if self.use_amp:
                        self.scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.m.parameters(), 0.5)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.m.parameters(), 0.5)
                        self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    micro = 0
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda error" in msg:
                    print("⚠️ OOM during backward — clearing cache and skipping batch")
                    try:
                        if self.use_amp:
                            self.scaler.update()
                    except Exception:
                        pass
                    try:
                        del loss, preds
                    except Exception:
                        pass
                    cleanup_memory()
                else:
                    print(f"⚠️ Backward/step error: {e}")
                self.opt.zero_grad(set_to_none=True)
                micro = 0
                continue

            if th.isfinite(loss):
                loss_sum += loss.item() * accum

            with th.no_grad():
                valid = masks != IGNORE_IDX
                tot_corr += (preds[valid] == masks[valid]).sum().item()
                tot_pix += valid.sum().item()
                miou, _ = self._batch_metrics(preds, masks)
                batch_ious.append(miou)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(tot_corr / tot_pix) if tot_pix > 0 else 0.0:.4f}',
                'iou': f'{(np.mean(batch_ious) if batch_ious else 0.0):.4f}',
                'lr': f"{self.opt.param_groups[0]['lr']:.2e}"
            })

        if micro > 0:
            try:
                if self.use_amp:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.m.parameters(), 0.5)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.m.parameters(), 0.5)
                    self.opt.step()
            except RuntimeError as e:
                print(f"⚠️ Step error at flush: {e}")
            finally:
                self.opt.zero_grad(set_to_none=True)

        avg_loss = loss_sum / max(1, len(loader))
        acc = (tot_corr / tot_pix) if tot_pix > 0 else 0.0
        iou = float(np.mean(batch_ious)) if batch_ious else 0.0
        print(f"🧰 Train • loss {avg_loss:.4f} • acc {acc:.4f} • mIoU {iou:.4f}")
        return avg_loss, acc, iou

    @th.no_grad()
    def validate(self, loader):
        self.m.eval()
        loss_sum = 0.0
        tot_corr = 0
        tot_pix = 0
        batch_ious = []
        inter = [0] * NUM_CLASSES
        union = [0] * NUM_CLASSES
        pbar = tqdm(loader, desc="🧪 Valid", dynamic_ncols=True)
        for imgs, masks in pbar:
            imgs = imgs.to(self.d, non_blocking=True).to(memory_format=th.channels_last)
            masks = masks.to(self.d, non_blocking=True)
            loss, preds = self._forward_loss(imgs, masks)
            valid = masks != IGNORE_IDX
            tot_corr += (preds[valid] == masks[valid]).sum().item()
            tot_pix += valid.sum().item()
            loss_sum += loss.item()
            miou, _ = self._batch_metrics(preds, masks)
            batch_ious.append(miou)
            for c in range(NUM_CLASSES):
                p = (preds == c) & valid
                t = (masks == c) & valid
                inter[c] += int((p & t).sum().item())
                union[c] += int((p | t).sum().item())
            acc = (tot_corr / tot_pix) if tot_pix > 0 else 0.0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}',
                              'iou': f'{(np.mean(batch_ious) if batch_ious else 0.0):.4f}'})
        pix_acc = (tot_corr / tot_pix) if tot_pix > 0 else 0.0
        avg_loss = loss_sum / max(1, len(loader))
        class_ious = [(i / u if u > 0 else 0.0) for i, u in zip(inter, union)]
        miou_all = float(np.mean(class_ious))
        print(f"✅ Val • loss {avg_loss:.4f} • acc {pix_acc:.4f} • mIoU {miou_all:.4f}")
        return {'val_loss': avg_loss, 'pixel_accuracy': pix_acc, 'mean_iou': miou_all, 'class_ious': class_ious}

    def train(self, train_loader, val_loader, num_epochs, variant, start_epoch=0):
        best = 0.0
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logf = f"training_log_{variant}_{ts}.csv"
        with open(logf, 'w') as f:
            f.write("epoch,train_loss,train_acc,train_iou,val_loss,val_acc,val_iou,lr\n")

        for self.ep in range(start_epoch, num_epochs):
            print(f"\n📈 Epoch {self.ep + 1}/{num_epochs} • {monitor_system_resources()}")
            self._maybe_switch_to_phase2(self.ep)
            self._fun_unfreeze_step(train_loader)
            tl, ta, ti = self.train_epoch(train_loader)
            vm = self.validate(val_loader)
            self.sched.step(vm['mean_iou'])

            print("📚 LR groups:")
            for g in self.opt.param_groups:
                print(f"   • {g.get('name', 'group')} : {g['lr']:.2e}")

            if vm['mean_iou'] > best:
                best = vm['mean_iou']
                self.best_iou = best
                th.save(self.m.state_dict(), f"segformer_{variant}_best.pth")
                print(f"💾 New best saved! mIoU {best:.4f}")

            with open(logf, 'a') as f:
                f.write(f"{self.ep + 1},{tl:.6f},{ta:.6f},{ti:.6f},{vm['val_loss']:.6f},"
                        f"{vm['pixel_accuracy']:.6f},{vm['mean_iou']:.6f},{self.opt.param_groups[0]['lr']:.2e}\n")

            if (self.ep + 1) % self.cfg['checkpoint_interval'] == 0:
                ckpt = {
                    'epoch': self.ep + 1,
                    'model_state_dict': self.m.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'scheduler_state_dict': self.sched.state_dict(),
                    'best_iou': self.best_iou,
                    'config': self.cfg
                }
                if self.scaler is not None:
                    try:
                        ckpt['scaler_state_dict'] = self.scaler.state_dict()
                    except Exception:
                        pass
                name = f"checkpoint_epoch_{self.ep + 1}.pth"
                th.save(ckpt, name)
                print(f"🧩 Checkpoint saved: {name}")

        print(f"🏁 Done. Best mIoU {best:.4f}")
        return best


# -------------------------
# Resume utility
# -------------------------
def load_checkpoint_if_any(trainer: Trainer, resume_path: str, device: th.device) -> int:
    if not resume_path or not os.path.exists(resume_path):
        return 0
    print(f"🔁 Resuming from: {resume_path}")
    ckpt = th.load(resume_path, map_location=device)
    trainer.m.load_state_dict(ckpt['model_state_dict'], strict=True)
    last_epoch = int(ckpt.get('epoch', 0))
    if last_epoch >= trainer.cfg['freeze_backbone_epochs']:
        set_requires_grad_named(trainer.m, lambda n: not is_head_param(n) and not is_lora_param(n), True)
        trainer.opt = optim.AdamW(
            param_groups(trainer.m, trainer.cfg['head_lr'], trainer.cfg['lora_lr'], trainer.cfg['backbone_lr_thawed']),
            betas=(0.9, 0.999), eps=1e-8
        )
    else:
        trainer._apply_initial_freeze()
        trainer.opt = optim.AdamW(
            param_groups(trainer.m, trainer.cfg['head_lr'], trainer.cfg['lora_lr'], trainer.cfg['backbone_lr']),
            betas=(0.9, 0.999), eps=1e-8
        )
    trainer.sched = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.opt, mode="max", factor=0.9, patience=20, verbose=True, min_lr=1e-7, eps=1e-8
    )
    if 'optimizer_state_dict' in ckpt:
        try:
            trainer.opt.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠️ opt state skipped: {e}")
    if 'scheduler_state_dict' in ckpt:
        try:
            trainer.sched.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception as e:
            print(f"⚠️ sched state skipped: {e}")
    if trainer.scaler is not None and 'scaler_state_dict' in ckpt:
        try:
            trainer.scaler.load_state_dict(ckpt['scaler_state_dict'])
        except Exception as e:
            print(f"⚠️ scaler state skipped: {e}")
    trainer.best_iou = float(ckpt.get('best_iou', 0.0))
    print(f"✅ Resume OK • last finished epoch = {last_epoch} • next = {last_epoch + 1}")
    return last_epoch


# -------------------------
# Main (library entry)
# -------------------------
def main(config):
    os.makedirs('weights', exist_ok=True)
    set_random_seed(config.get('seed', 42))

    if sys.platform.startswith("win"):
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    root = Path(config['dataset_root'])
    if not (root / "leftImg8bit" / "train").exists() or not (root / "gtFine" / "train").exists():
        raise FileNotFoundError(f"❌ Cityscapes path looks wrong: {root}")

    logger = create_logger('segformer_fun_lora_training')
    setup_cuda_optimizations()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"🧭 Device: {device}")

    cleanup_memory()

    variant = config['model_variant'].upper()
    model_cfg = MODEL_CONFIGS.get(variant, MODEL_CONFIGS['B0'])
    weight_path = model_cfg['weight_file']
    backbone_name = variant_to_backbone_name(variant)

    model = create_model(backbone=backbone_name, out_channels=NUM_CLASSES, pretrained=not os.path.exists(weight_path))

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if getattr(m, 'weight', None) is not None:
                nn.init.constant_(m.weight, 1)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(_init_weights)

    if not os.path.exists(weight_path):
        url_key = variant if variant in PRETRAIN_URLS else "B0"
        print(f"🌐 Downloading ADE20K weights for {variant} …")
        try:
            download_file(PRETRAIN_URLS[url_key], weight_path)
        except Exception as e:
            print(f"⚠️ Download failed: {e} • continuing with random/ImageNet init")

    if os.path.exists(weight_path):
        try:
            try:
                ckpt = th.load(weight_path, map_location='cpu', weights_only=True)
            except Exception:
                ckpt = th.load(weight_path, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt)
            msd = model.state_dict()
            filt = filter_backbone_from_mmseg_ckpt(sd, msd)
            model.load_state_dict(filt, strict=False)
            print(f"📦 Loaded {len(filt)} ADE20K params (backbone)")
        except Exception as e:
            print(f"⚠️ ADE20K load failed: {e} • using fresh init")

    if config['use_lora']:
        try:
            model = add_lora_to_backbone(
                model,
                r=config['lora_r'],
                alpha=config['lora_alpha'],
                dropout=config['lora_dropout']
            )
            print("✨ LoRA adapters injected")
        except Exception as e:
            print(f"⚠️ LoRA requested but could not be applied: {e}")
            config['use_lora'] = False

    enable_mit_gradient_checkpointing(model)
    model = model.to(device)

    train_ds = CityscapesDataset(
        root=config['dataset_root'], split='train',
        resize_short=config['resize_short'], crop_size=config['crop_size'], center_crop=False
    )
    val_ds = CityscapesDataset(
        root=config['dataset_root'], split='val',
        resize_short=config['resize_short'], crop_size=config['crop_size'], center_crop=True
    )

    def _dl_kwargs(bs, shuffle, workers, pin, persist, drop, pf):
        if os.name == 'nt':  # Windows
            workers = 0
            persist = False
            pf = None
        kw = dict(
            batch_size=bs,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=(pin and th.cuda.is_available()),
            persistent_workers=(persist and workers > 0),
            drop_last=drop,
            worker_init_fn=_worker_init_fn if workers > 0 else None
        )
        if workers > 0 and pf is not None:
            kw['prefetch_factor'] = pf
        return kw

    train_loader = DataLoader(
        train_ds,
        **_dl_kwargs(
            config['batch_size'],
            True,
            config['num_workers_train'],
            True,
            config['persistent_workers'],
            True,
            config['prefetch_factor']
        )
    )
    val_loader = DataLoader(
        val_ds,
        **_dl_kwargs(
            config['batch_size'],
            False,
            config['num_workers_val'],
            True,
            config['persistent_workers'],
            False,
            config['prefetch_factor']
        )
    )

    print("🔎 Sanity forward …")
    model.eval()
    with th.no_grad():
        x, _ = train_ds[0]
        h, w = x.shape[-2], x.shape[-1]
        y = model(th.randn(1, 3, h, w, device=device).to(memory_format=th.channels_last))
        print("✅ Output shape:", tuple(y.shape))
    model.train()

    trainer = Trainer(model, config, device, logger)

    start_epoch = 0
    resume_path = config.get('resume')
    if resume_path:
        start_epoch = load_checkpoint_if_any(trainer, resume_path, device)

    print("🏋️ Starting training …")
    best = trainer.train(train_loader, val_loader, config['num_epochs'], variant=variant, start_epoch=start_epoch)
    print(f"🏆 Best mIoU: {best:.4f}")
