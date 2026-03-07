"""
train.py — Training Engine + TTA + Ensemble
=============================================
Contains:
  Trainer                — training loop with AMP + gradient accumulation
  TestTimeAugmentation   — "amplification during testing"
  EnsemblePredictor      — integrate multiple models
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from metrics import SegmentationMetrics


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 device, cfg, experiment_name="experiment"):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg

        tc = cfg['training']
        self.epochs = tc['epochs']
        self.accum_steps = tc['accum_steps']
        self.use_amp = tc['amp']
        self.grad_clip = tc['grad_clip']
        self.patience = tc['early_stop_patience']
        self.num_classes = cfg['data']['num_classes']

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.exp_dir = os.path.join("experiments", experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.history = {
            "train_loss": [], "val_miou": [], "val_oa": [],
            "val_f1": [], "lr": [], "epoch_time": [],
        }
        self.best_miou = 0.0
        self.best_epoch = 0
        self.no_improve = 0

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        num_batches_total = len(train_loader)
        pbar = tqdm(train_loader, desc="  Train", leave=False)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # For the last partial accumulation group, scale by actual count
            remaining = num_batches_total - (i // self.accum_steps) * self.accum_steps
            effective_accum = min(self.accum_steps, remaining)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                scaled_loss = loss / effective_accum

            self.scaler.scale(scaled_loss).backward()

            if (i + 1) % self.accum_steps == 0 or (i + 1) == num_batches_total:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss/n_batches:.4f}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        metrics = SegmentationMetrics(self.num_classes)

        for images, masks in tqdm(val_loader, desc="  Val  ", leave=False):
            images = images.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
            preds = torch.argmax(logits, dim=1).cpu()
            metrics.update(preds, masks)

        return metrics.compute()

    def fit(self, train_loader, val_loader):
        print(f"\n{'='*60}")
        print(f"  Training: {self.exp_dir}")
        print(f"  Epochs: {self.epochs}, Effective batch: "
              f"{self.cfg['training']['batch_size']}×{self.accum_steps}")
        print(f"  Loss: {self.criterion}")
        print(f"{'='*60}\n")

        best_metrics = None

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_miou = val_metrics['mIoU']

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0

            self.history['train_loss'].append(train_loss)
            self.history['val_miou'].append(val_miou)
            self.history['val_oa'].append(val_metrics['OA'])
            self.history['val_f1'].append(val_metrics['macro_F1'])
            self.history['lr'].append(lr)
            self.history['epoch_time'].append(elapsed)

            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  Epoch {epoch:3d}/{self.epochs} │ "
                  f"Loss: {train_loss:.4f} │ mIoU: {val_miou:.2f}% │ "
                  f"OA: {val_metrics['OA']:.2f}% │ LR: {lr:.6f} │ "
                  f"VRAM: {vram:.1f}GB │ {elapsed:.0f}s")

            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.best_epoch = epoch
                self.no_improve = 0
                best_metrics = val_metrics
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_miou': self.best_miou,
                    'metrics': val_metrics,
                }, os.path.join(self.exp_dir, "best_model.pth"))
                print(f"    ✓ New best mIoU: {val_miou:.2f}%")
            else:
                self.no_improve += 1

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

            torch.cuda.empty_cache()

        with open(os.path.join(self.exp_dir, "history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)
        if best_metrics:
            with open(os.path.join(self.exp_dir, "best_metrics.json"), 'w') as f:
                json.dump(best_metrics, f, indent=2)

        print(f"\n  Best mIoU: {self.best_miou:.2f}% at epoch {self.best_epoch}\n")
        return self.history, best_metrics

    def load_best(self):
        path = os.path.join(self.exp_dir, "best_model.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded best model (epoch {ckpt['epoch']}, mIoU: {ckpt['best_miou']:.2f}%)")
        else:
            print(f"  No checkpoint at {path}")


# ============================================================
# TEST-TIME AUGMENTATION (TTA) — "Amplification During Testing"
# ============================================================

class TestTimeAugmentation:
    """
    Creates augmented copies of each test image, predicts on all,
    averages softmax probabilities for a smoother final prediction.
    Typically +1-3% mIoU improvement.
    """

    def __init__(self, model, device, use_flips=True, use_rotations=True):
        self.model = model
        self.device = device
        self.model.eval()

        self.augmentations = [("original", lambda x: x, lambda x: x)]

        if use_flips:
            self.augmentations += [
                ("hflip",  lambda x: torch.flip(x, [-1]),      lambda x: torch.flip(x, [-1])),
                ("vflip",  lambda x: torch.flip(x, [-2]),      lambda x: torch.flip(x, [-2])),
                ("hvflip", lambda x: torch.flip(x, [-2, -1]),  lambda x: torch.flip(x, [-2, -1])),
            ]
        if use_rotations:
            self.augmentations += [
                ("rot90",  lambda x: torch.rot90(x, 1, [-2,-1]), lambda x: torch.rot90(x, 3, [-2,-1])),
                ("rot180", lambda x: torch.rot90(x, 2, [-2,-1]), lambda x: torch.rot90(x, 2, [-2,-1])),
                ("rot270", lambda x: torch.rot90(x, 3, [-2,-1]), lambda x: torch.rot90(x, 1, [-2,-1])),
            ]

        print(f"  TTA: {len(self.augmentations)} augmentations")

    @torch.no_grad()
    def predict(self, image):
        image = image.to(self.device)
        sum_probs = None

        for _, aug_fn, rev_fn in self.augmentations:
            with torch.cuda.amp.autocast():
                logits = self.model(aug_fn(image))
            probs = rev_fn(F.softmax(logits, dim=1))
            sum_probs = probs if sum_probs is None else sum_probs + probs

        return torch.argmax(sum_probs / len(self.augmentations), dim=1)

    @torch.no_grad()
    def evaluate(self, val_loader, num_classes=6, ignore_index=255):
        metrics_no = SegmentationMetrics(num_classes, ignore_index)
        metrics_tta = SegmentationMetrics(num_classes, ignore_index)

        for images, masks in tqdm(val_loader, desc="  TTA Eval", leave=False):
            images = images.to(self.device)

            with torch.cuda.amp.autocast():
                logits = self.model(images)
            metrics_no.update(torch.argmax(logits, dim=1).cpu(), masks)
            metrics_tta.update(self.predict(images).cpu(), masks)

        r_no = metrics_no.compute()
        r_tta = metrics_tta.compute()

        print(f"\n  Without TTA: mIoU={r_no['mIoU']:.2f}%")
        print(f"  With TTA:    mIoU={r_tta['mIoU']:.2f}%")
        print(f"  Improvement: +{r_tta['mIoU'] - r_no['mIoU']:.2f}%")

        return r_no, r_tta


# ============================================================
# ENSEMBLE LEARNING — "Integrate Multiple Models"
# ============================================================

class EnsemblePredictor:
    """
    Average softmax probabilities from multiple models.
    Can combine with TTA for maximum performance.
    """

    def __init__(self, models, device, use_tta=False,
                 tta_flips=True, tta_rotations=True):
        self.models = models
        self.device = device

        for m in self.models.values():
            m.to(device).eval()

        self.augmentations = [("original", lambda x: x, lambda x: x)]
        if use_tta:
            if tta_flips:
                self.augmentations += [
                    ("hflip",  lambda x: torch.flip(x, [-1]),      lambda x: torch.flip(x, [-1])),
                    ("vflip",  lambda x: torch.flip(x, [-2]),      lambda x: torch.flip(x, [-2])),
                    ("hvflip", lambda x: torch.flip(x, [-2, -1]),  lambda x: torch.flip(x, [-2, -1])),
                ]
            if tta_rotations:
                self.augmentations += [
                    ("rot90",  lambda x: torch.rot90(x, 1, [-2,-1]), lambda x: torch.rot90(x, 3, [-2,-1])),
                    ("rot180", lambda x: torch.rot90(x, 2, [-2,-1]), lambda x: torch.rot90(x, 2, [-2,-1])),
                    ("rot270", lambda x: torch.rot90(x, 3, [-2,-1]), lambda x: torch.rot90(x, 1, [-2,-1])),
                ]

        n = len(self.models) * len(self.augmentations)
        mode = "Ensemble + TTA" if use_tta else "Ensemble"
        print(f"  {mode}: {len(self.models)} models × {len(self.augmentations)} augs = {n} predictions")

    @torch.no_grad()
    def predict(self, image):
        image = image.to(self.device)
        sum_probs = None
        count = 0

        for model in self.models.values():
            for _, aug_fn, rev_fn in self.augmentations:
                with torch.cuda.amp.autocast():
                    logits = model(aug_fn(image))
                probs = rev_fn(F.softmax(logits, dim=1))
                sum_probs = probs if sum_probs is None else sum_probs + probs
                count += 1

        return torch.argmax(sum_probs / count, dim=1)

    @torch.no_grad()
    def evaluate(self, val_loader, num_classes=6, ignore_index=255):
        model_metrics = {n: SegmentationMetrics(num_classes, ignore_index) for n in self.models}
        ens_metrics = SegmentationMetrics(num_classes, ignore_index)

        for images, masks in tqdm(val_loader, desc="  Ensemble Eval", leave=False):
            images = images.to(self.device)

            for name, model in self.models.items():
                with torch.cuda.amp.autocast():
                    pred = torch.argmax(model(images), dim=1).cpu()
                model_metrics[name].update(pred, masks)

            ens_metrics.update(self.predict(images).cpu(), masks)

        results = {"single_models": {}, "ensemble": None}

        print(f"\n  {'═'*55}")
        for name, m in model_metrics.items():
            r = m.compute()
            results["single_models"][name] = r
            print(f"  {name:30s} │ mIoU: {r['mIoU']:6.2f}%")

        r_ens = ens_metrics.compute()
        results["ensemble"] = r_ens
        print(f"  {'─'*55}")
        print(f"  {'>>> ENSEMBLE':30s} │ mIoU: {r_ens['mIoU']:6.2f}%")

        best_single = max(results["single_models"].values(), key=lambda x: x['mIoU'])
        improvement = r_ens['mIoU'] - best_single['mIoU']
        results["improvement_over_best_single"] = round(improvement, 2)
        print(f"  Improvement over best single: +{improvement:.2f}%")
        print(f"  {'═'*55}")

        return results