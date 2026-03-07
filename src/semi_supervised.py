"""
semi_supervised.py — Pseudo-Label Self-Training
=================================================
From slides: "Semi-supervised technology is used to make full use
of unmarked pixels to further improve model performance"

Strategy:
  Stage 1: Train on point labels (~0.1% pixels labeled)
  Stage 2: Generate pseudo labels from confident predictions → retrain
  Stage 3: Repeat with relaxed threshold
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from tqdm import tqdm


class PseudoLabelGenerator:
    """
    Generate pseudo labels from model predictions.
    
    Supports:
        1. Single model + TTA (default)
        2. Ensemble of models + TTA (strongest — use when ensemble is available)
    
    Why ensemble helps here:
        - DeepLabV3+ might be confident but WRONG on a boundary pixel
        - U-Net might be uncertain there (correct behavior)
        - Averaging them → lower confidence → pixel stays unlabeled (good!)
        - In homogeneous regions all 3 agree → very high confidence → pseudo label
        
    This means ensemble pseudo labels have:
        - Fewer false positives (wrong labels that poison training)
        - More true positives in easy regions
        - Better calibrated confidence scores overall
    """

    def __init__(self, model_or_models, device, confidence_threshold=0.9,
                 ignore_index=255):
        """
        Args:
            model_or_models: single nn.Module OR dict of {name: model} for ensemble
            device: cuda/cpu
            confidence_threshold: min probability to accept pseudo label
        """
        self.device = device
        self.threshold = confidence_threshold
        self.ignore_index = ignore_index

        # Handle both single model and ensemble
        if isinstance(model_or_models, dict):
            self.models = model_or_models
            self.is_ensemble = True
            for m in self.models.values():
                m.to(device).eval()
            print(f"  PseudoLabelGenerator: ENSEMBLE mode ({len(self.models)} models + TTA)")
        else:
            self.models = {"single": model_or_models}
            self.is_ensemble = False
            model_or_models.to(device).eval()
            print(f"  PseudoLabelGenerator: single model + TTA")

    @torch.no_grad()
    def generate_for_image(self, image):
        """
        Generate pseudo labels using all models + TTA (amplification).

        For ensemble mode (3 models × 4 augmentations = 12 predictions):
            Much more reliable confidence than single model.
            
        For single mode (1 model × 4 augmentations = 4 predictions):
            Still better than raw single prediction.
        """
        image = image.unsqueeze(0).to(self.device)

        # TTA augmentations: original + 3 flips
        augmentations = [
            (lambda x: x,                         lambda x: x),
            (lambda x: torch.flip(x, [-1]),        lambda x: torch.flip(x, [-1])),
            (lambda x: torch.flip(x, [-2]),        lambda x: torch.flip(x, [-2])),
            (lambda x: torch.flip(x, [-2, -1]),    lambda x: torch.flip(x, [-2, -1])),
        ]

        sum_probs = None
        count = 0

        for model in self.models.values():
            for aug_fn, rev_fn in augmentations:
                aug_image = aug_fn(image)
                with torch.cuda.amp.autocast():
                    logits = model(aug_image)
                probs = rev_fn(F.softmax(logits, dim=1))
                sum_probs = probs if sum_probs is None else sum_probs + probs
                count += 1

        # Average across all models × augmentations
        avg_probs = sum_probs / count                          # (1, C, H, W)
        max_probs, pred_classes = avg_probs.max(dim=1)         # (1, H, W)

        max_probs = max_probs.squeeze(0).cpu().numpy()
        pred_classes = pred_classes.squeeze(0).cpu().numpy()

        pseudo_mask = np.full_like(pred_classes, self.ignore_index, dtype=np.uint8)
        confident = max_probs >= self.threshold
        pseudo_mask[confident] = pred_classes[confident]
        return pseudo_mask, max_probs

    @torch.no_grad()
    def generate_for_dataset(self, image_dir, mask_dir,
                             num_classes=6, num_points_per_class=10,
                             pseudo_label_tag=None, split_file=None):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from dataset import simulate_point_labels

        # Normalize only — no cropping, so pseudo labels match full image size
        normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Use split file to only process train images (not val)
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                names = [line.strip() for line in f if line.strip()]
            images = [os.path.join(image_dir, n + ".jpg") for n in names]
            masks = [os.path.join(mask_dir, n + ".png") for n in names]
        else:
            images = sorted(glob(os.path.join(image_dir, "*.jpg")))
            if not images:
                images = sorted(glob(os.path.join(image_dir, "*.png")))
            masks = sorted(glob(os.path.join(mask_dir, "*.png")))
        assert len(images) == len(masks), \
            f"Image/mask count mismatch: {len(images)} vs {len(masks)}"

        dirname = f"pseudo_labels_{pseudo_label_tag}" if pseudo_label_tag else "pseudo_labels"
        pseudo_dir = os.path.join(os.path.dirname(mask_dir), dirname)
        os.makedirs(pseudo_dir, exist_ok=True)

        total_px = pseudo_px = point_px = ignore_px = 0

        print(f"\n  Generating pseudo labels (threshold={self.threshold})...")
        for img_path, mask_path in tqdm(zip(images, masks), total=len(images),
                                        desc="  PseudoGen", leave=False):
            image = np.array(Image.open(img_path).convert("RGB"))
            full_mask = np.array(Image.open(mask_path))

            # Remap WHDLD labels: 1-6 → 0-5; anything else → ignore
            full_mask = full_mask.astype(np.int16)
            valid = (full_mask >= 1) & (full_mask <= num_classes)
            remapped = np.full(full_mask.shape, self.ignore_index, dtype=np.uint8)
            remapped[valid] = (full_mask[valid] - 1).astype(np.uint8)
            full_mask = remapped

            aug = normalize(image=image)
            image_t = aug['image']

            pseudo_mask, _ = self.generate_for_image(image_t)
            point_mask = simulate_point_labels(full_mask, num_points_per_class, num_classes)

            # Merge: original points always override pseudo labels
            merged = pseudo_mask.copy()
            has_point = point_mask != self.ignore_index
            merged[has_point] = point_mask[has_point]

            h, w = merged.shape
            total_px += h * w
            point_px += has_point.sum()
            pseudo_px += ((merged != self.ignore_index) & (~has_point)).sum()
            ignore_px += (merged == self.ignore_index).sum()

            Image.fromarray(merged).save(os.path.join(pseudo_dir, os.path.basename(mask_path)))

        stats = {
            "total_pixels": int(total_px),
            "point_pixels": int(point_px),
            "pseudo_pixels": int(pseudo_px),
            "point_ratio": round(100 * point_px / total_px, 4),
            "pseudo_ratio": round(100 * pseudo_px / total_px, 2),
            "total_labeled_ratio": round(100 * (point_px + pseudo_px) / total_px, 2),
            "ignore_ratio": round(100 * ignore_px / total_px, 2),
            "confidence_threshold": self.threshold,
            "pseudo_label_dir": pseudo_dir,
        }

        print(f"\n  Point labels:  {stats['point_ratio']:.4f}%")
        print(f"  Pseudo labels: {stats['pseudo_ratio']:.2f}%")
        print(f"  Total labeled: {stats['total_labeled_ratio']:.2f}%")
        return stats


class SemiSupervisedDataset(Dataset):
    """Loads pre-generated pseudo label masks (points + pseudo labels merged)."""

    def __init__(self, image_dir, pseudo_mask_dir, num_classes=6,
                 ignore_index=255, transform=None, split_file=None):
        # Use split file to match only training images
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                names = [line.strip() for line in f if line.strip()]
            self.images = [os.path.join(image_dir, n + ".jpg") for n in names]
            self.masks = [os.path.join(pseudo_mask_dir, n + ".png") for n in names]
        else:
            self.images = sorted(glob(os.path.join(image_dir, "*.jpg")))
            if not self.images:
                self.images = sorted(glob(os.path.join(image_dir, "*.png")))
            self.masks = sorted(glob(os.path.join(pseudo_mask_dir, "*.png")))
        assert len(self.images) == len(self.masks), \
            f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}"
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.transform = transform

        sample = np.array(Image.open(self.masks[0]))
        ratio = 100 * (sample != ignore_index).sum() / sample.size
        print(f"  SemiSupervisedDataset: {len(self.images)} images, ~{ratio:.1f}% labeled")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))
        mask[mask >= self.num_classes] = self.ignore_index

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)
        return image, label.long()


class SelfTrainingPipeline:
    """
    Iterative self-training: train → pseudo label → retrain.

    Supports two modes:
      1. Single-model mode (ensemble_configs=None): trains one model per stage.
      2. Ensemble mode (ensemble_configs provided): trains an ensemble of
         architectures at every stage and uses the ensemble for pseudo label
         generation. Returns the final trained ensemble.

    Args:
        cfg: config dict
        num_points, loss_name, seed: experiment settings
        n_iterations: self-training rounds (1-3)
        thresholds: confidence per iteration (start strict, relax)
        ensemble_configs: list of dicts with keys {architecture, encoder, name}
                          for ensemble training at every stage. If None, falls
                          back to single-model mode.
        ensemble_models: dict of {name: trained_model} — pre-trained ensemble
                         used only in single-model mode for pseudo labeling.
                         Ignored when ensemble_configs is set.
    """

    def __init__(self, cfg, num_points=10, loss_name="pce", seed=42,
                 n_iterations=2, thresholds=None, ensemble_configs=None,
                 ensemble_models=None, architecture=None, encoder=None,
                 exp_prefix="semi_sup"):
        self.cfg = cfg
        self.num_points = num_points
        self.loss_name = loss_name
        self.seed = seed
        self.n_iterations = n_iterations
        self.thresholds = thresholds or [0.95, 0.90, 0.85][:n_iterations]
        self.ensemble_configs = ensemble_configs
        self.ensemble_models = ensemble_models
        self.architecture = architecture or cfg['model']['architecture']
        self.encoder = encoder or cfg['model']['encoder']
        self.exp_prefix = exp_prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_cls = cfg['data']['num_classes']
        self.patch = cfg['data']['patch_size']

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    def _train_single(self, cfg, build_model, build_loss, Trainer,
                      train_loader, val_loader, stage_name):
        """Train a single model and return (trained_model, best_metrics)."""
        tc = cfg['training']
        model = build_model(self.architecture, self.encoder, self.n_cls, True)
        criterion = build_loss(self.loss_name, self.n_cls)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=tc['learning_rate'],
                                      weight_decay=tc['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc['epochs'])

        trainer = Trainer(model, criterion, optimizer, scheduler,
                          self.device, cfg, stage_name)
        _, metrics = trainer.fit(train_loader, val_loader)
        trainer.load_best()
        return trainer.model, metrics

    def _train_ensemble(self, cfg, build_model, build_loss, Trainer,
                        train_loader, val_loader, stage_name):
        """Train all ensemble members and return (dict_of_models, best_metrics).

        best_metrics is taken from the member with highest val mIoU so the
        caller has a single representative metric for this stage.
        """
        from train import EnsemblePredictor

        tc = cfg['training']
        trained = {}
        member_metrics = {}

        for mcfg in self.ensemble_configs:
            name = mcfg['name']
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

            print(f"\n    Training ensemble member: {name}")
            model = build_model(mcfg['architecture'], mcfg['encoder'],
                                self.n_cls, True)
            criterion = build_loss(self.loss_name, self.n_cls)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=tc['learning_rate'],
                                          weight_decay=tc['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tc['epochs'])

            trainer = Trainer(model, criterion, optimizer, scheduler,
                              self.device, cfg,
                              f"{stage_name}_ens_{name}")
            _, m = trainer.fit(train_loader, val_loader)
            trainer.load_best()
            trained[name] = trainer.model
            member_metrics[name] = m

        # Evaluate the ensemble as a whole
        ens = EnsemblePredictor(trained, self.device, use_tta=False)
        ens_results = ens.evaluate(val_loader, self.n_cls)
        ens_metrics = ens_results['ensemble']

        print(f"\n    Ensemble mIoU: {ens_metrics['mIoU']:.2f}%  "
              f"(best single: "
              f"{max(m['mIoU'] for m in member_metrics.values()):.2f}%)")

        return trained, ens_metrics

    # ----------------------------------------------------------
    # main loop
    # ----------------------------------------------------------

    def run(self):
        from dataset import WHDLDPointDataset, get_train_transform, get_val_transform
        from losses import build_loss
        from model import build_model
        from train import Trainer

        use_ensemble = self.ensemble_configs is not None
        mode_str = (f"ENSEMBLE ({len(self.ensemble_configs)} archs)"
                    if use_ensemble
                    else f"SINGLE MODEL ({self.architecture}+{self.encoder})")

        all_metrics = {}
        current_model = None          # single model  (single mode)
        current_ensemble = None       # dict of models (ensemble mode)

        print(f"\n{'='*60}")
        print(f"  SEMI-SUPERVISED SELF-TRAINING — {mode_str}")
        print(f"  Iterations: {self.n_iterations}, Thresholds: {self.thresholds}")
        print(f"{'='*60}")

        cfg = self.cfg
        bs = cfg['training']['batch_size']
        nw = cfg['training']['num_workers']

        # ── Stage 1: Supervised on point labels ──
        print(f"\n  STAGE 1: Point labels only")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        train_ds = WHDLDPointDataset(
            cfg['data']['image_dir'], cfg['data']['mask_dir'],
            self.num_points, self.n_cls,
            transform=get_train_transform(self.patch), use_full_mask=False,
            split_file=cfg['data']['train_split'])
        val_ds = WHDLDPointDataset(
            cfg['data']['image_dir'], cfg['data']['mask_dir'],
            0, self.n_cls,
            transform=get_val_transform(self.patch), use_full_mask=True,
            split_file=cfg['data']['val_split'])

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=True)

        if use_ensemble:
            current_ensemble, stage1_metrics = self._train_ensemble(
                cfg, build_model, build_loss, Trainer,
                train_loader, val_loader, f"{self.exp_prefix}_stage1")
        else:
            current_model, stage1_metrics = self._train_single(
                cfg, build_model, build_loss, Trainer,
                train_loader, val_loader, f"{self.exp_prefix}_stage1")

        all_metrics["stage1_points_only"] = stage1_metrics
        print(f"\n  Stage 1 mIoU: {stage1_metrics['mIoU']:.2f}%")

        # ── Stage 2+: Self-training iterations ──
        for it in range(self.n_iterations):
            threshold = self.thresholds[it]
            stage = it + 2

            print(f"\n  STAGE {stage}: Self-training (threshold={threshold})")

            # Build pseudo label generator from the best available source
            if use_ensemble:
                pseudo_gen = PseudoLabelGenerator(
                    current_ensemble, self.device, threshold)
            elif self.ensemble_models is not None:
                pseudo_gen = PseudoLabelGenerator(
                    self.ensemble_models, self.device, threshold)
            else:
                pseudo_gen = PseudoLabelGenerator(
                    current_model, self.device, threshold)

            pseudo_tag = f"{self.exp_prefix}_stage{stage}"
            stats = pseudo_gen.generate_for_dataset(
                cfg['data']['image_dir'], cfg['data']['mask_dir'],
                self.n_cls, self.num_points,
                pseudo_label_tag=pseudo_tag,
                split_file=cfg['data'].get('train_split'))

            semi_ds = SemiSupervisedDataset(
                cfg['data']['image_dir'], stats['pseudo_label_dir'],
                self.n_cls, transform=get_train_transform(self.patch),
                split_file=cfg['data'].get('train_split'))
            semi_loader = DataLoader(semi_ds, batch_size=bs, shuffle=True,
                                     num_workers=nw, pin_memory=True, drop_last=True)

            if use_ensemble:
                current_ensemble, stage_metrics = self._train_ensemble(
                    cfg, build_model, build_loss, Trainer,
                    semi_loader, val_loader, f"{self.exp_prefix}_stage{stage}")
            else:
                current_model, stage_metrics = self._train_single(
                    cfg, build_model, build_loss, Trainer,
                    semi_loader, val_loader, f"{self.exp_prefix}_stage{stage}")

            all_metrics[f"stage{stage}_pseudo_t{threshold}"] = stage_metrics
            all_metrics[f"stage{stage}_pseudo_stats"] = stats

            delta = stage_metrics['mIoU'] - stage1_metrics['mIoU']
            print(f"\n  Stage {stage} mIoU: {stage_metrics['mIoU']:.2f}% "
                  f"(Δ={'+' if delta>=0 else ''}{delta:.2f}%)")

        # Save
        save_path = f"experiments/{self.exp_prefix}_results.json"
        serializable = {k: {kk: vv for kk, vv in v.items() if kk != 'confusion_matrix'}
                        for k, v in all_metrics.items() if isinstance(v, dict)}
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\n  Results saved to: {save_path}")

        if use_ensemble:
            return all_metrics, current_ensemble
        return all_metrics, current_model