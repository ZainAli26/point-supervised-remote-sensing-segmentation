"""
metrics.py — Segmentation Evaluation Metrics
==============================================
Computes mIoU, OA, F1, Kappa from accumulated confusion matrix.
All metrics computed on FULL ground truth masks (not point labels).
"""

import numpy as np
import torch


class SegmentationMetrics:
    """
    Accumulates predictions across batches, then computes metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=6)
        for pred, target in validation:
            metrics.update(pred, target)
        results = metrics.compute()
    """

    CLASS_NAMES = ["bare_soil", "building", "pavement",
                   "road", "vegetation", "water"]

    def __init__(self, num_classes=6, ignore_index=255):
        self.nc = num_classes
        self.ign = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.cm = np.zeros((self.nc, self.nc), dtype=np.int64)

    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        p, t = pred.flatten(), target.flatten()
        valid = (t != self.ign) & (t >= 0) & (t < self.nc) & (p >= 0) & (p < self.nc)
        p, t = p[valid], t[valid]
        idx = t * self.nc + p
        counts = np.bincount(idx, minlength=self.nc * self.nc)
        self.cm += counts.reshape(self.nc, self.nc)

    def compute(self):
        cm = self.cm
        iou, f1 = [], []

        for c in range(self.nc):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            union = tp + fp + fn
            iou.append(tp / union if union > 0 else 0.0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

        valid_cls = [i for i in range(self.nc) if cm[i, :].sum() > 0]
        oa = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
        miou = np.mean([iou[i] for i in valid_cls]) if valid_cls else 0.0
        mf1 = np.mean([f1[i] for i in valid_cls]) if valid_cls else 0.0

        pe = sum(cm[c, :].sum() * cm[:, c].sum() for c in range(self.nc))
        pe = pe / (cm.sum() ** 2) if cm.sum() > 0 else 0.0
        kappa = (oa - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

        return {
            "mIoU": round(miou * 100, 2),
            "OA": round(oa * 100, 2),
            "macro_F1": round(mf1 * 100, 2),
            "kappa": round(kappa, 4),
            "per_class_iou": {
                self.CLASS_NAMES[c]: round(iou[c] * 100, 2) for c in range(self.nc)
            },
            "per_class_f1": {
                self.CLASS_NAMES[c]: round(f1[c] * 100, 2) for c in range(self.nc)
            },
            "confusion_matrix": cm.tolist(),
        }

    def summary(self):
        r = self.compute()
        print(f"\n{'='*55}")
        print(f"  mIoU: {r['mIoU']:.2f}%  |  OA: {r['OA']:.2f}%  |  "
              f"F1: {r['macro_F1']:.2f}%  |  κ: {r['kappa']:.4f}")
        print(f"{'─'*55}")
        for c in range(self.nc):
            name = self.CLASS_NAMES[c]
            print(f"  {name:15s} {r['per_class_iou'][name]:7.2f}% {r['per_class_f1'][name]:7.2f}%")
        print(f"{'='*55}")
        return r


if __name__ == "__main__":
    metrics = SegmentationMetrics(6)
    for _ in range(10):
        metrics.update(torch.randint(0, 6, (4, 64, 64)), torch.randint(0, 6, (4, 64, 64)))
    metrics.summary()
    print("  ✓ Verified")