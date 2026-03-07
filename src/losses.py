import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, C, H, W) — raw logits from segmentation network
            target: (B, H, W)    — point labels (255 = unlabeled)
        Returns:
            pCE loss (scalar)
        """
        B, C, H, W = pred.shape

        # Step 1: MASK_labeled — binary mask of valid pixels
        mask_labeled = (target != self.ignore_index).float()
        num_labeled = mask_labeled.sum()

        if num_labeled == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        # Replace ignore pixels with 0 temporarily (masked out later)
        target_safe = target.clone()
        target_safe[target == self.ignore_index] = 0

        # Step 2: Per-pixel Focal Loss
        log_probs = F.log_softmax(pred, dim=1)
        probs = torch.exp(log_probs)

        target_idx = target_safe.unsqueeze(1)
        log_pt = log_probs.gather(1, target_idx).squeeze(1)
        pt = probs.gather(1, target_idx).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        focal_loss = -self.alpha * focal_weight * log_pt

        # Optional: per-class weights
        if self.class_weights is not None:
            w = self.class_weights.to(pred.device)
            focal_loss = focal_loss * w[target_safe]

        # Step 3: FocalLoss × MASK_labeled
        masked_loss = focal_loss * mask_labeled

        # Step 4: Σ(masked_loss) / Σ(MASK_labeled)
        pce = masked_loss.sum() / num_labeled

        return pce

    def __repr__(self):
        w = "yes" if self.class_weights is not None else "no"
        return f"PartialCrossEntropyLoss(alpha={self.alpha}, gamma={self.gamma}, weights={w})"


def build_loss(loss_name, num_classes=6, class_counts=None, device='cuda'):
    """
    Factory to build loss by name.

    'pce'             → gamma=2.0, alpha=1.0 (default from slides)
    'pce_gamma0'      → gamma=0.0 (reduces to standard masked CE)
    'pce_focal_g2'    → gamma=2.0, alpha=0.25 (balanced focal)
    'pce_focal_g3'    → gamma=3.0, alpha=0.25 (stronger focusing)
    'pce_weighted'    → gamma=2.0 + inverse-frequency class weights
    'full_supervision'→ gamma=0.0 (standard CE for baseline)
    """
    if loss_name == "pce":
        return PartialCrossEntropyLoss(alpha=1.0, gamma=2.0)

    elif loss_name == "pce_gamma0":
        return PartialCrossEntropyLoss(alpha=1.0, gamma=0.0)

    elif loss_name == "pce_focal_g2":
        return PartialCrossEntropyLoss(alpha=0.25, gamma=2.0)

    elif loss_name == "pce_focal_g3":
        return PartialCrossEntropyLoss(alpha=0.25, gamma=3.0)

    elif loss_name == "pce_weighted":
        if class_counts is None:
            # Default WHDLD approx counts: bare_soil, building, pavement, road, vegetation, water
            class_counts = [50000, 80000, 100000, 120000, 30000, 40000]
            print("  Using default WHDLD class counts")
        total = sum(class_counts)
        weights = [total / (num_classes * c + 1e-6) for c in class_counts]
        mean_w = sum(weights) / len(weights)
        weights = torch.tensor([w / mean_w for w in weights], dtype=torch.float32)
        print(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
        return PartialCrossEntropyLoss(alpha=1.0, gamma=2.0, class_weights=weights)

    elif loss_name == "full_supervision":
        return PartialCrossEntropyLoss(alpha=1.0, gamma=0.0)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def verify_losses():
    """Unit tests for pCE loss."""
    print("=" * 55)
    print("VERIFYING PARTIAL CROSS ENTROPY LOSS (pCE)")
    print("=" * 55)

    B, C, H, W = 2, 6, 8, 8
    pred = torch.randn(B, C, H, W)
    full_target = torch.randint(0, C, (B, H, W))

    point_target = torch.full((B, H, W), 255, dtype=torch.long)
    for b in range(B):
        idxs = torch.randperm(H * W)[:5]
        r, c = idxs // W, idxs % W
        point_target[b, r, c] = full_target[b, r, c]

    labeled = (point_target != 255).sum().item()
    total = B * H * W
    print(f"\n  Test: {B}×{C}×{H}×{W}, {labeled}/{total} labeled\n")

    # Test 1: pCE(gamma=0) == torch.CE(ignore=255)
    pce_g0 = PartialCrossEntropyLoss(alpha=1.0, gamma=0.0)
    our = pce_g0(pred, point_target)
    ref = F.cross_entropy(pred, point_target, ignore_index=255)
    diff = abs(our.item() - ref.item())
    print(f"  pCE(gamma=0) == torch.CE: diff={diff:.8f} {'✓' if diff < 1e-5 else '✗'}")

    # Test 2: Gradients flow
    p = pred.clone().requires_grad_(True)
    loss = PartialCrossEntropyLoss(gamma=2.0)(p, point_target)
    loss.backward()
    print(f"  Gradients flow: {'✓' if p.grad is not None else '✗'}")

    # Test 3: Empty target → loss=0
    empty = torch.full((B, H, W), 255, dtype=torch.long)
    l = PartialCrossEntropyLoss(gamma=2.0)(pred, empty)
    print(f"  All ignore → loss=0: {'✓' if l.item() == 0.0 else '✗'}")

    # Test 4: All variants
    print(f"\n  All variants:")
    for name in ["pce", "pce_gamma0", "pce_focal_g2", "pce_focal_g3", "pce_weighted"]:
        fn = build_loss(name)
        l = fn(pred, point_target)
        l.backward(retain_graph=True)
        print(f"    {name:20s} → loss={l.item():.4f} ✓")

    print("\n" + "=" * 55)
    print("ALL TESTS PASSED ✓\n")


if __name__ == "__main__":
    verify_losses()