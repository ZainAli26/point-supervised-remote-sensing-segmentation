"""
model.py — Segmentation Model Factory
=======================================
Supports multiple architectures for ensemble learning:
  DeepLabV3+ — ASPP multi-scale context
  U-Net      — skip connections for fine boundaries
  FPN        — feature pyramid for multi-scale objects
All use ImageNet pretrained encoders (transfer learning).
"""

import segmentation_models_pytorch as smp


ENSEMBLE_CONFIGS = {
    "deeplabv3plus_resnet34": {"architecture": "DeepLabV3Plus", "encoder": "resnet34"},
    "unet_resnet34":          {"architecture": "Unet",          "encoder": "resnet34"},
    "fpn_resnet34":           {"architecture": "FPN",           "encoder": "resnet34"},
    "deeplabv3plus_resnet50": {"architecture": "DeepLabV3Plus", "encoder": "resnet50"},
    "unet_resnet50":          {"architecture": "Unet",          "encoder": "resnet50"},
    "fpn_resnet50":           {"architecture": "FPN",           "encoder": "resnet50"},
}


def build_model(architecture="DeepLabV3Plus", encoder="resnet34",
                num_classes=6, pretrained=True):
    """Build a single segmentation model."""
    weights = "imagenet" if pretrained else None
    builders = {
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "Unet": smp.Unet,
        "FPN": smp.FPN,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = builders[architecture](
        encoder_name=encoder, encoder_weights=weights,
        in_channels=3, classes=num_classes,
    )
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {architecture} + {encoder} ({n:.1f}M params)")
    return model


def build_ensemble_models(num_classes=6, pretrained=True):
    """Build all models for ensemble."""
    models = {}
    for name, cfg in ENSEMBLE_CONFIGS.items():
        print(f"\n  Building ensemble member: {name}")
        models[name] = build_model(cfg['architecture'], cfg['encoder'],
                                   num_classes, pretrained)
    print(f"\n  Ensemble: {len(models)} models built")
    return models


if __name__ == "__main__":
    import torch
    model = build_model("DeepLabV3Plus", "resnet34", 6)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 6, 256, 256)
    print(f"  Input: {x.shape} → Output: {y.shape} ✓")