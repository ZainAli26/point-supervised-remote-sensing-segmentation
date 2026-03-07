"""
visualize.py — All figures for the technical report
=====================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image

# WHDLD classes (remapped 0-5): bare_soil, building, pavement, road, vegetation, water
COLORS = {
    0: (255,0,0), 1: (255,255,0), 2: (192,192,0),
    3: (0,255,0), 4: (128,128,128), 5: (0,0,255),
}
CLASS_NAMES = ["bare_soil", "building", "pavement",
               "road", "vegetation", "water"]
NUM_CLASSES = 6
SAVE_DIR = "visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)


def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb


# ── Experiment 1: Point Density ──

def plot_point_density_results(results, save=True):
    """results: {pts: [mIoU_seed1, ...]}"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    points = sorted(results.keys())
    means = [np.mean(results[p]) for p in points]
    stds = [np.std(results[p]) for p in points]

    ax = axes[0]
    ax.errorbar(points, means, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#2196F3')
    ax.fill_between(points, [m-s for m,s in zip(means,stds)],
                    [m+s for m,s in zip(means,stds)], alpha=0.2, color='#2196F3')
    ax.set_xlabel("Points per Class", fontsize=12)
    ax.set_ylabel("mIoU (%)", fontsize=12)
    ax.set_title("Effect of Point Label Density", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log'); ax.set_xticks(points); ax.set_xticklabels(points)
    for x, m in zip(points, means):
        ax.annotate(f"{m:.1f}%", (x, m), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

    ax = axes[1]
    gains = [0] + [means[i]-means[i-1] for i in range(1, len(means))]
    colors = ['#4CAF50' if g > 0 else '#F44336' for g in gains]
    ax.bar(range(len(points)), gains, color=colors, alpha=0.8)
    ax.set_xticks(range(len(points))); ax.set_xticklabels(points)
    ax.set_xlabel("Points per Class"); ax.set_ylabel("Marginal mIoU Gain (%)")
    ax.set_title("Diminishing Returns"); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "exp1_point_density.png"), dpi=150)
    plt.close()


# ── Experiment 2: Loss Comparison ──

def plot_loss_comparison(results, save=True):
    """results: {loss_name: {"mIoU": [...], "OA": [...], "F1": [...]}}"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = list(results.keys())
    display = {"pce": "pCE (γ=2)", "pce_focal_g2": "pCE Focal (γ=2)",
               "pce_focal_g3": "pCE Focal (γ=3)", "pce_weighted": "pCE Weighted"}
    colors = ['#2196F3', '#FF9800', '#F44336', '#4CAF50']

    for ax, key, title in zip(axes, ['mIoU','OA','F1'],
                               ['mIoU (%)','OA (%)','F1 (%)']):
        means = [np.mean(results[n][key]) for n in names]
        stds = [np.std(results[n][key]) for n in names]
        labels = [display.get(n, n) for n in names]
        bars = ax.bar(labels, means, yerr=stds, capsize=5,
                      color=colors[:len(names)], alpha=0.85, edgecolor='black', linewidth=0.5)
        for b, m in zip(bars, means):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                    f"{m:.1f}", ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel(title); ax.set_title(title); ax.grid(True, alpha=0.3, axis='y')
        if means:
            ax.set_ylim(max(0, min(means)-10), max(means)+8)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "exp2_loss_comparison.png"), dpi=150)
    plt.close()


def plot_per_class_iou_heatmap(results, save=True):
    """results: {loss_name: metrics_dict_with_per_class_iou}"""
    names = list(results.keys())
    display = {"pce": "pCE (γ=2)", "pce_focal_g2": "pCE Focal (γ=2)",
               "pce_focal_g3": "pCE Focal (γ=3)", "pce_weighted": "pCE Weighted"}
    matrix = np.array([[results[n]['per_class_iou'][c] for c in CLASS_NAMES] for n in names])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(len(names))); ax.set_yticklabels([display.get(n,n) for n in names])
    for i in range(len(names)):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, label='IoU (%)')
    ax.set_title("Per-Class IoU by Loss Function")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "exp2_per_class_iou.png"), dpi=150)
    plt.close()


# ── Training Curves ──

def plot_training_curves(histories, labels, save_name="training_curves", save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.tab10
    for i, (h, label) in enumerate(zip(histories, labels)):
        axes[0].plot(h['train_loss'], label=label, color=cmap(i), linewidth=1.5)
        axes[1].plot(h['val_miou'], label=label, color=cmap(i), linewidth=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU (%)"); axes[1].set_title("Validation mIoU")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"), dpi=150)
    plt.close()


# ── Qualitative Segmentation Maps ──

def plot_segmentation_samples(model, dataset, device, n_samples=4,
                              save_name="qualitative", save=True):
    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1: axes = axes.reshape(1, -1)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for row, idx in enumerate(indices):
        img_t, mask = dataset[idx]
        mask_np = mask.numpy()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = torch.argmax(model(img_t.unsqueeze(0).to(device)), dim=1).squeeze().cpu().numpy()

        img_d = img_t.permute(1,2,0).numpy() * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
        img_d = np.clip(img_d, 0, 1)

        axes[row,0].imshow(img_d); axes[row,0].set_title("Input" if row==0 else ""); axes[row,0].axis('off')
        axes[row,1].imshow(mask_to_rgb(mask_np)); axes[row,1].set_title("Ground Truth" if row==0 else ""); axes[row,1].axis('off')
        axes[row,2].imshow(mask_to_rgb(pred)); axes[row,2].set_title("Prediction" if row==0 else ""); axes[row,2].axis('off')

    patches = [mpatches.Patch(color=np.array(COLORS[i])/255, label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if save:
        plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"), dpi=150)
    plt.close()


# ── Point Label Demo ──

def plot_point_labels_demo(full_mask, points_list=[1, 5, 10, 50], save=True):
    from dataset import simulate_point_labels
    n = len(points_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    axes[0].imshow(mask_to_rgb(full_mask)); axes[0].set_title("Full Mask"); axes[0].axis('off')

    for i, pts in enumerate(points_list):
        pm = simulate_point_labels(full_mask, pts, NUM_CLASSES)
        vis = np.full((*full_mask.shape, 3), 30, dtype=np.uint8)
        for cls in range(NUM_CLASSES): vis[pm == cls] = COLORS[cls]
        n_lab = (pm != 255).sum()
        axes[i+1].imshow(vis); axes[i+1].set_title(f"{pts} pts/class\n({n_lab} px)"); axes[i+1].axis('off')

    plt.suptitle("Point Label Simulation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "point_label_demo.png"), dpi=150)
    plt.close()


# ── TTA Comparison ──

def plot_tta_comparison(no_tta, with_tta, labels, save=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x-w/2, no_tta, w, label='Without TTA', color='#90CAF9', edgecolor='black', linewidth=0.5)
    ax.bar(x+w/2, with_tta, w, label='With TTA', color='#2196F3', edgecolor='black', linewidth=0.5)
    for i in range(len(labels)):
        ax.annotate(f"+{with_tta[i]-no_tta[i]:.1f}%", xy=(x[i]+w/2, with_tta[i]),
                    xytext=(0,8), textcoords="offset points", ha='center', fontsize=9, color='green')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel("mIoU (%)"); ax.set_title("TTA Improvement"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "tta_comparison.png"), dpi=150)
    plt.close()


# ── Ensemble Results ──

def plot_ensemble_results(results, save=True):
    """results: dict with 'single_models', 'ensemble_no_tta', 'ensemble_with_tta'"""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, vals, colors = [], [], []
    display = {"deeplabv3plus_resnet34": "DeepLabV3+", "unet_resnet34": "U-Net",
               "fpn_resnet34": "FPN", "deeplabv3plus_resnet50": "DeepLabV3+",
               "unet_resnet50": "U-Net", "fpn_resnet50": "FPN"}
    for n, m in results.get('single_models', {}).items():
        labels.append(display.get(n, n)); vals.append(m['mIoU']); colors.append('#90CAF9')
    if 'ensemble_no_tta' in results:
        labels.append("Ensemble"); vals.append(results['ensemble_no_tta']['mIoU']); colors.append('#2196F3')
    if 'ensemble_with_tta' in results:
        labels.append("Ensemble+TTA"); vals.append(results['ensemble_with_tta']['mIoU']); colors.append('#0D47A1')

    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}%",
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel("mIoU (%)"); ax.set_title("Ensemble Learning Results"); ax.grid(True, alpha=0.3, axis='y')
    if vals: ax.set_ylim(max(0, min(vals)-5), max(vals)+5)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "ensemble_results.png"), dpi=150)
    plt.close()


# ── Semi-Supervised Results ──

def plot_semi_supervised_results(all_metrics, save=True):
    stages, miou_vals = [], []
    for key, val in all_metrics.items():
        if isinstance(val, dict) and 'mIoU' in val:
            stages.append(key.replace('_', '\n')); miou_vals.append(val['mIoU'])

    if not stages: return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#90CAF9'] + ['#2196F3'] * (len(stages)-1)
    bars = ax.bar(range(len(stages)), miou_vals, color=colors, edgecolor='black', linewidth=0.5)
    for i, (b, v) in enumerate(zip(bars, miou_vals)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}%",
                ha='center', fontsize=11, fontweight='bold')
        if i > 0:
            d = v - miou_vals[0]; c = 'green' if d > 0 else 'red'
            ax.text(b.get_x()+b.get_width()/2, b.get_height()-2,
                    f"({'+'if d>0 else ''}{d:.1f}%)", ha='center', fontsize=9, color=c)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel("mIoU (%)"); ax.set_title("Semi-Supervised Self-Training Progression")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "semi_supervised_results.png"), dpi=150)
    plt.close()


# ── Results Table ──

def generate_results_table(all_results, save=True):
    lines = ["=" * 70,
             f"{'Experiment':25s} {'mIoU':>8s} {'OA':>8s} {'F1':>8s} {'Kappa':>8s}",
             "-" * 70]
    for name, r in all_results.items():
        lines.append(f"{name:25s} {r.get('mIoU',0):7.2f}% {r.get('OA',0):7.2f}% "
                     f"{r.get('macro_F1',0):7.2f}% {r.get('kappa',0):7.4f}")
    lines.append("=" * 70)
    table = "\n".join(lines)
    print(table)
    if save:
        with open(os.path.join(SAVE_DIR, "results_table.txt"), 'w') as f:
            f.write(table)
    return table