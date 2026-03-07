"""
histogram_matching.py — Picture Style Normalization
=====================================================
From slides: "Referring to the original ML pipeline, histogram matching
is used to ensure the stability of the picture style"

Normalizes color distributions across all images to a single reference,
removing lighting/sensor/seasonal variation.
Applied as preprocessing BEFORE any training.
"""

import os
import json
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


def match_histogram_channel(source, reference):
    """Match histogram of one channel via CDF matching."""
    src_hist, _ = np.histogram(source.flatten(), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(reference.flatten(), bins=256, range=(0, 256))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    mapping = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        mapping[v] = np.argmin(np.abs(ref_cdf - src_cdf[v]))

    return mapping[source]


def match_histogram(source_img, reference_img):
    """Match RGB histogram of source to reference (per-channel)."""
    matched = np.zeros_like(source_img)
    for ch in range(3):
        matched[:, :, ch] = match_histogram_channel(
            source_img[:, :, ch], reference_img[:, :, ch])
    return matched


def _glob_images(image_dir):
    """Find all images (jpg or png) in a directory."""
    paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    if not paths:
        paths = sorted(glob(os.path.join(image_dir, "*.png")))
    return paths


def select_reference_image(image_dir, method="median"):
    """Select reference image closest to median brightness/contrast."""
    paths = _glob_images(image_dir)
    assert len(paths) > 0, f"No images in {image_dir}"

    if method == "first":
        ref = np.array(Image.open(paths[0]).convert("RGB"))
        print(f"  Reference: {os.path.basename(paths[0])}")
        return ref, paths[0], {}

    print(f"  Analyzing {len(paths)} images for reference selection...")
    stats = []
    for p in tqdm(paths, desc="  Stats", leave=False):
        img = np.array(Image.open(p).convert("RGB"))
        gray = np.mean(img, axis=2)
        stats.append({"path": p, "bright": np.mean(gray), "contrast": np.std(gray)})

    med_b = np.median([s["bright"] for s in stats])
    med_c = np.median([s["contrast"] for s in stats])

    b_range = max(s["bright"] for s in stats) - min(s["bright"] for s in stats) + 1e-6
    c_range = max(s["contrast"] for s in stats) - min(s["contrast"] for s in stats) + 1e-6

    best = min(stats, key=lambda s:
               ((s["bright"] - med_b) / b_range) ** 2 +
               ((s["contrast"] - med_c) / c_range) ** 2)

    ref = np.array(Image.open(best["path"]).convert("RGB"))
    print(f"  Reference: {os.path.basename(best['path'])} "
          f"(brightness={best['bright']:.1f}, contrast={best['contrast']:.1f})")
    return ref, best["path"], {"num_images": len(paths)}


def apply_histogram_matching(image_dir, output_dir, reference_img=None,
                             reference_path=None):
    """Apply histogram matching to all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    paths = _glob_images(image_dir)

    if reference_img is None:
        reference_img, reference_path, _ = select_reference_image(image_dir)

    print(f"\n  Matching {len(paths)} images to reference...")
    bright_before, bright_after = [], []

    for p in tqdm(paths, desc="  Matching", leave=False):
        img = np.array(Image.open(p).convert("RGB"))
        bright_before.append(np.mean(img))
        matched = match_histogram(img, reference_img)
        bright_after.append(np.mean(matched))
        Image.fromarray(matched).save(os.path.join(output_dir, os.path.basename(p)))

    std_b = np.std(bright_before)
    std_a = np.std(bright_after)
    improvement = ((std_b - std_a) / std_b) * 100 if std_b > 0 else 0

    print(f"  Brightness std: {std_b:.2f} → {std_a:.2f} ({improvement:.1f}% improvement)")
    return {
        "num_images": len(paths),
        "brightness_std_before": round(float(std_b), 2),
        "brightness_std_after": round(float(std_a), 2),
        "consistency_improvement": round(float(improvement), 1),
        "output_dir": output_dir,
    }


def preprocess_dataset(cfg):
    """
    Full preprocessing pipeline:
    1. Select reference from image set
    2. Match all images to reference
    3. Update config path to matched directory
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING: Histogram Matching")
    print("=" * 60)

    image_dir = cfg['data']['image_dir']
    # Detect image extension (jpg for WHDLD, png for others)
    jpg_files = glob(os.path.join(image_dir, "*.jpg"))
    png_files = glob(os.path.join(image_dir, "*.png"))
    img_ext = "*.jpg" if len(jpg_files) >= len(png_files) else "*.png"

    out_dir = image_dir + "_matched"

    # Skip if already done
    existing = glob(os.path.join(out_dir, img_ext))
    if os.path.exists(out_dir) and len(existing) > 0:
        print("  Already processed. Skipping.")
        cfg['data']['image_dir'] = out_dir
        return cfg, {"skipped": True}

    ref_img, ref_path, _ = select_reference_image(image_dir)

    print("\n  Matching all images...")
    match_stats = apply_histogram_matching(image_dir, out_dir, ref_img, ref_path)

    cfg['data']['image_dir'] = out_dir

    stats = {"reference": os.path.basename(ref_path),
             "images": match_stats}

    stats_path = os.path.join(os.path.dirname(image_dir), "matching_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Done. Updated config path to matched images.")
    return cfg, stats


def visualize_histogram_matching(original_dir, matched_dir, reference_path,
                                 n_samples=3, save=True):
    """Before/after comparison visualization."""
    import matplotlib.pyplot as plt

    ref_img = np.array(Image.open(reference_path).convert("RGB"))
    originals = _glob_images(original_dir)
    matched = _glob_images(matched_dir)
    indices = np.random.choice(len(originals), min(n_samples, len(originals)), replace=False)

    fig, axes = plt.subplots(n_samples + 1, 4, figsize=(16, 4 * (n_samples + 1)))

    axes[0, 0].imshow(ref_img)
    axes[0, 0].set_title("REFERENCE", fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].set_visible(False)
    for ch, c in [(0,'red'),(1,'green'),(2,'blue')]:
        axes[0, 2].hist(ref_img[:,:,ch].flatten(), 64, color=c, alpha=0.5, density=True)
    axes[0, 2].set_title("Reference Histogram")
    axes[0, 3].set_visible(False)

    for row, idx in enumerate(indices, 1):
        orig = np.array(Image.open(originals[idx]).convert("RGB"))
        match = np.array(Image.open(matched[idx]).convert("RGB"))

        axes[row, 0].imshow(orig); axes[row, 0].set_title("Original"); axes[row, 0].axis('off')
        axes[row, 1].imshow(match); axes[row, 1].set_title("Matched"); axes[row, 1].axis('off')

        for ch, c in [(0,'red'),(1,'green'),(2,'blue')]:
            axes[row, 2].hist(orig[:,:,ch].flatten(), 64, color=c, alpha=0.5, density=True)
            axes[row, 3].hist(match[:,:,ch].flatten(), 64, color=c, alpha=0.5, density=True)
        axes[row, 2].set_title("Original Hist")
        axes[row, 3].set_title("Matched Hist")

    plt.suptitle("Histogram Matching: Picture Style Stability", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/histogram_matching.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: histogram_matching.png")
    plt.close()


if __name__ == "__main__":
    print("Verifying histogram matching...")
    np.random.seed(42)
    src = np.random.normal(80, 20, (64, 64, 3)).clip(0, 255).astype(np.uint8)
    ref = np.random.normal(180, 30, (64, 64, 3)).clip(0, 255).astype(np.uint8)
    matched = match_histogram(src, ref)

    d_before = abs(np.mean(src) - np.mean(ref))
    d_after = abs(np.mean(matched) - np.mean(ref))
    print(f"  Distance to ref: {d_before:.1f} → {d_after:.1f} ✓")