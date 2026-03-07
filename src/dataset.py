import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2


def simulate_point_labels(full_mask, num_points_per_class=10,
                          num_classes=6, ignore_index=255):
    """
    Convert full segmentation mask to sparse point labels.

    For each class, randomly sample N pixels and keep their labels.
    All other pixels are set to ignore_index (255).
    """
    point_mask = np.full_like(full_mask, fill_value=ignore_index, dtype=np.uint8)

    for cls in range(num_classes):
        cls_pixels = np.argwhere(full_mask == cls)
        if len(cls_pixels) == 0:
            continue
        n = min(num_points_per_class, len(cls_pixels))
        chosen = np.random.choice(len(cls_pixels), n, replace=False)
        pts = cls_pixels[chosen]
        point_mask[pts[:, 0], pts[:, 1]] = cls

    return point_mask


class WHDLDPointDataset(Dataset):
    """
    WHDLD dataset returning point labels (training) or full masks (validation).

    Original labels are 1-6, remapped to 0-5:
        0=bare_soil, 1=building, 2=pavement, 3=road, 4=vegetation, 5=water
    """

    CLASS_NAMES = ["bare_soil", "building", "pavement",
                   "road", "vegetation", "water"]

    def __init__(self, image_dir, mask_dir, num_points_per_class=10,
                 num_classes=6, ignore_index=255, transform=None,
                 use_full_mask=False, split_file=None):
        """
        Args:
            image_dir: path to JPEGImages directory
            mask_dir: path to SegmentationClass directory
            split_file: path to train.txt / val.txt listing filenames (no ext).
                        If None, loads all images from image_dir.
        """
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                names = [line.strip() for line in f if line.strip()]
            self.images = [os.path.join(image_dir, n + ".jpg") for n in names]
            self.masks = [os.path.join(mask_dir, n + ".png") for n in names]
        else:
            self.images = sorted(glob(os.path.join(image_dir, "*.jpg")))
            self.masks = sorted(glob(os.path.join(mask_dir, "*.png")))

        assert len(self.images) > 0, f"No images found in {image_dir}"
        assert len(self.images) == len(self.masks), \
            f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}"

        self.num_points = num_points_per_class
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.transform = transform
        self.use_full_mask = use_full_mask

        mode = "full_mask" if use_full_mask else f"points ({num_points_per_class}/class)"
        print(f"  Dataset: {len(self.images)} images, mode={mode}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))

        # Remap WHDLD labels: 1-6 → 0-5; anything else → ignore
        mask = mask.astype(np.int16)
        valid = (mask >= 1) & (mask <= self.num_classes)
        remapped = np.full(mask.shape, self.ignore_index, dtype=np.uint8)
        remapped[valid] = (mask[valid] - 1).astype(np.uint8)
        mask = remapped

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask

        if self.use_full_mask:
            label = torch.from_numpy(mask_np).long() if not isinstance(mask, torch.Tensor) else mask.long()
        else:
            point_mask = simulate_point_labels(mask_np, self.num_points, self.num_classes)
            label = torch.from_numpy(point_mask).long()

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def get_class_counts(self, max_images=50):
        """Compute class pixel counts for class weighting."""
        counts = Counter()
        for i in range(min(max_images, len(self.images))):
            m = np.array(Image.open(self.masks[i]))
            # Remap 1-6 → 0-5
            m = m.astype(np.int16)
            valid = (m >= 1) & (m <= self.num_classes)
            remapped = np.full(m.shape, self.ignore_index, dtype=np.uint8)
            remapped[valid] = (m[valid] - 1).astype(np.uint8)
            for c in range(self.num_classes):
                counts[c] += int(np.sum(remapped == c))
        return [counts[c] for c in range(self.num_classes)]


def get_train_transform(patch=256):
    return A.Compose([
        A.RandomCrop(patch, patch),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(patch=256):
    return A.Compose([
        A.CenterCrop(patch, patch),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    print("Point label simulation test:")
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:128, :128] = 0
    mask[:128, 128:] = 1
    mask[128:, :128] = 4
    mask[128:, 128:] = 5
    for pts in [1, 5, 10, 20, 50]:
        pm = simulate_point_labels(mask, pts, 6)
        n = (pm != 255).sum()
        print(f"  {pts:3d} pts/class -> {n:4d} labeled ({100*n/mask.size:.3f}%)")
    print("  Verified")
