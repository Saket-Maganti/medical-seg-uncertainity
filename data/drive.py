from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.transforms import get_eval_transform, get_train_full_transform, get_train_patch_transform


class DRIVEPatchDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "training",
        patch_size: int = 256,
        patches_per_image: int = 32,
        augment: bool = True,
        image_indices: list[int] | None = None,
        min_vessel_pixels: int = 16,
        vessel_sampling_prob: float = 0.8,
    ):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.min_vessel_pixels = min_vessel_pixels
        self.vessel_sampling_prob = vessel_sampling_prob

        img_dir = Path(root_dir) / split / "images"
        mask_dir = Path(root_dir) / split / "1st_manual"
        fov_dir = Path(root_dir) / split / "mask"

        img_paths = sorted(img_dir.glob("*.tif"))
        mask_paths = sorted(mask_dir.glob("*.gif"))
        fov_paths = sorted(fov_dir.glob("*.gif"))

        self.images, self.masks, self.fovs = [], [], []
        for ip, mp, fp in zip(img_paths, mask_paths, fov_paths):
            self.images.append(np.array(Image.open(ip).convert("RGB")))
            self.masks.append((np.array(Image.open(mp).convert("L")) > 127).astype(np.float32))
            self.fovs.append((np.array(Image.open(fp).convert("L")) > 127).astype(np.float32))

        if image_indices is not None:
            self.images = [self.images[i] for i in image_indices]
            self.masks = [self.masks[i] for i in image_indices]
            self.fovs = [self.fovs[i] for i in image_indices]

        self.n_images = len(self.images)
        self.transform = get_train_patch_transform() if augment else get_eval_transform(patch_size)

    def __len__(self):
        return self.n_images * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx % self.n_images
        img = self.images[img_idx]
        mask = self.masks[img_idx]
        fov = self.fovs[img_idx]
        height, width = img.shape[:2]
        patch = self.patch_size

        y = 0
        x = 0
        max_y = height - patch
        max_x = width - patch
        prefer_vessel = np.random.rand() < self.vessel_sampling_prob
        for _ in range(50):
            y = np.random.randint(0, max_y + 1)
            x = np.random.randint(0, max_x + 1)
            mask_patch = mask[y : y + patch, x : x + patch]
            fov_patch = fov[y : y + patch, x : x + patch]
            if fov_patch.mean() <= 0.5:
                continue
            if prefer_vessel:
                if mask_patch.sum() >= self.min_vessel_pixels:
                    break
            else:
                break

        image_patch = img[y : y + patch, x : x + patch]
        mask_patch = mask[y : y + patch, x : x + patch]
        fov_patch = fov[y : y + patch, x : x + patch]

        aug = self.transform(image=image_patch, masks=[mask_patch, fov_patch])
        image = aug["image"]
        mask_t, fov_t = aug["masks"]
        return {
            "image": image,
            "mask": mask_t.unsqueeze(0),
            "fov": fov_t.unsqueeze(0),
        }


class DRIVEFullImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        img_size: int = 512,
        image_indices: list[int] | None = None,
    ):
        img_dir = Path(root_dir) / split / "images"
        mask_dir = Path(root_dir) / split / "1st_manual"
        fov_dir = Path(root_dir) / split / "mask"

        self.img_paths = sorted(img_dir.glob("*.tif"))
        self.mask_paths = sorted(mask_dir.glob("*.gif"))
        self.fov_paths = sorted(fov_dir.glob("*.gif"))

        if image_indices is not None:
            self.img_paths = [self.img_paths[i] for i in image_indices]
            self.mask_paths = [self.mask_paths[i] for i in image_indices]
            self.fov_paths = [self.fov_paths[i] for i in image_indices]

        self.transform = get_eval_transform(img_size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = (np.array(Image.open(self.mask_paths[idx]).convert("L")) > 127).astype(np.float32)
        fov = (np.array(Image.open(self.fov_paths[idx]).convert("L")) > 127).astype(np.float32)
        aug = self.transform(image=image, masks=[mask, fov])
        image = aug["image"]
        mask_t, fov_t = aug["masks"]
        return {
            "image": image,
            "mask": mask_t.unsqueeze(0),
            "fov": fov_t.unsqueeze(0),
            "img_path": str(self.img_paths[idx]),
        }


class DRIVETrainFullImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "training",
        img_size: int = 512,
        image_indices: list[int] | None = None,
    ):
        img_dir = Path(root_dir) / split / "images"
        mask_dir = Path(root_dir) / split / "1st_manual"
        fov_dir = Path(root_dir) / split / "mask"

        self.img_paths = sorted(img_dir.glob("*.tif"))
        self.mask_paths = sorted(mask_dir.glob("*.gif"))
        self.fov_paths = sorted(fov_dir.glob("*.gif"))

        if image_indices is not None:
            self.img_paths = [self.img_paths[i] for i in image_indices]
            self.mask_paths = [self.mask_paths[i] for i in image_indices]
            self.fov_paths = [self.fov_paths[i] for i in image_indices]

        self.transform = get_train_full_transform(img_size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = (np.array(Image.open(self.mask_paths[idx]).convert("L")) > 127).astype(np.float32)
        fov = (np.array(Image.open(self.fov_paths[idx]).convert("L")) > 127).astype(np.float32)
        aug = self.transform(image=image, masks=[mask, fov])
        image = aug["image"]
        mask_t, fov_t = aug["masks"]
        return {
            "image": image,
            "mask": mask_t.unsqueeze(0),
            "fov": fov_t.unsqueeze(0),
            "img_path": str(self.img_paths[idx]),
        }


def build_drive_loaders(
    root_dir: str,
    img_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    val_split: float = 0.2,
    patch_size: int = 256,
    patches_per_image: int = 32,
    split_seed: int = 42,
    min_vessel_pixels: int = 16,
    vessel_sampling_prob: float = 0.8,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    train_mode: str = "patch",
):
    all_img_paths = sorted((Path(root_dir) / "training" / "images").glob("*.tif"))
    n_images = len(all_img_paths)
    if train_indices is None or val_indices is None:
        n_val = max(1, int(n_images * val_split))
        n_train = n_images - n_val
        generator = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(n_images, generator=generator).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
    else:
        train_idx = train_indices
        val_idx = val_indices

    if train_mode == "full":
        train_ds = DRIVETrainFullImageDataset(
            root_dir=root_dir,
            split="training",
            img_size=img_size,
            image_indices=train_idx,
        )
    else:
        train_ds = DRIVEPatchDataset(
            root_dir=root_dir,
            split="training",
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            augment=True,
            image_indices=train_idx,
            min_vessel_pixels=min_vessel_pixels,
            vessel_sampling_prob=vessel_sampling_prob,
        )
    val_ds = DRIVEFullImageDataset(
        root_dir=root_dir,
        split="training",
        img_size=img_size,
        image_indices=val_idx,
    )
    test_ds = DRIVEFullImageDataset(root_dir=root_dir, split="test", img_size=img_size)

    use_persistent_workers = persistent_workers and num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        drop_last=train_mode == "patch",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return train_loader, val_loader, test_loader


def get_drive_fold_indices(
    root_dir: str,
    n_splits: int = 5,
    fold: int = 0,
    split_seed: int = 42,
):
    all_img_paths = sorted((Path(root_dir) / "training" / "images").glob("*.tif"))
    n_images = len(all_img_paths)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation")
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}]")

    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_images, generator=generator).tolist()
    fold_sizes = [n_images // n_splits] * n_splits
    for i in range(n_images % n_splits):
        fold_sizes[i] += 1

    offsets = [0]
    for size in fold_sizes:
        offsets.append(offsets[-1] + size)

    val_idx = perm[offsets[fold] : offsets[fold + 1]]
    train_idx = perm[: offsets[fold]] + perm[offsets[fold + 1] :]
    return train_idx, val_idx
