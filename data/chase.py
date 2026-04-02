"""
CHASE_DB1 Dataset Loader.

Dataset: https://blogs.kingston.ac.uk/retinal/chasedb1/
28 retinal images from 14 children (left + right eye).
Two manual annotations available — we use the first (1stHO).
Image size: 999×960 pixels.

Download structure expected:
    data/CHASE_DB1/
        Image_01L.jpg
        Image_01L_1stHO.png
        Image_01L_2ndHO.png
        Image_01R.jpg
        ... (28 images total)
"""

import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CHASEDB1Dataset(Dataset):
    """
    CHASE_DB1 dataset loader.
    Files follow naming: Image_XXY.jpg / Image_XXY_1stHO.png
    """

    def __init__(self, root_dir: str, transform=None, use_second_expert=False):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        expert_tag     = "2ndHO" if use_second_expert else "1stHO"

        # Find all image files
        self.img_paths  = sorted(self.root_dir.glob("Image_*.jpg")) + \
                          sorted(self.root_dir.glob("Image_*.png"))
        # Exclude annotation files
        self.img_paths  = [p for p in self.img_paths if "HO" not in p.name]
        self.img_paths  = sorted(self.img_paths)

        # Match annotation files
        self.mask_paths = []
        for img_path in self.img_paths:
            stem = img_path.stem          # e.g. "Image_01L"
            mask_name = f"{stem}_{expert_tag}.png"
            mask_path = self.root_dir / mask_name
            if not mask_path.exists():
                # Try .jpg variant
                mask_path = self.root_dir / f"{stem}_{expert_tag}.jpg"
            self.mask_paths.append(mask_path)

        assert len(self.img_paths) > 0, \
            f"No images found in {self.root_dir}. " \
            f"Download CHASE_DB1 from https://blogs.kingston.ac.uk/retinal/chasedb1/"

        print(f"[CHASEDB1Dataset] Loaded {len(self.img_paths)} images")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask  = (mask > 127).astype(np.float32)

        # No FOV mask for CHASE_DB1 — use full image
        fov   = np.ones(mask.shape, dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, masks=[mask, fov])
            image = augmented["image"]
            mask, fov = augmented["masks"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).unsqueeze(0)
            fov   = torch.from_numpy(fov).unsqueeze(0)

        return {"image": image, "mask": mask, "fov": fov,
                "img_path": str(self.img_paths[idx])}


def get_chase_loader(root_dir: str, img_size: int = 512,
                      batch_size: int = 1, num_workers: int = 2):
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    ds = CHASEDB1Dataset(root_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
