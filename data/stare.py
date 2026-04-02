"""
STARE (Structured Analysis of the Retina) Dataset Loader.

Dataset: http://cecas.clemson.edu/~ahoover/stare/
20 retinal fundus images (605×700), manual vessel annotations by two experts.
We use the first expert's annotations (ah labels) as ground truth.

Download structure expected:
    data/STARE/
        images/     *.ppm
        labels-ah/  *.ppm  (first expert ground truth)
        labels-vk/  *.ppm  (second expert — optional)
"""

import numpy as np
import gzip
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class STAREDataset(Dataset):
    """
    STARE dataset loader for cross-dataset generalization evaluation.
    Models trained on DRIVE are evaluated zero-shot on STARE.
    """

    def __init__(self, root_dir: str, transform=None, use_second_expert=False):
        self.root_dir  = Path(root_dir)
        self.transform = transform

        self.img_dir  = self.root_dir / "images"
        label_dir     = "labels-vk" if use_second_expert else "labels-ah"
        self.mask_dir = self.root_dir / label_dir

        self.img_paths  = sorted(self.img_dir.glob("*.ppm")) + sorted(self.img_dir.glob("*.ppm.gz"))
        self.mask_paths = (
            sorted(self.mask_dir.glob("*.ppm")) + sorted(self.mask_dir.glob("*.ppm.gz"))
            if self.mask_dir.exists()
            else []
        )

        if len(self.img_paths) == 0:
            # Try tif/gif fallbacks
            self.img_paths  = sorted(list(self.img_dir.glob("*.tif")) +
                                      list(self.img_dir.glob("*.png")))
            self.mask_paths = (
                sorted(list(self.mask_dir.glob("*.tif")) +
                       list(self.mask_dir.glob("*.png")))
                if self.mask_dir.exists()
                else []
            )

        if len(self.mask_paths) == 0:
            # Support mixed-folder STARE dumps where images and labels coexist in one directory.
            mixed_ppm = sorted(self.root_dir.glob("*.ppm"))
            if mixed_ppm:
                ah = [p for p in mixed_ppm if p.name.endswith(".ah.ppm")]
                vk = [p for p in mixed_ppm if p.name.endswith(".vk.ppm")]
                imgs = [p for p in mixed_ppm if not p.name.endswith((".ah.ppm", ".vk.ppm"))]
                chosen = vk if use_second_expert else ah
                if imgs and chosen:
                    self.img_paths = imgs
                    self.mask_paths = chosen
                    self.img_dir = self.root_dir
                    self.mask_dir = self.root_dir

        if len(self.img_paths) > 0 and len(self.mask_paths) == 0:
            raise FileNotFoundError(
                "STARE annotations were not found. Expected one of:\n"
                f"  1. {self.root_dir / 'labels-ah'} with vessel masks\n"
                "  2. A mixed-folder layout containing files like 'im0001.ah.ppm'\n\n"
                f"Found {len(self.img_paths)} image files but 0 annotation files.\n"
                "Please import the STARE vessel labels before running cross-dataset evaluation."
            )

        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(
                f"STARE image/mask count mismatch: {len(self.img_paths)} images vs "
                f"{len(self.mask_paths)} masks in {self.root_dir}"
            )

        assert len(self.img_paths) > 0, \
            f"No images found in {self.img_dir}. " \
            f"Download STARE from http://cecas.clemson.edu/~ahoover/stare/"

        print(f"[STAREDataset] Loaded {len(self.img_paths)} images")

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                return np.array(Image.open(f).convert("RGB"))
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                return np.array(Image.open(f).convert("L"))
        return np.array(Image.open(path).convert("L"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self._read_image(self.img_paths[idx])
        mask  = self._read_mask(self.mask_paths[idx])
        mask  = (mask > 127).astype(np.float32)

        # STARE has no FOV mask — use full image
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


def get_stare_loader(root_dir: str, img_size: int = 512,
                      batch_size: int = 1, num_workers: int = 2):
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    ds = STAREDataset(root_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
