import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_patch_transform() -> A.Compose:
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=3.0, p=1.0),
                    A.RandomBrightnessContrast(0.1, 0.1, p=1.0),
                    A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                ],
                p=0.25,
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_train_full_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=3.0, p=1.0),
                    A.RandomBrightnessContrast(0.1, 0.1, p=1.0),
                    A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                ],
                p=0.2,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
