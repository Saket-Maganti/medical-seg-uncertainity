from data.drive import DRIVEFullImageDataset, DRIVEPatchDataset, build_drive_loaders, get_drive_fold_indices


def get_dataloaders(
    root_dir,
    img_size=512,
    batch_size=16,
    num_workers=4,
    pin_memory=False,
    persistent_workers=True,
    val_split=0.2,
    patch_size=256,
    patches_per_image=32,
    split_seed=42,
    min_vessel_pixels=16,
    vessel_sampling_prob=0.8,
    train_indices=None,
    val_indices=None,
    train_mode="patch",
):
    return build_drive_loaders(
        root_dir=root_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        val_split=val_split,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        split_seed=split_seed,
        min_vessel_pixels=min_vessel_pixels,
        vessel_sampling_prob=vessel_sampling_prob,
        train_indices=train_indices,
        val_indices=val_indices,
        train_mode=train_mode,
    )
