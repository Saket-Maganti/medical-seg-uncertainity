"""
Training script for U-Net with MC Dropout.

Usage:
    python train.py --data_dir data/DRIVE --run_name baseline
    python train.py --data_dir data/DRIVE --run_name ensemble_0 --seed 0
"""

import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from data.dataset import get_dataloaders
from models.deterministic_unet import DeterministicUNet
from models.unet_mc import MCDropoutUNet
from models.losses import build_loss
from utils.checkpoints import load_model_state, save_checkpoint
from utils.device import get_device
from utils.metrics import evaluate_all
from utils.seed import set_seed
import numpy as np


def train_one_epoch(model, loader, optimizer, criterion, device, channels_last=False):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        masks  = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        logits = logits.contiguous()
        masks = masks.contiguous()
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, channels_last=False):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels, all_fovs = [], [], []
    all_auc, all_ece = [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        masks  = batch["mask"].to(device)
        fov    = batch["fov"].squeeze(1).cpu().numpy()
        logits = model(images)
        logits = logits.contiguous()
        masks = masks.contiguous()
        loss   = criterion(logits, masks)
        total_loss += loss.item()
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        labels = masks.squeeze(1).cpu().numpy()
        for prob, label, f in zip(probs, labels, fov):
            m = evaluate_all(prob, label, fov_mask=f, threshold=0.5)
            all_auc.append(m["auc"])
            all_ece.append(m["ece"])
            all_probs.append(prob)
            all_labels.append(label)
            all_fovs.append(f)

    thresholds = np.linspace(0.3, 0.7, 17)
    threshold_scores = []
    for threshold in thresholds:
        dice_scores = [
            evaluate_all(prob, label, fov_mask=f, threshold=threshold)["dice"]
            for prob, label, f in zip(all_probs, all_labels, all_fovs)
        ]
        threshold_scores.append((float(threshold), float(np.mean(dice_scores))))

    best_threshold, best_dice = max(threshold_scores, key=lambda x: x[1])
    fixed_dice = float(
        np.mean(
            [
                evaluate_all(prob, label, fov_mask=f, threshold=0.5)["dice"]
                for prob, label, f in zip(all_probs, all_labels, all_fovs)
            ]
        )
    )
    return {
        "val_loss": total_loss / len(loader),
        "val_dice": best_dice,
        "val_threshold": best_threshold,
        "val_dice_fixed_05": fixed_dice,
        "val_auc":  np.mean(all_auc),
        "val_ece":  np.mean(all_ece),
    }


def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    wandb.init(
        project="medical-seg-uncertainty",
        name=args.run_name,
        config=vars(args),
        settings=wandb.Settings(_disable_stats=True),
    )

    train_loader, val_loader, _ = get_dataloaders(
        root_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        split_seed=args.seed,
        min_vessel_pixels=args.min_vessel_pixels,
        vessel_sampling_prob=args.vessel_sampling_prob,
        train_indices=args.train_indices,
        val_indices=args.val_indices,
        train_mode=args.train_mode,
    )

    model_cls = DeterministicUNet if args.deterministic else MCDropoutUNet
    model = model_cls(
        encoder_name=args.encoder,
        dropout_p=args.dropout_p,
    ).to(device)
    if args.resume_checkpoint:
        checkpoint = load_model_state(model, args.resume_checkpoint, device)
        print(
            f"Loaded checkpoint from {args.resume_checkpoint} "
            f"(epoch={checkpoint.get('epoch', 'unknown')}, val_dice={checkpoint.get('val_dice', 'unknown')})"
        )
    # Channels-last can trigger unsupported view/stride behavior on MPS during backward.
    # Keep it for CUDA, but disable it automatically on Apple Metal.
    use_channels_last = args.channels_last and device.type not in {"cpu", "mps"}
    if args.channels_last and not use_channels_last and device.type == "mps":
        print("Disabling --channels_last on MPS for stability.")
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = build_loss(
        loss_name=args.loss,
        alpha=args.loss_alpha,
        pos_weight=args.pos_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
        min_lr=1e-6,
    )

    patience = args.early_stopping_patience
    min_delta = args.early_stopping_min_delta
    no_improve = 0
    best_dice = 0.0
    ckpt_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, channels_last=use_channels_last)
        should_validate = epoch == 1 or epoch % args.val_interval == 0 or epoch == args.epochs
        log_payload = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}

        if should_validate:
            val_metrics = validate(model, val_loader, criterion, device, channels_last=use_channels_last)
            scheduler.step(val_metrics["val_dice"])
            log_payload.update(val_metrics)
            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                f"dice={val_metrics['val_dice']:.4f} @thr={val_metrics['val_threshold']:.2f} | "
                f"auc={val_metrics['val_auc']:.4f}"
            )

            if val_metrics["val_dice"] > best_dice + min_delta:
                best_dice = val_metrics["val_dice"]
                no_improve = 0
                save_checkpoint(
                    path=os.path.join(ckpt_dir, "best_model.pth"),
                    model=model,
                    epoch=epoch,
                    val_dice=best_dice,
                    args=vars(args),
                    seed=args.seed,
                    extra={
                        "best_threshold": val_metrics["val_threshold"],
                        "val_dice_fixed_05": val_metrics["val_dice_fixed_05"],
                        "fold_index": args.fold_index,
                    },
                )
                print(f"  ✓ Saved best model (dice={best_dice:.4f})")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                wandb.log(log_payload)
                break
        else:
            print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | val=skipped")

        wandb.log(log_payload)

    wandb.finish()
    print(f"\nBest val Dice: {best_dice:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str,   default="data/DRIVE")
    parser.add_argument("--run_name",       type=str,   default="unet_mc_dropout")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--encoder",        type=str,   default="resnet50")
    parser.add_argument("--img_size",       type=int,   default=512)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--epochs",         type=int,   default=60)
    parser.add_argument("--val_interval",   type=int,   default=2)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=3e-4)
    parser.add_argument("--dropout_p",      type=float, default=0.3)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--patch_size",     type=int,   default=384)
    parser.add_argument("--patches_per_image", type=int, default=48)
    parser.add_argument("--train_mode",     type=str,   default="patch", choices=["patch", "full"])
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--min_vessel_pixels", type=int, default=24)
    parser.add_argument("--vessel_sampling_prob", type=float, default=0.85)
    parser.add_argument("--loss",           type=str,   default="focal_tversky")
    parser.add_argument("--loss_alpha",     type=float, default=0.6)
    parser.add_argument("--pos_weight",     type=float, default=6.0)
    parser.add_argument("--focal_alpha",    type=float, default=0.8)
    parser.add_argument("--focal_gamma",    type=float, default=2.0)
    parser.add_argument("--tversky_alpha",  type=float, default=0.7)
    parser.add_argument("--tversky_beta",   type=float, default=0.3)
    parser.add_argument("--tversky_gamma",  type=float, default=1.33)
    parser.add_argument("--lr_patience",    type=int,   default=5)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.002)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--pin_memory",     action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--channels_last",  action="store_true")
    parser.add_argument("--fold_index",     type=int,   default=None)
    parser.add_argument("--deterministic",  action="store_true")
    parser.set_defaults(persistent_workers=True)
    parser.add_argument("--train_indices", nargs="*", type=int, default=None)
    parser.add_argument("--val_indices", nargs="*", type=int, default=None)
    args = parser.parse_args()
    train(args)
