import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import get_dataloaders
from models.deterministic_unet import DeterministicUNet
from models.unet_mc import MCDropoutUNet
from utils.calibration import TemperatureScaler
from utils.checkpoints import load_model_state
from utils.device import get_device
from utils.io import dump_json, ensure_dir
from utils.metrics import evaluate_all, expected_calibration_error


@torch.no_grad()
def _mean_logits(model, images: torch.Tensor, deterministic: bool, n_passes: int) -> torch.Tensor:
    if deterministic:
        return model(images)

    was_training = model.training
    model.eval()
    if hasattr(model, "enable_mc"):
        model.enable_mc()
    logits = []
    for _ in range(n_passes):
        logits.append(model(images))
    if hasattr(model, "disable_mc"):
        model.disable_mc()
    model.train(was_training)
    return torch.stack(logits, dim=0).mean(dim=0)


def _collect_logits_and_labels(model, loader, device, deterministic: bool, n_passes: int):
    logits_all, labels_all = [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].squeeze(1).cpu().numpy()
        fov = batch["fov"].squeeze(1).cpu().numpy()
        logits = _mean_logits(model, images, deterministic, n_passes).squeeze(1).cpu().numpy()
        for i in range(len(logits)):
            mask = fov[i].astype(bool)
            logits_all.append(logits[i][mask])
            labels_all.append(masks[i][mask])
    return np.concatenate(logits_all), np.concatenate(labels_all)


def _evaluate_with_temperature(model, loader, device, deterministic: bool, n_passes: int, threshold: float, scaler: TemperatureScaler | None):
    per_image = []
    flat_probs, flat_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].squeeze(1).cpu().numpy()
        fov = batch["fov"].squeeze(1).cpu().numpy()
        logits = _mean_logits(model, images, deterministic, n_passes).squeeze(1).cpu().numpy()
        probs = scaler.calibrate(logits) if scaler is not None else 1.0 / (1.0 + np.exp(-logits))

        for i in range(len(probs)):
            metrics = evaluate_all(probs[i], masks[i], fov_mask=fov[i], threshold=threshold)
            per_image.append(metrics)
            mask = fov[i].astype(bool)
            flat_probs.append(probs[i][mask])
            flat_labels.append(masks[i][mask])

    flat_probs_np = np.concatenate(flat_probs)
    flat_labels_np = np.concatenate(flat_labels)
    summary = {
        key: float(np.nanmean([m[key] for m in per_image if key in m]))
        for key in per_image[0].keys()
    }
    summary["global_ece"] = expected_calibration_error(flat_probs_np, flat_labels_np)
    return summary, per_image


def main(args):
    output_dir = ensure_dir(args.output_dir)
    device = get_device()
    print(f"Using device: {device}")

    model_cls = DeterministicUNet if args.deterministic else MCDropoutUNet
    model = model_cls(encoder_name=args.encoder, dropout_p=args.dropout_p).to(device)
    ckpt = load_model_state(model, args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else float(ckpt.get("best_threshold", 0.5))
    print(f"Loaded: {args.checkpoint}  (val_dice={ckpt.get('val_dice', float('nan')):.4f})")
    print(f"Using segmentation threshold: {threshold:.3f}")

    _, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        split_seed=args.seed,
    )

    val_logits, val_labels = _collect_logits_and_labels(
        model, val_loader, device, args.deterministic, args.n_passes
    )
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels, verbose=True)

    before_summary, before_per_image = _evaluate_with_temperature(
        model, test_loader, device, args.deterministic, args.n_passes, threshold, scaler=None
    )
    after_summary, after_per_image = _evaluate_with_temperature(
        model, test_loader, device, args.deterministic, args.n_passes, threshold, scaler=scaler
    )

    comparison = scaler.compare_ece(
        probs_before=1.0 / (1.0 + np.exp(-val_logits)),
        probs_after=scaler.calibrate(val_logits),
        labels=val_labels,
        output_dir=output_dir,
    )

    result = {
        "checkpoint": args.checkpoint,
        "temperature": float(temperature),
        "threshold": float(threshold),
        "before": before_summary,
        "after": after_summary,
        "validation_calibration": comparison,
    }

    dump_json(output_dir / "temperature_scaling_results.json", result)
    dump_json(output_dir / "temperature_scaling_per_image_before.json", before_per_image)
    dump_json(output_dir / "temperature_scaling_per_image_after.json", after_per_image)
    print(f"Saved temperature scaling results to {output_dir / 'temperature_scaling_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DRIVE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/summaries"))
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--n_passes", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.set_defaults(persistent_workers=True)
    args = parser.parse_args()
    main(args)
