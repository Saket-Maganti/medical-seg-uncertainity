"""
Evaluate the TTA baseline on DRIVE using the same output schema as the
single-model and ensemble evaluators.
"""

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.tta import TTAWrapper
from models.unet_mc import MCDropoutUNet
from utils.checkpoints import load_model_state
from utils.device import get_device
from utils.io import dump_json, ensure_dir
from utils.metrics import evaluate_all, expected_calibration_error


def evaluate_tta(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")
    ensure_dir(args.output_dir)

    model = MCDropoutUNet(encoder_name=args.encoder, dropout_p=args.dropout_p).to(device)
    checkpoint = load_model_state(model, args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}  (val_dice={checkpoint.get('val_dice', float('nan')):.4f})")
    tta = TTAWrapper(model, n_augmentations=args.n_augmentations)

    _, _, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=1,
        num_workers=4,
        pin_memory=False,
    )

    all_metrics, all_probs, all_labels, all_unc, all_err = [], [], [], [], []
    per_image_metrics = []
    runtime_s = []

    for batch in tqdm(test_loader, desc="TTA eval"):
        images = batch["image"].to(device)
        masks = batch["mask"].squeeze(1).cpu().numpy()
        fov = batch["fov"].squeeze(1).cpu().numpy()

        start = time.perf_counter()
        output = tta.forward(images)
        runtime_s.append(time.perf_counter() - start)
        mean = output["mean"].squeeze(1).cpu().numpy()
        var = output["variance"].squeeze(1).cpu().numpy()

        for i in range(len(mean)):
            metrics = evaluate_all(mean[i], masks[i], uncertainty=var[i], fov_mask=fov[i])
            all_metrics.append(metrics)
            per_image_metrics.append({
                "img_path": batch.get("img_path", ["unknown"])[i],
                "mean_uncertainty": float(var[i][fov[i].astype(bool)].mean()),
                "runtime_s": float(runtime_s[-1]),
                **{k: float(v) for k, v in metrics.items()},
            })

            fov_b = fov[i].astype(bool)
            all_probs.extend(mean[i][fov_b].tolist())
            all_labels.extend(masks[i][fov_b].tolist())
            all_unc.extend(var[i][fov_b].tolist())
            err = ((mean[i] > 0.5) != masks[i]).astype(np.float32)
            all_err.extend(err[fov_b].tolist())

    summary = {
        k: float(np.nanmean([m[k] for m in all_metrics if k in m]))
        for k in all_metrics[0].keys()
    }
    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    summary["global_ece"] = expected_calibration_error(probs_arr, labels_arr)

    err_arr = np.array(all_err)
    if err_arr.sum() > 0:
        from sklearn.metrics import roc_auc_score

        summary["global_unc_auroc"] = roc_auc_score((err_arr > 0).astype(int), np.array(all_unc))
    summary["avg_runtime_s_per_image"] = float(np.mean(runtime_s))
    summary["method"] = "tta"

    dump_json(os.path.join(args.output_dir, "results.json"), summary)
    dump_json(os.path.join(args.output_dir, "per_image_metrics.json"), per_image_metrics)
    dump_json(os.path.join(args.output_dir, "runtime.json"), {"avg_runtime_s_per_image": summary["avg_runtime_s_per_image"]})
    dump_json(
        os.path.join(args.output_dir, "config.json"),
        {"args": vars(args), "checkpoint_meta": {k: v for k, v in checkpoint.items() if k != "state_dict"}},
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DRIVE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/drive/tta")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--n_augmentations", type=int, default=6)
    args = parser.parse_args()
    evaluate_tta(args)
