"""
Evaluate the Deep Ensemble on the test set.

Usage:
    python evaluate_ensemble.py \
        --data_dir data/DRIVE \
        --checkpoint_dir checkpoints \
        --output_dir results/ensemble
"""

import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from data.dataset import get_dataloaders
from models.unet_mc import MCDropoutUNet
from utils.checkpoints import load_checkpoint
from utils.device import get_device
from utils.io import dump_json, ensure_dir
from utils.metrics import evaluate_all, expected_calibration_error, reliability_diagram_data
from utils.mc_dropout import mc_dropout_predict


def load_models(checkpoint_dir, device, encoder="resnet34", dropout_p=0.3):
    """Load only ensemble_* checkpoints — skips unet_mc_dropout and others."""
    models = []
    for name in sorted(os.listdir(checkpoint_dir)):
        # FIX: only load ensemble members, not the single MC model
        if not name.startswith("ensemble_"):
            continue
        path = os.path.join(checkpoint_dir, name, "best_model.pth")
        if not os.path.exists(path):
            continue

        model = MCDropoutUNet(encoder_name=encoder, dropout_p=dropout_p).to(
            device, memory_format=torch.channels_last)

        # FIX: handle both full dict and bare state_dict
        ckpt = load_checkpoint(path, device)
        model.load_state_dict(ckpt["state_dict"])
        val_dice = ckpt.get("val_dice", float("nan"))

        model.eval()
        models.append(model)
        print(f"  Loaded {name}  (val_dice={val_dice:.4f})")

    print(f"\nTotal ensemble models: {len(models)}")
    return models


@torch.no_grad()
def ensemble_predict(models, x, T=10):
    """Run MC Dropout on each ensemble member, average across models."""
    means = []
    for model in models:
        mean, _ = mc_dropout_predict(model, x, T=T)
        means.append(mean)
    means = torch.stack(means)          # (N, B, 1, H, W)
    final_mean = means.mean(dim=0)      # (B, 1, H, W)
    final_var  = means.var(dim=0)       # (B, 1, H, W)
    return final_mean, final_var


def main(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")
    ensure_dir(args.output_dir)

    _, _, test_loader = get_dataloaders(
        root_dir=args.data_dir, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=4, pin_memory=False,
    )

    models = load_models(args.checkpoint_dir, device, args.encoder, args.dropout_p)
    if not models:
        raise FileNotFoundError(
            f"No ensemble_* checkpoints found in {args.checkpoint_dir}. "
            "Run train_ensemble.py first.")

    all_metrics, all_probs, all_labels, all_unc, all_err = [], [], [], [], []
    per_image_metrics = []
    runtime_s = []

    for batch in tqdm(test_loader, desc="Ensemble eval"):
        images = batch["image"].to(device, memory_format=torch.channels_last)
        masks  = batch["mask"].squeeze(1).cpu().numpy()
        fov    = batch["fov"].squeeze(1).cpu().numpy()

        start = time.perf_counter()
        mean_t, var_t = ensemble_predict(models, images, T=args.mc_samples)
        runtime_s.append(time.perf_counter() - start)
        mean = mean_t.squeeze(1).cpu().numpy()
        var  = var_t.squeeze(1).cpu().numpy()

        for i in range(len(mean)):
            m = evaluate_all(mean[i], masks[i], uncertainty=var[i], fov_mask=fov[i])
            all_metrics.append(m)
            per_image_metrics.append({
                "img_path": batch.get("img_path", ["unknown"])[i],
                "mean_uncertainty": float(var[i][fov[i].astype(bool)].mean()),
                "runtime_s": float(runtime_s[-1]),
                **{k: float(v) for k, v in m.items()},
            })
            fov_b = fov[i].astype(bool)
            all_probs.extend(mean[i][fov_b].tolist())
            all_labels.extend(masks[i][fov_b].tolist())
            all_unc.extend(var[i][fov_b].tolist())
            err = ((mean[i] > 0.5) != masks[i]).astype(np.float32)
            all_err.extend(err[fov_b].tolist())

    keys    = all_metrics[0].keys()
    summary = {k: float(np.nanmean([m[k] for m in all_metrics if k in m])) for k in keys}

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    summary["global_ece"] = expected_calibration_error(probs_arr, labels_arr)

    from sklearn.metrics import roc_auc_score
    err_arr = np.array(all_err)
    if err_arr.sum() > 0:
        summary["global_unc_auroc"] = roc_auc_score(
            (err_arr > 0).astype(int), np.array(all_unc))
    summary["avg_runtime_s_per_image"] = float(np.mean(runtime_s))
    summary["method"] = "ensemble"

    print("\n===== ENSEMBLE RESULTS =====")
    for k, v in summary.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(f"  {k:25s}: {float(v):.4f}")
        else:
            print(f"  {k:25s}: {v}")

    dump_json(os.path.join(args.output_dir, "results.json"), summary)
    dump_json(os.path.join(args.output_dir, "per_image_metrics.json"), per_image_metrics)
    dump_json(os.path.join(args.output_dir, "runtime.json"), {"avg_runtime_s_per_image": summary["avg_runtime_s_per_image"]})
    dump_json(os.path.join(args.output_dir, "config.json"), vars(args))
    print(f"\nSaved to {args.output_dir}/results.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str,   default="data/DRIVE")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--output_dir",     type=str,   default="results/ensemble")
    parser.add_argument("--encoder",        type=str,   default="resnet34")
    parser.add_argument("--dropout_p",      type=float, default=0.3)
    parser.add_argument("--img_size",       type=int,   default=512)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--mc_samples",     type=int,   default=10)
    args = parser.parse_args()
    main(args)
