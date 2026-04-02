"""
Ablation Study: sensitivity to MC passes, dropout probability, image size.

Usage:
    python experiments/ablation.py \
        --data_dir data/DRIVE \
        --checkpoint checkpoints/unet_mc_dropout/best_model.pth \
        --output_dir results/ablation
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from data.dataset import get_dataloaders
from models.unet_mc import MCDropoutUNet
from utils.checkpoints import load_model_state
from utils.device import get_device
from utils.io import dump_json
from utils.metrics import evaluate_all
from utils.mc_dropout import mc_dropout_predict


def load_model(path, device, dropout_p=0.3):
    model = MCDropoutUNet(dropout_p=dropout_p).to(
        device, memory_format=torch.channels_last)
    load_model_state(model, path, device)
    model.eval()
    return model


@torch.no_grad()
def eval_mc(model, loader, device, T):
    metrics, times = [], []
    for batch in tqdm(loader, desc=f"T={T}", leave=False):
        images = batch["image"].to(device, memory_format=torch.channels_last)
        masks  = batch["mask"].squeeze(1).cpu().numpy()
        fov    = batch["fov"].squeeze(1).cpu().numpy()

        t0 = time.time()
        mean_t, var_t = mc_dropout_predict(model, images, T=T)
        times.append(time.time() - t0)

        mean = mean_t.squeeze(1).cpu().numpy()
        var  = var_t.squeeze(1).cpu().numpy()

        for i in range(len(mean)):
            m = evaluate_all(mean[i], masks[i], uncertainty=var[i], fov_mask=fov[i])
            metrics.append(m)

    return metrics, float(np.mean(times))


def run_ablation(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Ablation 1: Number of MC passes
    print("\n[Ablation 1] MC passes: T ∈ {5, 10, 20, 30, 50}")
    _, _, loader = get_dataloaders(args.data_dir, img_size=512, batch_size=1,
                                    num_workers=4, pin_memory=False)
    model = load_model(args.checkpoint, device)
    n_pass_results = {}

    for T in [5, 10, 20, 30, 50]:
        per_img, t = eval_mc(model, loader, device, T)
        n_pass_results[T] = {
            "dice":      float(np.nanmean([m["dice"]      for m in per_img])),
            "auc":       float(np.nanmean([m["auc"]       for m in per_img])),
            "ece":       float(np.nanmean([m["ece"]       for m in per_img])),
            "unc_auroc": float(np.nanmean([m.get("unc_auroc", float("nan")) for m in per_img])),
            "avg_time_s": t,
        }
        r = n_pass_results[T]
        print(f"  T={T:3d}: dice={r['dice']:.4f}  auc={r['auc']:.4f}  "
              f"ece={r['ece']:.4f}  unc_auroc={r['unc_auroc']:.4f}  t={t:.2f}s")
    results["n_passes"] = n_pass_results

    # Ablation 2: Dropout probability (test-time variation on same weights)
    print("\n[Ablation 2] Dropout p ∈ {0.1, 0.2, 0.3, 0.4, 0.5}")
    dropout_results = {}
    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        model_p = load_model(args.checkpoint, device, dropout_p=p)
        per_img, t = eval_mc(model_p, loader, device, T=20)
        dropout_results[p] = {
            "dice":      float(np.nanmean([m["dice"]      for m in per_img])),
            "auc":       float(np.nanmean([m["auc"]       for m in per_img])),
            "ece":       float(np.nanmean([m["ece"]       for m in per_img])),
            "unc_auroc": float(np.nanmean([m.get("unc_auroc", float("nan")) for m in per_img])),
        }
        r = dropout_results[p]
        print(f"  p={p:.1f}: dice={r['dice']:.4f}  unc_auroc={r['unc_auroc']:.4f}")
    results["dropout_p"] = dropout_results

    # Ablation 3: Image size
    print("\n[Ablation 3] Patch size ∈ {128, 256, 384}")
    size_results = {}
    for patch_size in [128, 256, 384]:
        _, _, size_loader = get_dataloaders(
            args.data_dir,
            img_size=args.img_size,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            patch_size=patch_size,
        )
        per_img, t = eval_mc(model, size_loader, device, T=20)
        size_results[patch_size] = {
            "dice":       float(np.nanmean([m["dice"]      for m in per_img])),
            "auc":        float(np.nanmean([m["auc"]       for m in per_img])),
            "unc_auroc":  float(np.nanmean([m.get("unc_auroc", float("nan")) for m in per_img])),
            "avg_time_s": t,
        }
        r = size_results[patch_size]
        print(f"  patch={patch_size}: dice={r['dice']:.4f}  unc_auroc={r['unc_auroc']:.4f}  t={t:.2f}s")
    results["patch_size"] = size_results

    # Save + plot
    dump_json(output_dir / "ablation_results.json", results)
    _plot_ablation(results, output_dir / "ablation_results.png")
    print(f"\n[Ablation] Saved to {output_dir}/")
    return results


def _plot_ablation(results, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metric_keys = ["dice", "auc", "unc_auroc"]
    labels      = ["Dice", "AUC", "Unc-AUROC"]
    colors      = ["#4C72B0", "#54A87A", "#9B59B6"]

    ablations = [
        ("n_passes",  "MC passes T",         [5, 10, 20, 30, 50]),
        ("dropout_p", "Dropout p",            [0.1, 0.2, 0.3, 0.4, 0.5]),
        ("patch_size",  "Patch size (px)",      [128, 256, 384]),
    ]

    for col, (key, xlabel, xvals) in enumerate(ablations):
        if key not in results:
            continue
        for row, (metric, label, color) in enumerate(zip(metric_keys[:2], labels[:2], colors[:2])):
            ax = axes[row][col]
            yvals = [results[key][x][metric] for x in xvals if x in results[key]]
            xplot = [x for x in xvals if x in results[key]]
            ax.plot(xplot, yvals, "o-", color=color, linewidth=2, markersize=6)
            if row == 0:
                ax.set_title(xlabel, fontsize=11, fontweight="bold")
            ax.set_ylabel(label, fontsize=10)
            ax.grid(alpha=0.3)
            if metric == "dice":
                ax.axhline(0.82, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            elif metric == "auc":
                ax.axhline(0.98, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.suptitle("Ablation Study", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data/DRIVE")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/unet_mc_dropout/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()
    run_ablation(args)
