"""
Cross-Dataset Generalization Evaluation.

Tests a model trained on DRIVE zero-shot on:
  - STARE  (20 images, different scanner/population)
  - CHASE_DB1 (28 images, pediatric patients)

Key question: does uncertainty remain a reliable failure predictor
on unseen datasets? If yes, the deferral policy generalizes.

Usage:
    python experiments/cross_dataset.py \
        --checkpoint checkpoints/unet_mc_dropout/best_model.pth \
        --stare_dir data/STARE \
        --chase_dir data/CHASE_DB1 \
        --output_dir results/cross_dataset
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.stare import get_stare_loader
from data.chase import get_chase_loader
from models.unet_mc import MCDropoutUNet
from utils.checkpoints import load_model_state
from utils.device import get_device
from utils.io import dump_json
from utils.metrics import evaluate_all, expected_calibration_error
from utils.deferral import DeferralPolicy
from utils.mc_dropout import mc_dropout_predict
from utils.selective_prediction import SelectivePrediction
from utils.stats import bootstrap_ci_dict
from utils.figure_style import apply_publication_style, save_figure, style_axes


apply_publication_style()


def evaluate_on_dataset(model, loader, device, n_passes, dataset_name, dataset_output_dir: Path):
    print(f"\n[CrossDataset] Evaluating on {dataset_name}...")
    per_img = []
    all_preds, all_gts, all_uncs, all_fovs = [], [], [], []
    per_image_uncertainty = []

    for batch in tqdm(loader, desc=f"  {dataset_name}"):
        images = batch["image"].to(device)
        masks  = batch["mask"].squeeze(1).cpu().numpy()
        fov    = batch["fov"].squeeze(1).cpu().numpy()

        mean_t, var_t = mc_dropout_predict(model, images, T=n_passes)
        mean = mean_t.squeeze(1).cpu().numpy()
        var = var_t.squeeze(1).cpu().numpy()

        for i in range(len(mean)):
            m = evaluate_all(mean[i], masks[i], uncertainty=var[i],
                             fov_mask=fov[i])
            per_img.append(m)
            all_preds.append(mean[i])
            all_gts.append(masks[i])
            all_uncs.append(var[i])
            all_fovs.append(fov[i])
            per_image_uncertainty.append(float(var[i][fov[i].astype(bool)].mean()))

    # Confidence intervals
    keys = ["dice", "auc", "ece", "unc_auroc"]
    cis  = bootstrap_ci_dict(per_img, keys)

    print(f"\n  {dataset_name} Results:")
    print(f"  {'Metric':<15} {'Mean':>8} {'95% CI':>20}")
    print("  " + "-" * 45)
    for k, v in cis.items():
        print(f"  {k:<15} {v['mean']:>8.4f}  [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    # Deferral policy on this dataset
    error_masks = [((p > 0.5) != g).astype(float)
                   for p, g in zip(all_preds, all_gts)]
    deferral = DeferralPolicy(all_uncs, error_masks, all_preds, all_gts, all_fovs)
    deferral.sweep_thresholds()
    deferral.find_optimal_thresholds()
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    deferral_summary = deferral.run(dataset_output_dir / "deferral")

    selective = SelectivePrediction(all_preds, all_gts, all_uncs, all_fovs)
    selective_summary = selective.run(dataset_output_dir / "selective", method_name=f"MC Dropout ({dataset_name})")

    return {
        "per_image_metrics": per_img,
        "confidence_intervals": cis,
        "deferral_optimal": deferral.optimal,
        "deferral_summary": deferral_summary,
        "selective_summary": selective_summary,
        "summary": {k: v["mean"] for k, v in cis.items()},
        "per_image_uncertainty": per_image_uncertainty,
    }


def cross_dataset_eval(args):
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MCDropoutUNet(encoder_name=args.encoder, dropout_p=args.dropout_p).to(device)
    ckpt = load_model_state(model, args.checkpoint, device)
    model.eval()
    print(f"Loaded: {args.checkpoint}  (val_dice={ckpt.get('val_dice', float('nan')):.4f})")

    all_dataset_results = {}

    # STARE
    if args.stare_dir and Path(args.stare_dir).exists():
        loader = get_stare_loader(args.stare_dir, img_size=args.img_size)
        results = evaluate_on_dataset(
            model, loader, device, args.n_passes, "STARE", output_dir / "stare"
        )
        all_dataset_results["STARE"] = results
    else:
        print(f"[CrossDataset] STARE dir not found: {args.stare_dir}")
        print("  Download from: http://cecas.clemson.edu/~ahoover/stare/")

    # CHASE_DB1
    if args.chase_dir and Path(args.chase_dir).exists():
        loader = get_chase_loader(args.chase_dir, img_size=args.img_size)
        results = evaluate_on_dataset(
            model, loader, device, args.n_passes, "CHASE_DB1", output_dir / "chase_db1"
        )
        all_dataset_results["CHASE_DB1"] = results
    else:
        print(f"[CrossDataset] CHASE_DB1 dir not found: {args.chase_dir}")
        print("  Download from: https://blogs.kingston.ac.uk/retinal/chasedb1/")

    # Comparison plot: DRIVE vs STARE vs CHASE
    if all_dataset_results:
        _plot_cross_dataset(all_dataset_results, output_dir / "cross_dataset.png")
        _plot_uncertainty_shift(all_dataset_results, output_dir / "uncertainty_shift.png")

    # Save
    summary = {
        k: {
            **v["summary"],
            "deferral_balanced_f1_coverage": v["deferral_summary"]["operating_points"]["balanced_f1"]["coverage"],
            "deferral_balanced_f1_dice_acc": v["deferral_summary"]["operating_points"]["balanced_f1"]["dice_accepted"],
            "selective_aucc_dice": v["selective_summary"]["aucc_dice"],
            "selective_aucc_auc": v["selective_summary"]["aucc_auc"],
        }
        for k, v in all_dataset_results.items()
    }
    dump_json(output_dir / "cross_dataset_results.json", summary)
    dump_json(output_dir / "cross_dataset_full_results.json", all_dataset_results)

    print(f"\n[CrossDataset] Results saved to {output_dir}/")
    return all_dataset_results


def _plot_cross_dataset(all_results, save_path):
    datasets = list(all_results.keys())
    metrics  = ["dice", "auc", "ece", "unc_auroc"]
    labels   = ["Dice", "AUC", "ECE", "Unc-AUROC"]
    colors   = ["#54A87A", "#E07B54", "#9B59B6"]

    fig, axes = plt.subplots(1, 4, figsize=(16.2, 5.3), constrained_layout=True)

    for col, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[col]
        means = [all_results[ds]["summary"].get(metric, 0) for ds in datasets]
        lo    = [all_results[ds]["confidence_intervals"].get(metric, {}).get("ci_lo", m)
                 for ds, m in zip(datasets, means)]
        hi    = [all_results[ds]["confidence_intervals"].get(metric, {}).get("ci_hi", m)
                 for ds, m in zip(datasets, means)]
        errs  = [[m - l for m, l in zip(means, lo)],
                 [h - m for m, h in zip(means, hi)]]

        ax.bar(
            datasets,
            means,
            color=colors[:len(datasets)],
            alpha=0.84,
            yerr=errs,
            capsize=5,
            error_kw={"linewidth": 1.5},
        )
        ax.set_title(label, fontsize=12)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.08, y=0.1)
        style_axes(ax)

        if metric == "dice":
            ax.axhline(0.82, color="red", linestyle="--", linewidth=0.8)
        elif metric == "auc":
            ax.axhline(0.98, color="red", linestyle="--", linewidth=0.8)

    plt.suptitle("Cross-Dataset Generalization (zero-shot)\n"
                 "Error bars = 95% bootstrap CI", fontsize=13)
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")


def _plot_uncertainty_shift(all_results, save_path):
    datasets = list(all_results.keys())
    values = [all_results[ds].get("per_image_uncertainty", []) for ds in datasets]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.violinplot(values, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(datasets) + 1))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean image uncertainty")
    ax.set_title("Uncertainty shift across datasets")
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.08, y=0.1)
    style_axes(ax)
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str,
                        default="checkpoints/unet_mc_dropout/best_model.pth")
    parser.add_argument("--stare_dir",   type=str, default="data/STARE")
    parser.add_argument("--chase_dir",   type=str, default="data/CHASE_DB1")
    parser.add_argument("--output_dir",  type=str, default="results/cross_dataset")
    parser.add_argument("--encoder",     type=str, default="resnet34")
    parser.add_argument("--dropout_p",   type=float, default=0.3)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--n_passes",    type=int, default=30)
    args = parser.parse_args()
    cross_dataset_eval(args)
