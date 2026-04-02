"""
Evaluation script: full test-set evaluation with MC Dropout uncertainty.

Usage:
    python evaluate.py \
        --data_dir data/DRIVE \
        --checkpoint checkpoints/unet_mc_dropout/best_model.pth \
        --n_passes 30 \
        --output_dir results/mc_dropout
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from data.dataset import get_dataloaders
from models.deterministic_unet import DeterministicUNet
from models.unet_mc import MCDropoutUNet
from utils.checkpoints import load_model_state
from utils.device import get_device
from utils.io import dump_json, ensure_dir
from utils.deferral import DeferralPolicy
from utils.metrics import evaluate_all, reliability_diagram_data, expected_calibration_error
from utils.mc_dropout import mc_dropout_predict
from utils.reliability_checks import ReliabilityChecker
from utils.visualization import save_prediction_artifacts


def save_uncertainty_figure(image, pred_mean, uncertainty, error_mask, gt_mask, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    mean_np = np.array([0.485, 0.456, 0.406])
    std_np  = np.array([0.229, 0.224, 0.225])
    img_display = (image.transpose(1, 2, 0) * std_np + mean_np).clip(0, 1)

    axes[0].imshow(img_display);            axes[0].set_title("Input");          axes[0].axis("off")
    axes[1].imshow(pred_mean, cmap="gray"); axes[1].contour(gt_mask, levels=[0.5], colors="lime", linewidths=0.8)
    axes[1].set_title("Prediction");        axes[1].axis("off")
    im = axes[2].imshow(uncertainty, cmap="hot")
    axes[2].set_title("Epistemic uncertainty"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(error_mask, cmap="RdYlGn_r"); axes[3].set_title("Errors"); axes[3].axis("off")
    plt.suptitle("MC Dropout Uncertainty Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_reliability_diagram(centers, confidences, accuracies, ece, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect", linewidth=1.5)
    ax.bar(centers, accuracies, width=1.0/len(centers), alpha=0.7, color="#4C72B0")
    ax.plot(centers, confidences, "ro-", markersize=4, label="Confidence")
    ax.set_title(f"Reliability Diagram  (ECE={ece:.4f})")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def evaluate(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")
    ensure_dir(args.output_dir)

    model_cls = DeterministicUNet if args.deterministic else MCDropoutUNet
    model = model_cls(encoder_name=args.encoder, dropout_p=args.dropout_p).to(device)
    checkpoint = load_model_state(model, args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}  (val_dice={checkpoint.get('val_dice', float('nan')):.4f})")
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("best_threshold", 0.5))
    print(f"Using segmentation threshold: {threshold:.3f}")

    _, _, test_loader = get_dataloaders(
        root_dir=args.data_dir, img_size=args.img_size, batch_size=1,
        num_workers=4, pin_memory=False,
    )

    wandb.init(project="medical-seg-uncertainty",
               name=f"eval_{args.run_name}", config=vars(args))

    all_metrics, all_probs, all_labels, all_unc, all_err = [], [], [], [], []
    per_image_metrics = []
    all_preds, all_gts, all_fovs, all_unc_maps = [], [], [], []
    per_image_uncertainty = []
    runtime_s = []
    artifact_dir = ensure_dir(os.path.join(args.output_dir, "artifacts"))
    reliability_dir = ensure_dir(os.path.join(args.output_dir, "reliability"))

    for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = batch["image"].to(device)
        masks  = batch["mask"].squeeze(1).cpu().numpy()
        fov    = batch["fov"].squeeze(1).cpu().numpy()
        image_paths = batch.get("img_path", ["unknown"])

        start = time.perf_counter()
        if args.deterministic:
            logits = model(images)
            mean_t = torch.sigmoid(logits)
            var_t = torch.zeros_like(mean_t)
        else:
            mean_t, var_t = mc_dropout_predict(model, images, T=args.n_passes)
        mean_t = mean_t.detach()
        var_t = var_t.detach()
        runtime_s.append(time.perf_counter() - start)
        mean = mean_t.squeeze(1).cpu().numpy()
        var  = var_t.squeeze(1).cpu().numpy()

        for i in range(len(mean)):
            m = evaluate_all(mean[i], masks[i], uncertainty=var[i], fov_mask=fov[i], threshold=threshold)
            all_metrics.append(m)
            img_path = image_paths[i]
            image_id = os.path.splitext(os.path.basename(img_path))[0] if img_path != "unknown" else f"sample_{idx:02d}_{i}"
            err = ((mean[i] > threshold) != masks[i]).astype(np.float32)
            artifact_paths = save_prediction_artifacts(
                artifact_dir=artifact_dir,
                image_id=image_id,
                pred_mean=mean[i],
                uncertainty=var[i],
                error_mask=err,
                gt_mask=masks[i],
            )
            per_image_metrics.append({
                "img_path": img_path,
                "mean_uncertainty": float(var[i][fov[i].astype(bool)].mean()),
                "runtime_s": float(runtime_s[-1]),
                **artifact_paths,
                **{k: float(v) for k, v in m.items()},
            })

            fov_b = fov[i].astype(bool)
            all_probs.extend(mean[i][fov_b].tolist())
            all_labels.extend(masks[i][fov_b].tolist())
            all_unc.extend(var[i][fov_b].tolist())
            all_err.extend(err[fov_b].tolist())
            per_image_uncertainty.append({
                "img_path": img_path,
                "mean_uncertainty": float(var[i][fov_b].mean()),
            })
            all_preds.append(mean[i])
            all_gts.append(masks[i])
            all_fovs.append(fov[i])
            all_unc_maps.append(var[i])

            if idx < args.save_n_images:
                save_uncertainty_figure(
                    images[i].cpu().numpy(), mean[i], var[i], err, masks[i],
                    os.path.join(args.output_dir, f"sample_{idx:02d}.png"))

    keys    = all_metrics[0].keys()
    summary = {k: float(np.nanmean([m[k] for m in all_metrics if k in m])) for k in keys}

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    err_arr    = np.array(all_err)
    unc_arr    = np.array(all_unc)

    summary["global_ece"] = expected_calibration_error(probs_arr, labels_arr)
    summary["threshold"] = float(threshold)
    summary["uncertainty_mean"] = float(np.mean(unc_arr))
    summary["uncertainty_std"] = float(np.std(unc_arr))
    summary["uncertainty_p95"] = float(np.percentile(unc_arr, 95))
    summary["uncertainty_max"] = float(np.max(unc_arr))

    from sklearn.metrics import roc_auc_score
    if err_arr.sum() > 0:
        summary["global_unc_auroc"] = roc_auc_score(
            (err_arr > 0).astype(int), unc_arr)
    summary["avg_runtime_s_per_image"] = float(np.mean(runtime_s))
    summary["method"] = "deterministic" if args.deterministic else "mc_dropout"

    centers, conf, acc, _ = reliability_diagram_data(probs_arr, labels_arr)
    plot_reliability_diagram(centers, conf, acc, summary["global_ece"],
                             os.path.join(args.output_dir, "reliability_diagram.png"))

    error_masks = [((p > threshold) != g).astype(float) for p, g in zip(all_preds, all_gts)]
    deferral = DeferralPolicy(
        uncertainty_maps=all_unc_maps,
        error_masks=error_masks,
        pred_probs=all_preds,
        gt_masks=all_gts,
        fov_masks=all_fovs,
    )
    deferral_summary = deferral.run(os.path.join(args.output_dir, "deferral"))

    checker = ReliabilityChecker(model, device, n_passes=max(1, args.n_passes))
    overconfidence = checker.check_overconfident_failures(
        pred_probs=all_preds,
        gt_masks=all_gts,
        uncertainties=all_unc_maps,
        fov_masks=all_fovs,
        output_dir=reliability_dir,
    )

    print("\n" + "="*50 + "\nTEST SET RESULTS\n" + "="*50)
    for k, v in summary.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(f"  {k:25s}: {float(v):.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print(f"\n  Dice > 0.82 : {'✓' if summary.get('dice',0) > 0.82 else '✗'}")
    print(f"  AUC  > 0.98 : {'✓' if summary.get('auc',0) > 0.98 else '✗'}")
    print(f"  ECE  < 0.05 : {'✓' if summary.get('global_ece',1) < 0.05 else '✗'}")

    wandb.log(summary); wandb.finish()
    dump_json(os.path.join(args.output_dir, "results.json"), summary)
    dump_json(os.path.join(args.output_dir, "per_image_metrics.json"), per_image_metrics)
    dump_json(os.path.join(args.output_dir, "per_image_uncertainty.json"), per_image_uncertainty)
    dump_json(os.path.join(args.output_dir, "runtime.json"), {"avg_runtime_s_per_image": summary["avg_runtime_s_per_image"]})
    dump_json(os.path.join(args.output_dir, "deferral_operating_points.json"), deferral_summary)
    dump_json(os.path.join(args.output_dir, "overconfident_failures.json"), overconfidence)
    dump_json(
        os.path.join(args.output_dir, "config.json"),
        {"args": vars(args), "checkpoint_meta": {k: v for k, v in checkpoint.items() if k != "state_dict"}},
    )
    print(f"\nResults saved to {args.output_dir}/results.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      type=str,   default="data/DRIVE")
    parser.add_argument("--checkpoint",    type=str,   required=True)
    parser.add_argument("--run_name",      type=str,   default="mc_dropout")
    parser.add_argument("--output_dir",    type=str,   default="results/mc_dropout")
    parser.add_argument("--encoder",       type=str,   default="resnet34")
    parser.add_argument("--img_size",      type=int,   default=512)
    parser.add_argument("--dropout_p",     type=float, default=0.3)
    parser.add_argument("--n_passes",      type=int,   default=30)
    parser.add_argument("--save_n_images", type=int,   default=10)
    parser.add_argument("--threshold",     type=float, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    evaluate(args)
