"""
Unified comparison of all uncertainty methods.
Methods: MC Dropout · Deep Ensemble · TTA · (EDL optional)

Usage:
    python experiments/compare_methods.py \
        --data_dir data/DRIVE \
        --checkpoint_dir checkpoints \
        --output_dir results/comparison
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from data.dataset import get_dataloaders
from models.unet_mc import MCDropoutUNet
from models.tta import TTAWrapper
from utils.checkpoints import load_checkpoint, load_model_state
from utils.device import get_device
from utils.io import dump_json
from utils.metrics import evaluate_all
from utils.mc_dropout import mc_dropout_predict
from utils.deferral import DeferralPolicy
from utils.stats import wilcoxon_comparison, significance_table, save_stats_report
from utils.selective_prediction import SelectivePrediction
from utils.failure_analysis import FailureModeAnalyzer

def _load_state(path, device):
    """Load checkpoint handling both full dict and bare state_dict."""
    ckpt = load_checkpoint(path, device)
    return ckpt["state_dict"], ckpt.get("val_dice", float("nan"))


def load_mc_model(checkpoint_dir, device):
    path = os.path.join(checkpoint_dir, "unet_mc_dropout", "best_model.pth")
    model = MCDropoutUNet().to(device, memory_format=torch.channels_last)
    checkpoint = load_model_state(model, path, device)
    vd = checkpoint.get("val_dice", float("nan"))
    model.eval()
    print(f"  MC Dropout loaded  (val_dice={vd:.4f})")
    return model


def load_ensemble_models(checkpoint_dir, device, n_models=5):
    """Load only ensemble_* members."""
    models = []
    for i in range(n_models):
        path = os.path.join(checkpoint_dir, f"ensemble_{i}", "best_model.pth")
        if not os.path.exists(path):
            print(f"  Warning: ensemble_{i} not found, skipping")
            continue
        m = MCDropoutUNet().to(device, memory_format=torch.channels_last)
        sd, vd = _load_state(path, device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
        print(f"  Ensemble member {i} loaded  (val_dice={vd:.4f})")
    return models


@torch.no_grad()
def run_inference(method_name, infer_fn, test_loader, device):
    all_metrics = []
    all_preds, all_gts, all_uncs, all_fovs, all_images = [], [], [], [], []

    for batch in tqdm(test_loader, desc=f"  [{method_name}]"):
        images = batch["image"].to(device, memory_format=torch.channels_last)
        masks  = batch["mask"].squeeze(1).cpu().numpy()
        fov    = batch["fov"].squeeze(1).cpu().numpy()

        output = infer_fn(images)
        mean   = output["mean"].squeeze(1).cpu().numpy()
        var    = output["variance"].squeeze(1).cpu().numpy()

        for i in range(len(mean)):
            m = evaluate_all(mean[i], masks[i], uncertainty=var[i], fov_mask=fov[i])
            all_metrics.append(m)
            all_preds.append(mean[i])
            all_gts.append(masks[i])
            all_uncs.append(var[i])
            all_fovs.append(fov[i])
            all_images.append(batch["image"][i].numpy())

    return all_metrics, all_preds, all_gts, all_uncs, all_fovs, all_images


def compare_all(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=1,
        num_workers=4, pin_memory=False,
    )

    print("\n[Compare] Loading models...")
    mc_model       = load_mc_model(args.checkpoint_dir, device)
    ensemble_models = load_ensemble_models(args.checkpoint_dir, device, args.n_models)
    tta_model = TTAWrapper(mc_model, n_augmentations=args.n_augmentations)

    def mc_infer(x):
        mean, var = mc_dropout_predict(mc_model, x, T=args.n_passes)
        return {"mean": mean, "variance": var}

    def ensemble_infer(x):
        means = []
        for m in ensemble_models:
            mn, _ = mc_dropout_predict(m, x, T=5)
            means.append(mn)
        means = torch.stack(means)
        return {"mean": means.mean(0), "variance": means.var(0)}

    methods = {
        "MC Dropout":    mc_infer,
        "Deep Ensemble": ensemble_infer,
        "TTA":           lambda x: tta_model.forward(x),
    }

    all_per_image = {}
    all_results   = {}

    for name, fn in methods.items():
        print(f"\n[Compare] Running {name}...")
        per_img, preds, gts, uncs, fovs, imgs = run_inference(
            name, fn, test_loader, device)
        all_per_image[name] = per_img

        # Deferral policy
        err_masks = [((p > 0.5) != g).astype(float) for p, g in zip(preds, gts)]
        deferral  = DeferralPolicy(uncs, err_masks, preds, gts, fovs)
        def_sum   = deferral.run(str(output_dir / name.replace(" ", "_") / "deferral"))

        # Selective prediction
        sp      = SelectivePrediction(preds, gts, uncs, fovs)
        sp_sum  = sp.run(str(output_dir / name.replace(" ", "_") / "selective"), name)

        # Failure mode analysis
        analyzer  = FailureModeAnalyzer()
        fail_sum  = analyzer.analyze_dataset(preds, gts, uncs, imgs, fovs)
        analyzer.plot_taxonomy(fail_sum,
            str(output_dir / name.replace(" ", "_") / "failure"))

        keys = ["dice", "auc", "iou", "ece", "unc_auroc"]
        agg  = {k: float(np.nanmean([m[k] for m in per_img if k in m])) for k in keys}

        all_results[name] = {"metrics": agg, "deferral": def_sum,
                             "selective": sp_sum, "failure_modes": fail_sum}

    # Statistical comparison
    print("\n[Compare] Statistical tests...")
    metric_keys = ["dice", "auc", "ece", "unc_auroc"]
    cis   = significance_table(all_per_image, metric_keys)
    stats = wilcoxon_comparison(
        all_per_image.get("MC Dropout", []),
        all_per_image.get("Deep Ensemble", []),
        metric_keys)

    report = {"method_results": all_results, "confidence_intervals": cis, "wilcoxon": stats}
    save_stats_report(report, str(output_dir / "full_comparison.json"))
    dump_json(output_dir / "method_results.json", all_results)
    print(f"\n[Compare] Done. Results in {output_dir}/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str, default="data/DRIVE")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir",     type=str, default="results/comparison")
    parser.add_argument("--img_size",       type=int, default=512)
    parser.add_argument("--n_models",       type=int, default=5)
    parser.add_argument("--n_passes",       type=int, default=10)
    parser.add_argument("--n_augmentations", type=int, default=6)
    args = parser.parse_args()
    compare_all(args)
