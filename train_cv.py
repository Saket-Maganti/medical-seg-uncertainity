import argparse
import copy
import json
from pathlib import Path

import numpy as np

from data.drive import get_drive_fold_indices
from evaluate import evaluate
from train import train
from utils.io import dump_json, ensure_dir
from utils.stats import bootstrap_ci_dict


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _make_patch_args(args, fold: int, train_idx, val_idx):
    patch_args = copy.deepcopy(args)
    patch_args.run_name = f"{args.run_name}_fold{fold}_patch"
    patch_args.train_indices = train_idx
    patch_args.val_indices = val_idx
    patch_args.fold_index = fold
    patch_args.train_mode = "patch"
    patch_args.batch_size = args.batch_batch_size
    patch_args.epochs = args.patch_epochs
    patch_args.lr = args.patch_lr
    patch_args.loss = args.patch_loss
    patch_args.loss_alpha = args.patch_loss_alpha
    patch_args.pos_weight = args.patch_pos_weight
    patch_args.resume_checkpoint = None
    return patch_args


def _make_full_args(args, fold: int, train_idx, val_idx, resume_checkpoint: Path):
    full_args = copy.deepcopy(args)
    full_args.run_name = f"{args.run_name}_fold{fold}_fullft"
    full_args.train_indices = train_idx
    full_args.val_indices = val_idx
    full_args.fold_index = fold
    full_args.train_mode = "full"
    full_args.batch_size = args.full_batch_size
    full_args.epochs = args.full_epochs
    full_args.val_interval = 1
    full_args.lr = args.full_lr
    full_args.loss = args.full_loss
    full_args.loss_alpha = args.full_loss_alpha
    full_args.pos_weight = args.full_pos_weight
    full_args.resume_checkpoint = str(resume_checkpoint)
    return full_args


def _make_eval_args(args, fold: int, checkpoint: Path, output_dir: Path):
    return argparse.Namespace(
        data_dir=args.data_dir,
        checkpoint=str(checkpoint),
        run_name=f"{args.run_name}_fold{fold}_eval",
        output_dir=str(output_dir),
        encoder=args.encoder,
        img_size=args.img_size,
        dropout_p=args.dropout_p,
        n_passes=args.n_passes,
        save_n_images=0,
        threshold=None,
        deterministic=False,
    )


def _aggregate_fold_reports(fold_reports: list[dict]):
    metric_keys = ["dice", "auc", "ece", "global_ece", "global_unc_auroc", "avg_runtime_s_per_image"]
    rows = []
    for report in fold_reports:
        eval_summary = report.get("evaluation", {})
        row = {"fold": report["fold"]}
        for key in metric_keys:
            if key in eval_summary:
                row[key] = eval_summary[key]
        rows.append(row)

    aggregate = {}
    for key in metric_keys:
        vals = np.array([row[key] for row in rows if key in row and not np.isnan(row[key])], dtype=np.float32)
        if vals.size == 0:
            continue
        aggregate[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return rows, aggregate


def run_cross_validation(args):
    output_dir = ensure_dir(args.output_dir)
    fold_reports = []
    all_per_image_metrics = []

    for fold in range(args.n_splits):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold + 1}/{args.n_splits}")
        print(f"{'=' * 72}\n")

        train_idx, val_idx = get_drive_fold_indices(
            root_dir=args.data_dir,
            n_splits=args.n_splits,
            fold=fold,
            split_seed=args.seed,
        )

        fold_dir = ensure_dir(output_dir / f"fold_{fold}")
        patch_dir = ensure_dir(fold_dir / "patch_train")
        full_dir = ensure_dir(fold_dir / "fullft_train")
        eval_dir = ensure_dir(fold_dir / "evaluation")

        patch_args = _make_patch_args(args, fold, train_idx, val_idx)
        patch_args.checkpoint_dir = str(patch_dir)
        train(patch_args)
        patch_ckpt = patch_dir / patch_args.run_name / "best_model.pth"

        full_args = _make_full_args(args, fold, train_idx, val_idx, patch_ckpt)
        full_args.checkpoint_dir = str(full_dir)
        train(full_args)
        full_ckpt = full_dir / full_args.run_name / "best_model.pth"

        eval_args = _make_eval_args(args, fold, full_ckpt, eval_dir)
        eval_summary = evaluate(eval_args)
        per_image = _load_json(eval_dir / "per_image_metrics.json") or []
        all_per_image_metrics.extend(per_image)

        report = {
            "fold": fold,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "patch_checkpoint": str(patch_ckpt),
            "full_checkpoint": str(full_ckpt),
            "evaluation": eval_summary,
        }
        dump_json(fold_dir / "fold_summary.json", report)
        fold_reports.append(report)

    fold_rows, aggregate = _aggregate_fold_reports(fold_reports)
    ci_metrics = bootstrap_ci_dict(
        all_per_image_metrics,
        keys=["dice", "auc", "ece", "unc_auroc"],
        n_bootstrap=args.n_bootstrap,
    ) if all_per_image_metrics else {}

    summary = {
        "folds": fold_rows,
        "aggregate": aggregate,
        "bootstrap_ci": ci_metrics,
        "n_folds": args.n_splits,
        "run_name": args.run_name,
        "official_pipeline": {
            "stage1": "patch warm-up",
            "stage2": "full-image fine-tuning",
            "inference": "mc_dropout",
        },
    }
    dump_json(output_dir / "crossval_summary.json", summary)
    print(f"\nCross-validation summary saved to {output_dir / 'crossval_summary.json'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DRIVE")
    parser.add_argument("--run_name", type=str, default="unet_mc_dropout_cv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=Path, default=Path("results/crossval"))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--n_passes", type=int, default=30)

    parser.add_argument("--patch_epochs", type=int, default=30)
    parser.add_argument("--patch_lr", type=float, default=1e-4)
    parser.add_argument("--batch_batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=384)
    parser.add_argument("--patches_per_image", type=int, default=40)
    parser.add_argument("--min_vessel_pixels", type=int, default=24)
    parser.add_argument("--vessel_sampling_prob", type=float, default=0.85)
    parser.add_argument("--patch_loss", type=str, default="focal_tversky")
    parser.add_argument("--patch_loss_alpha", type=float, default=0.6)
    parser.add_argument("--patch_pos_weight", type=float, default=6.0)

    parser.add_argument("--full_epochs", type=int, default=20)
    parser.add_argument("--full_lr", type=float, default=3e-5)
    parser.add_argument("--full_batch_size", type=int, default=1)
    parser.add_argument("--full_loss", type=str, default="dice_bce")
    parser.add_argument("--full_loss_alpha", type=float, default=0.6)
    parser.add_argument("--full_pos_weight", type=float, default=4.0)

    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--loss", type=str, default="focal_tversky")
    parser.add_argument("--loss_alpha", type=float, default=0.6)
    parser.add_argument("--pos_weight", type=float, default=6.0)
    parser.add_argument("--focal_alpha", type=float, default=0.8)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--tversky_alpha", type=float, default=0.7)
    parser.add_argument("--tversky_beta", type=float, default=0.3)
    parser.add_argument("--tversky_gamma", type=float, default=1.33)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.002)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--train_indices", nargs="*", type=int, default=None)
    parser.add_argument("--val_indices", nargs="*", type=int, default=None)
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default="patch", choices=["patch", "full"])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.set_defaults(persistent_workers=True)
    args = parser.parse_args()
    run_cross_validation(args)
