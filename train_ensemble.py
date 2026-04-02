"""
Train all N ensemble members sequentially.
Multiprocessing is intentionally NOT used — MPS cannot share tensors
across processes on macOS (OS-level limitation, not fixable).

For parallel training: open 2 terminals and run members manually.

Usage:
    python train_ensemble.py --data_dir data/DRIVE --n_models 5
"""

import argparse
import copy
from train import train


def train_ensemble(args):
    for i in range(args.n_models):
        print(f"\n{'='*60}")
        print(f" Training ensemble member {i+1}/{args.n_models}  (seed={i})")
        print(f"{'='*60}\n")

        member_args = copy.deepcopy(args)
        member_args.run_name = f"ensemble_{i}"
        member_args.seed     = i
        train(member_args)

    print(f"\n✓ All {args.n_models} ensemble members trained.")
    print(f"  Checkpoints: checkpoints/ensemble_*/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str,   default="data/DRIVE")
    parser.add_argument("--n_models",       type=int,   default=5)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--encoder",        type=str,   default="resnet50")
    parser.add_argument("--img_size",       type=int,   default=512)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--epochs",         type=int,   default=40)
    parser.add_argument("--val_interval",   type=int,   default=2)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=3e-4)
    parser.add_argument("--dropout_p",      type=float, default=0.3)
    parser.add_argument("--patch_size",     type=int,   default=384)
    parser.add_argument("--patches_per_image", type=int, default=40)
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
    parser.add_argument("--lr_patience",    type=int,   default=4)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.002)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--pin_memory",     action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--channels_last",  action="store_true")
    parser.add_argument("--train_mode",     type=str,   default="patch", choices=["patch", "full"])
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--fold_index",     type=int,   default=None)
    parser.add_argument("--deterministic",  action="store_true")
    parser.add_argument("--train_indices", nargs="*", type=int, default=None)
    parser.add_argument("--val_indices", nargs="*", type=int, default=None)
    parser.set_defaults(persistent_workers=True)
    args = parser.parse_args()
    train_ensemble(args)
