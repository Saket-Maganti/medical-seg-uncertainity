# Medical Segmentation Uncertainty

Research codebase for retinal vessel segmentation with predictive uncertainty,
calibration checks, selective prediction, deferral analysis, cross-dataset
evaluation, and ablation studies.

This repository currently includes the codebase and generated experimental
results. The manuscript sources are being kept out of scope for now.

## What is included

- Training and evaluation scripts for MC Dropout, deterministic, ensemble, and
  test-time augmentation baselines
- Configuration files for the main DRIVE benchmark, cross-dataset runs, and
  ablation sweeps
- Utility code for metrics, calibration, selective prediction, deferral, and
  visualization
- Generated result folders and summary tables for the reported experiments

## Repository layout

- `train.py`: train a single MC Dropout U-Net
- `train_ensemble.py`: train ensemble members with different seeds
- `train_cv.py`: run cross-validation experiments
- `evaluate.py`: evaluate a single model on DRIVE
- `evaluate_ensemble.py`: evaluate a deep ensemble on DRIVE
- `eval_tta.py`: evaluate the TTA baseline on DRIVE
- `configs/`: experiment configurations
- `models/`: segmentation models and losses
- `utils/`: metrics, calibration, uncertainty, decision, and plotting helpers
- `experiments/`: comparison, ablation, statistics, and figure-generation scripts
- `results/`: saved outputs, metrics, and aggregate summaries

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main commands

```bash
python train.py --config configs/drive_mc.yaml
python evaluate.py --config configs/drive_mc.yaml
python train_ensemble.py --config configs/drive_ensemble.yaml
python evaluate_ensemble.py --config configs/drive_ensemble.yaml
python eval_tta.py --config configs/drive_tta.yaml
python experiments/cross_dataset.py --config configs/cross_dataset.yaml
python experiments/ablation.py --config configs/ablations.yaml
```

## Result folders

- `results/drive/`: main DRIVE benchmark outputs, qualitative samples, runtime,
  reliability, and deferral artifacts
- `results/cross_dataset/`: zero-shot evaluation results for STARE and
  CHASE_DB1
- `results/crossval/`: fold-wise cross-validation outputs
- `results/ablations/`: parameter sweep outputs for MC passes, dropout, patch
  size, and ensemble size
- `results/summaries/`: compact CSV and JSON summaries for downstream reporting
- `results/paper_artifacts/`: exported figures derived from the above results

See `results/README.md` for a compact guide to the shipped outputs.
