# From Prediction to Decision: Uncertainty-Aware Deferral for Reliable Medical Image Segmentation

This repository contains the code, experiment artifacts, and paper sources for uncertainty-aware retinal vessel segmentation on DRIVE, STARE, and CHASE_DB1. The project studies not only how to predict segmentation masks, but how to decide when those predictions should be accepted automatically versus deferred for review.

Using a U-Net-based segmentation pipeline with uncertainty estimation and selective deferral, the best configuration in the repository achieves about 80% error reduction while deferring only 25% of pixels, showing that uncertainty becomes most useful when paired with an explicit decision policy.

## Key Contributions

- Reframes medical image segmentation as a decision problem, not just a prediction problem, by coupling segmentation with uncertainty-aware deferral.
- Implements and compares two uncertainty sources on the same task: MC Dropout and Test-Time Augmentation (TTA).
- Evaluates three deferral strategies: global thresholding, image-adaptive deferral, and confidence-aware deferral.
- Includes calibration analysis with temperature scaling and shows that improved calibration does not necessarily improve downstream deferral quality.
- Provides selective prediction outputs, risk-coverage curves, reliability diagrams, cross-dataset evaluation, and ablation studies in a single reproducible codebase.
- Demonstrates that TTA produces stronger uncertainty signals than MC Dropout for downstream error detection and deferral.

## Method Overview

The core model is a U-Net-style retinal vessel segmentation network with a ResNet encoder. At inference time, the repository estimates uncertainty in two ways:

- `MC Dropout`: runs multiple stochastic forward passes with dropout enabled and measures disagreement across predictions.
- `TTA`: runs deterministic test-time augmentations and measures prediction variance across transformed views.

The project then uses uncertainty for `deferral`, meaning the system can abstain on uncertain pixels and send them for review instead of forcing an automatic segmentation everywhere. Three policies are implemented:

- `global`: a single threshold over uncertainty.
- `adaptive`: per-image percentile thresholding.
- `adaptive_confidence`: a confidence-aware score that combines uncertainty with distance from the decision boundary.

This lets the pipeline optimize for operational outcomes such as error reduction at fixed review budgets, rather than segmentation quality alone.

## Results

The repository includes full evaluation artifacts under `results/`, with the strongest decision-time result coming from TTA plus adaptive deferral.

### Headline Findings

- `TTA > MC Dropout` for uncertainty quality on DRIVE.
- `~80% error reduction` is achieved by TTA with adaptive deferral at 25% deferred pixels.
- `Confidence-aware deferral` is the most efficient low-budget policy, delivering strong error reduction at substantially lower deferral rates than global thresholding.
- `Calibration is not enough`: temperature scaling changes ECE, but does not improve downstream decision quality in the main comparisons.

### Example Metrics From Shipped Results

| Setting | Dice | AUC | ECE | Unc-AUROC | Error Reduction | Deferred Pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MC Dropout | 0.764 | 0.966 | 0.038 | 0.768 | 26.5% | 19.7% |
| TTA | 0.768 | 0.969 | 0.037 | 0.881 | 42.2% | 20.0% |
| TTA + Adaptive Deferral | 0.768 | 0.969 | 0.037 | 0.881 | 79.4% | 25.0% |
| TTA + Confidence-Aware Deferral | 0.768 | 0.969 | 0.037 | 0.881 | 53.8% | 12.0% |

### Interpretation

- TTA uncertainty separates correct from incorrect pixels better than MC Dropout (`Unc-AUROC 0.881` vs `0.768` in the main result artifacts).
- TTA adaptive deferral reduces mean pixel error from `0.0639` to `0.0132`, corresponding to about `79.4%` error reduction.
- Confidence-aware deferral is a strong operating point when review budget is limited: TTA confidence-aware deferral reduces error by about `53.8%` while deferring only about `12%` of pixels.
- Cross-dataset uncertainty quality remains strong under shift, with `Unc-AUROC 0.846` on STARE and `0.835` on CHASE_DB1.

## Visual Results

### Method Comparison

![Method comparison](results/comparison_final/comparison/method_comparison_bars.png)

### Deferral Mode Comparison

![TTA deferral comparison](results/comparison_final/comparison/deferral_3mode_tta.png)

### Qualitative Comparison

![Qualitative comparison](results/comparison_final/comparison/qualitative_comparison.png)

## Installation

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Main Dependencies

- `torch`
- `torchvision`
- `segmentation-models-pytorch`
- `albumentations`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `wandb`
- `tqdm`

## Usage

### 1. Train the main segmentation model

```bash
python train.py \
  --data_dir data/DRIVE \
  --run_name unet_mc_dropout_fullft \
  --encoder resnet34 \
  --img_size 512 \
  --dropout_p 0.3 \
  --train_mode full
```

### 2. Evaluate MC Dropout

Global deferral:

```bash
python evaluate.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --n_passes 30 \
  --output_dir results/mc_dropout \
  --deferral_mode global
```

Confidence-aware deferral:

```bash
python evaluate.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --n_passes 30 \
  --output_dir results/mc_dropout_conf_aware \
  --deferral_mode adaptive_confidence
```

Temperature calibration:

```bash
python evaluate.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --n_passes 30 \
  --output_dir results/mc_dropout_calibrated \
  --calibration temperature
```

### 3. Evaluate TTA

Global deferral:

```bash
python eval_tta.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --n_augmentations 4 \
  --output_dir results/tta \
  --deferral_mode global
```

Adaptive deferral:

```bash
python eval_tta.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --n_augmentations 4 \
  --output_dir results/tta_adaptive \
  --deferral_mode adaptive \
  --adaptive_percentile 75
```

Confidence-aware deferral with calibration:

```bash
python eval_tta.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --n_augmentations 4 \
  --output_dir results/tta_cal_conf \
  --calibration temperature \
  --deferral_mode adaptive_confidence
```

### 4. Run the full MC Dropout vs TTA comparison

```bash
python compare_uncertainty_methods.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --output_dir results/comparison_final \
  --n_mc_passes 20 \
  --n_augmentations 4
```

### 5. Cross-dataset evaluation

```bash
python experiments/cross_dataset.py \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --stare_dir data/STARE \
  --chase_dir data/CHASE_DB1 \
  --output_dir results/cross_dataset
```

### 6. Ablation studies

```bash
python experiments/ablation.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout_fullft/best_model.pth \
  --output_dir results/ablations
```

## Project Structure

```text
medical-seg-uncertainty/
├── train.py                        # Main training entry point
├── evaluate.py                     # MC Dropout evaluation, calibration, deferral
├── eval_tta.py                     # TTA evaluation, calibration, deferral
├── compare_uncertainty_methods.py  # Unified MC Dropout vs TTA comparison pipeline
├── train_ensemble.py               # Ensemble training
├── evaluate_ensemble.py            # Ensemble evaluation
├── configs/                        # Experiment configuration snapshots
├── data/                           # DRIVE, STARE, CHASE_DB1 loaders and assets
├── models/                         # U-Net variants, TTA wrapper, losses
├── utils/                          # Metrics, calibration, deferral, plotting, I/O
├── experiments/                    # Ablation, comparison, statistics, cross-dataset scripts
├── results/                        # Saved metrics, plots, summaries, paper artifacts
├── checkpoints/                    # Trained checkpoints
├── paper/                          # Current LaTeX paper source
└── medicalpaper/                   # Alternate manuscript variants and figures
```

### Important Result Folders

- `results/mc_dropout/`: MC Dropout evaluation outputs, reliability diagrams, risk-coverage curves, and deferral reports.
- `results/tta_adaptive/`: strongest adaptive-deferral result set.
- `results/tta_cal_conf/`: calibrated TTA with confidence-aware deferral.
- `results/comparison_final/comparison/`: consolidated comparison plots and summary tables.
- `results/cross_dataset/`: zero-shot transfer results on STARE and CHASE_DB1.
- `results/summaries/`: compact CSV and JSON tables used for reporting.
- `paper/`: the main manuscript currently prepared for publication and arXiv packaging.

## Reproducibility

To reproduce the main paper-style outputs:

1. Train the U-Net checkpoint or use the provided checkpoint directory layout.
2. Run `evaluate.py` for MC Dropout outputs.
3. Run `eval_tta.py` for TTA outputs under global, adaptive, and confidence-aware deferral.
4. Run `compare_uncertainty_methods.py` to generate the consolidated comparison plots and summary tables.
5. Run `experiments/cross_dataset.py` and `experiments/ablation.py` for transfer and sensitivity studies.
6. Use the generated outputs in `results/` and `paper/figures/` to rebuild the manuscript figures.

The most important summary artifacts already included in the repository are:

- `results/comparison_final/comparison/comparison_summary.csv`
- `results/summaries/drive_main_table.csv`
- `results/summaries/stats_report.json`
- `results/cross_dataset/cross_dataset_results.json`

## Paper

ArXiv: `TODO`

Current manuscript sources are stored under `paper/`.

## License

No license file is currently included in the repository. Add a project license before public redistribution or reuse.
