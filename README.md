# Medical Segmentation Uncertainty

Retinal vessel segmentation research codebase with uncertainty estimation,
calibration analysis, selective prediction, deferral analysis, cross-dataset
evaluation, and ablation studies.

This public repository snapshot is intentionally curated.

It includes the code, lightweight result summaries, visual artifacts, and
experiment outputs that are useful for understanding the project.

It does not include every local file that existed in the original development
workspace.

That was a deliberate choice so the repository remains understandable,
cloneable, and pushable on GitHub.

This README is intentionally very detailed.

It is meant to act as:

- a project overview
- a reproducibility guide
- a map of the codebase
- a guide to the included results
- a statement of what was excluded from the public snapshot and why

## Table Of Contents

- [1. Project Summary](#1-project-summary)
- [2. Why This Repository Exists](#2-why-this-repository-exists)
- [3. What The Project Studies](#3-what-the-project-studies)
- [4. Main Questions Answered By The Repo](#4-main-questions-answered-by-the-repo)
- [5. Public Snapshot Policy](#5-public-snapshot-policy)
- [6. What Is Included In This Public Repo](#6-what-is-included-in-this-public-repo)
- [7. What Was Not Added To This Repo And Why](#7-what-was-not-added-to-this-repo-and-why)
- [8. High-Level Methodology](#8-high-level-methodology)
- [9. Baselines And Experiment Families](#9-baselines-and-experiment-families)
- [10. Repository Layout](#10-repository-layout)
- [11. File-By-File Source Guide](#11-file-by-file-source-guide)
- [12. Data Loader Source Files](#12-data-loader-source-files)
- [13. Environment And Dependencies](#13-environment-and-dependencies)
- [14. Installation](#14-installation)
- [15. Dataset Preparation](#15-dataset-preparation)
- [16. Expected Dataset Directory Layout](#16-expected-dataset-directory-layout)
- [17. Training Workflow](#17-training-workflow)
- [18. Evaluation Workflow](#18-evaluation-workflow)
- [19. Cross-Validation Workflow](#19-cross-validation-workflow)
- [20. Comparison Workflow](#20-comparison-workflow)
- [21. Cross-Dataset Workflow](#21-cross-dataset-workflow)
- [22. Ablation Workflow](#22-ablation-workflow)
- [23. Temperature Scaling And Reporting Scripts](#23-temperature-scaling-and-reporting-scripts)
- [24. Main Commands](#24-main-commands)
- [25. Results Included In The Repo](#25-results-included-in-the-repo)
- [26. How To Read The Results Folders](#26-how-to-read-the-results-folders)
- [27. Important Output Files](#27-important-output-files)
- [28. Included Images And Figures](#28-included-images-and-figures)
- [29. What The Raw Artifacts Usually Contain](#29-what-the-raw-artifacts-usually-contain)
- [30. Reproducibility Notes](#30-reproducibility-notes)
- [31. Known Limitations Of This Snapshot](#31-known-limitations-of-this-snapshot)
- [32. Known Limitations Of The Research Pipeline](#32-known-limitations-of-the-research-pipeline)
- [33. Practical Tips For Running The Code](#33-practical-tips-for-running-the-code)
- [34. Troubleshooting](#34-troubleshooting)
- [35. FAQ](#35-faq)
- [36. Suggested Next Steps](#36-suggested-next-steps)
- [37. Public Snapshot Inventory](#37-public-snapshot-inventory)
- [38. Closing Notes](#38-closing-notes)

## 1. Project Summary

This project studies medical image segmentation uncertainty in the context of
retinal vessel segmentation.

The central model family is a U-Net style segmentation model with MC Dropout.

The repository also contains deterministic, ensemble, and test-time
augmentation style evaluation baselines.

The goal is not only to obtain segmentation masks.

The larger goal is to understand when the model is uncertain, when that
uncertainty is meaningful, and whether uncertainty can improve downstream
decision-making.

In practical terms, the repository focuses on the following outcomes:

- strong segmentation metrics on DRIVE
- usable uncertainty maps
- calibration measurements
- reliability diagrams
- uncertainty-aware failure analysis
- deferral policies
- selective prediction curves
- cross-dataset transfer checks
- ablation studies on uncertainty-related design choices

This repository includes code and results.

The paper source files are intentionally not part of this public snapshot.

## 2. Why This Repository Exists

A lot of segmentation repositories stop at accuracy metrics.

That is usually not enough for medical use cases.

If a model produces a segmentation but cannot express confidence well, it is
hard to build safe workflows around it.

This repository exists to support a broader evaluation mindset.

Instead of asking only:

- how good is the segmentation?

It also asks:

- how calibrated are the probabilities?
- can uncertainty predict errors?
- can we reject or defer uncertain cases?
- do uncertainty signals still help on new datasets?
- which design choices improve reliability instead of only raw accuracy?

That broader framing is the point of the project.

## 3. What The Project Studies

The problem setting is binary vessel segmentation in retinal fundus images.

The main in-domain dataset is DRIVE.

The out-of-domain checks use STARE and CHASE_DB1.

The project studies:

- segmentation performance
- predictive uncertainty
- epistemic uncertainty approximated by MC Dropout
- deferral behavior
- selective prediction behavior
- calibration quality
- failure localization
- cross-dataset uncertainty behavior
- sensitivity to inference settings

## 4. Main Questions Answered By The Repo

This codebase is organized around several concrete research questions.

1. Can an MC Dropout U-Net produce strong vessel segmentation on DRIVE?
2. Is the uncertainty estimate useful for detecting model errors?
3. How does MC Dropout compare with deterministic, ensemble, and TTA baselines?
4. Can uncertainty guide deferral to a human reviewer?
5. Can uncertainty support selective prediction with better coverage-risk tradeoffs?
6. Does uncertainty remain informative when evaluated zero-shot on STARE and CHASE_DB1?
7. How sensitive are the conclusions to the number of MC passes?
8. How sensitive are the conclusions to dropout probability?
9. How sensitive are the conclusions to patch size and training setup?
10. Which results are lightweight enough to publish directly in the repository?

## 5. Public Snapshot Policy

The GitHub version of this repository is not a verbatim dump of the original
local development folder.

It is a cleaned public snapshot.

That matters because the original workspace contained:

- raw datasets
- local Python environments
- tracked checkpoints
- experiment tracking logs
- manuscript sources
- large intermediate artifacts
- cache files
- local editor files

Keeping all of that in the public repo would make the repository:

- extremely large
- hard to clone
- hard to push
- noisy to navigate
- less reproducible in practice

The public snapshot therefore follows a simple rule:

- include code that explains and runs the project
- include lightweight results that document the outputs
- exclude bulky or downloadable files that are not essential to version

## 6. What Is Included In This Public Repo

This public repo includes:

- training scripts
- evaluation scripts
- cross-validation scripts
- uncertainty comparison scripts
- cross-dataset evaluation scripts
- ablation scripts
- utility code for calibration, reliability, metrics, statistics, visualization,
  deferral, and selective prediction
- data loader source files
- configuration files
- result summaries
- result figures
- representative output JSON files
- representative output CSV files
- representative output PNG files

The repository is therefore useful for:

- reading the methodology
- understanding the experiment structure
- reviewing code organization
- reproducing experiments after downloading data
- examining result summaries without rerunning everything

## 7. What Was Not Added To This Repo And Why

This is the most important practical section in the README.

The following file groups were intentionally not added to the public GitHub
snapshot.

### 7.1 Raw datasets

Not added:

- `data/DRIVE/` image files
- `data/STARE/` image files
- `data/CHASE_DB1/` image files
- any other dataset dumps placed under `data/` or `datasets/`

Why:

- these datasets can be downloaded separately from their original sources
- some datasets have their own distribution terms
- raw medical image data inflates repo size quickly
- dataset binaries are not source code

Important nuance:

- the `data/*.py` loader source files are included
- the raw image files themselves are excluded

### 7.2 Model checkpoints

Not added:

- `checkpoints/**/*.pth`
- `checkpoints/**/*.pt`
- any local training checkpoints under similar folders
- heavyweight fold checkpoints that were stored inside results subfolders

Why:

- checkpoints are large
- checkpoints are generated outputs rather than source
- many checkpoints can be recreated by rerunning training
- keeping them in GitHub made the repository too large to push reliably

### 7.3 Raw array artifacts

Not added:

- `results/**/artifacts/**`
- `results/**/*.npy`

Why:

- these files are large intermediate arrays
- they mainly store raw prediction maps, uncertainty maps, error masks, or
  related tensors
- they are useful for deep analysis but not necessary for understanding the
  project from GitHub
- keeping them would significantly increase repository size

### 7.4 Intermediate training outputs inside results

Not added:

- `results/**/patch_train/**`
- `results/**/fullft_train/**`
- `results/**/*.pth` under `results/`

Why:

- these are effectively training checkpoints or training-stage outputs
- they are derivable by rerunning the corresponding experiments
- they add weight without improving readability of the public repository

### 7.5 Local Python environments

Not added:

- `venv/`
- `.venv/`
- `env/`

Why:

- local environments are machine-specific
- they are reproducible from `requirements.txt`
- they contain compiled binaries that should not be versioned

### 7.6 Experiment tracking logs

Not added:

- `wandb/`

Why:

- local W&B runs contain logs, metadata, and run artifacts
- they are large
- they are generated rather than authored
- if needed, tracking outputs should live in the tracking system itself rather
  than in Git history

### 7.7 Python caches

Not added:

- `__pycache__/`
- `*.pyc`
- tool caches

Why:

- these are derived automatically
- they are not source
- they create repository noise

### 7.8 Local operating system files

Not added:

- `.DS_Store`

Why:

- macOS creates these automatically
- they are never needed for reproducibility

### 7.9 Local tool metadata

Not added:

- `.claude/`
- other local editor or assistant folders if present

Why:

- these are workspace-specific
- they do not define the scientific pipeline
- they are not meant for public consumption

### 7.10 Paper sources

Not added:

- `paper/`
- manuscript `.tex` files
- manuscript section files
- paper-only figure directories

Why:

- the paper was explicitly left out of scope for this snapshot
- the request for this public repo was to add the repository contents and
  results, but not the paper yet

### 7.11 What was kept even though it is result-related

Kept:

- result JSON summaries
- result CSV tables
- result PNG figures
- paper-artifact PNG exports under `results/paper_artifacts/`
- figure-generation scripts

Why:

- these are lightweight enough to publish
- they make the repo immediately informative
- they show the outputs without requiring a full rerun

## 8. High-Level Methodology

The project uses a segmentation pipeline centered around a U-Net style model.

The main uncertainty method is MC Dropout.

At test time, dropout remains active and multiple stochastic forward passes are
performed.

The mean prediction is used as the segmentation probability map.

The variance across passes is used as an uncertainty estimate.

The repository also compares that approach against:

- deterministic inference
- deep ensemble style evaluation
- test-time augmentation

The evaluation does not stop at Dice or AUC.

It also computes:

- ECE
- reliability diagrams
- uncertainty AUROC for error prediction
- deferral operating points
- selective prediction curves
- failure analyses

## 9. Baselines And Experiment Families

The repository supports several main experiment families.

### 9.1 MC Dropout baseline

This is the central method in the project.

Training uses the `train.py` entry point.

Evaluation uses the `evaluate.py` entry point.

### 9.2 Deterministic baseline

A deterministic U-Net style baseline is supported.

This is useful to compare segmentation quality and calibration behavior against
the stochastic model.

### 9.3 Deep ensemble baseline

The repository supports evaluating multiple independently trained ensemble
members.

This provides a stronger uncertainty baseline than a single deterministic model.

### 9.4 Test-time augmentation baseline

TTA provides another way to induce predictive diversity at inference time.

It acts as a baseline for comparing uncertainty quality and downstream decision
behavior.

### 9.5 Cross-validation pipeline

The repository includes a more structured two-stage cross-validation workflow:

- patch warm-up
- full-image fine-tuning
- final MC Dropout evaluation

### 9.6 Cross-dataset generalization

The repository includes zero-shot evaluation on:

- STARE
- CHASE_DB1

### 9.7 Ablations

The repository includes ablations over:

- number of MC passes
- dropout probability
- patch size
- ensemble size outputs in results

## 10. Repository Layout

Top-level structure:

```text
.
├── configs/
├── data/
├── experiments/
├── models/
├── results/
├── utils/
├── eval_tta.py
├── evaluate.py
├── evaluate_ensemble.py
├── train.py
├── train_cv.py
├── train_ensemble.py
├── run_all.sh
├── requirements.txt
└── pyproject.toml
```

High-level meaning:

- `configs/` stores lightweight experiment configuration files
- `data/` stores dataset loader source code, not the raw downloaded datasets
- `experiments/` stores higher-level experiment orchestration and reporting
- `models/` stores architecture and loss definitions
- `results/` stores public result outputs included in the repo
- `utils/` stores metric, calibration, decision, reliability, and visualization
  utilities

## 11. File-By-File Source Guide

This section explains the main source files one by one.

### 11.1 Top-level scripts

- `train.py`: trains a single model and saves the best checkpoint
- `evaluate.py`: evaluates one trained model with MC Dropout or deterministic
  inference
- `evaluate_ensemble.py`: evaluates an ensemble of trained members
- `eval_tta.py`: evaluates the TTA baseline
- `train_ensemble.py`: trains multiple ensemble members with different seeds
- `train_cv.py`: runs the structured cross-validation pipeline
- `run_all.sh`: convenience shell script for running a larger workflow

### 11.2 Config files

- `configs/base.yaml`: common settings base
- `configs/drive_mc.yaml`: main MC Dropout run settings
- `configs/drive_deterministic.yaml`: deterministic evaluation configuration
- `configs/drive_ensemble.yaml`: ensemble evaluation configuration
- `configs/drive_tta.yaml`: TTA evaluation configuration
- `configs/cross_dataset.yaml`: STARE and CHASE_DB1 evaluation configuration
- `configs/ablations.yaml`: ablation parameter lists

### 11.3 Model files

- `models/unet_mc.py`: MC Dropout U-Net definition
- `models/deterministic_unet.py`: deterministic segmentation baseline
- `models/tta.py`: TTA wrapper logic
- `models/losses.py`: segmentation loss construction
- `models/edl.py`: evidential-style model component or reference module
- `models/__init__.py`: package initialization

### 11.4 Utility files

- `utils/metrics.py`: core segmentation and uncertainty metrics
- `utils/calibration.py`: calibration logic
- `utils/deferral.py`: deferral policy logic
- `utils/selective_prediction.py`: selective prediction analysis
- `utils/failure_analysis.py`: failure taxonomy support
- `utils/reliability_checks.py`: overconfidence and reliability analysis
- `utils/visualization.py`: artifact and figure helpers
- `utils/io.py`: JSON dumping and path helpers
- `utils/checkpoints.py`: checkpoint save and load helpers
- `utils/device.py`: CPU, CUDA, or MPS device selection
- `utils/mc_dropout.py`: repeated stochastic forward-pass logic
- `utils/stats.py`: statistical comparison helpers
- `utils/decision.py`: decision-oriented helper functions
- `utils/seed.py`: reproducibility seeding helpers

### 11.5 Experiment scripts

- `experiments/compare_methods.py`: compare MC Dropout, ensemble, and TTA
- `experiments/cross_dataset.py`: zero-shot evaluation on STARE and CHASE_DB1
- `experiments/ablation.py`: ablations for uncertainty-related design choices
- `experiments/temperature_scaling.py`: calibration refinement utilities
- `experiments/generate_figures.py`: build visual outputs from result files
- `experiments/generate_tables.py`: generate tabular summaries
- `experiments/run_stats.py`: compute statistical reports
- `experiments/generate_paper_assets.py`: generate figures from results even
  though the paper sources are not included yet

## 12. Data Loader Source Files

The repository includes the dataset loader code, which is important.

These files are source code and therefore belong in the public repo.

- `data/__init__.py`
- `data/dataset.py`
- `data/drive.py`
- `data/stare.py`
- `data/chase.py`
- `data/transforms.py`

What they do:

- define DRIVE patch and full-image datasets
- define fold splitting logic for DRIVE
- define STARE loader behavior
- define CHASE_DB1 loader behavior
- define train-time and eval-time transforms

What they do not include:

- the raw image data itself

## 13. Environment And Dependencies

The repository currently declares dependencies in `requirements.txt`.

Main packages:

- `torch`
- `torchvision`
- `segmentation-models-pytorch`
- `albumentations`
- `numpy`
- `Pillow`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `wandb`
- `tqdm`

Formatting and lint settings are in `pyproject.toml`.

That file currently contains tool settings for:

- `black`
- `ruff`

## 14. Installation

Minimal installation flow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional quality-of-life setup:

```bash
pip install black ruff
```

If you use Apple Silicon:

- the code already sets `PYTORCH_ENABLE_MPS_FALLBACK=1` in main scripts
- some channels-last behavior is disabled on MPS for stability in training

## 15. Dataset Preparation

The code assumes the user downloads the datasets separately.

That is intentional.

### 15.1 DRIVE

Used for:

- training
- validation
- test evaluation
- most in-domain quantitative analyses

### 15.2 STARE

Used for:

- cross-dataset generalization
- uncertainty shift analysis
- deferral and selective prediction on an unseen distribution

### 15.3 CHASE_DB1

Used for:

- another cross-dataset generalization test
- pediatric retinal images with a different data distribution

## 16. Expected Dataset Directory Layout

The public repo expects you to populate `data/` with downloaded datasets.

### 16.1 DRIVE expected layout

```text
data/DRIVE/
├── training/
│   ├── images/
│   ├── 1st_manual/
│   └── mask/
└── test/
    ├── images/
    ├── 1st_manual/
    └── mask/
```

### 16.2 STARE expected layout

```text
data/STARE/
├── images/
├── labels-ah/
└── labels-vk/   # optional
```

The loader also supports some mixed-folder dump layouts.

### 16.3 CHASE_DB1 expected layout

```text
data/CHASE_DB1/
├── Image_01L.jpg
├── Image_01L_1stHO.png
├── Image_01L_2ndHO.png
├── Image_01R.jpg
└── ...
```

## 17. Training Workflow

The main single-run training path is in `train.py`.

Training behavior includes:

- seed setup
- device selection
- W&B initialization
- DRIVE loader construction
- optional checkpoint resume
- configurable loss construction
- AdamW optimization
- ReduceLROnPlateau scheduling
- validation with threshold sweep
- early stopping
- best-model checkpoint saving

Notable design choices:

- threshold sweeping is done during validation
- Dice at the best validation threshold is tracked
- the best threshold is saved in the checkpoint metadata
- training supports patch mode and full-image mode

## 18. Evaluation Workflow

The main evaluation path is in `evaluate.py`.

Evaluation behavior includes:

- loading a trained model
- loading checkpoint metadata
- selecting threshold from checkpoint or CLI
- running MC Dropout inference or deterministic inference
- saving per-image artifacts
- computing summary metrics
- plotting a reliability diagram
- running deferral analysis
- checking overconfident failures
- saving JSON outputs

Artifacts saved by evaluation include:

- `results.json`
- `per_image_metrics.json`
- `per_image_uncertainty.json`
- `runtime.json`
- `deferral_operating_points.json`
- `overconfident_failures.json`
- `reliability_diagram.png`
- selected qualitative sample PNGs

## 19. Cross-Validation Workflow

The cross-validation logic in `train_cv.py` is more involved than a simple
single-stage train loop.

It performs:

1. fold splitting on DRIVE
2. patch warm-up training
3. full-image fine-tuning
4. final evaluation for each fold
5. aggregate summary creation
6. bootstrap confidence interval calculation

This matters because:

- it is a stronger evaluation structure than a single split
- it reflects the intended full pipeline more closely
- it generates fold-level summary reports

## 20. Comparison Workflow

`experiments/compare_methods.py` runs a unified comparison between uncertainty
methods.

Methods included in that comparison:

- MC Dropout
- Deep Ensemble
- TTA

For each method, it computes:

- aggregate segmentation metrics
- deferral summaries
- selective prediction summaries
- failure mode analysis

It also runs statistical comparison utilities and writes summary outputs.

## 21. Cross-Dataset Workflow

`experiments/cross_dataset.py` loads a trained MC Dropout model and evaluates it
zero-shot on STARE and CHASE_DB1.

It produces:

- dataset-wise summary metrics
- confidence intervals
- deferral summaries
- selective prediction summaries
- uncertainty shift plots
- cross-dataset comparison plots

This is one of the most important parts of the repo because it tests whether
uncertainty is still useful outside the training distribution.

## 22. Ablation Workflow

`experiments/ablation.py` studies sensitivity to several choices.

Ablations currently include:

- MC passes
- dropout probability
- patch size

This is useful because uncertainty methods often look good under a single
configuration but become brittle when the inference settings change.

The ablation outputs help show whether that is happening here.

## 23. Temperature Scaling And Reporting Scripts

The repository also includes helper scripts for reporting and post-processing.

Examples:

- `experiments/temperature_scaling.py`
- `experiments/generate_tables.py`
- `experiments/generate_figures.py`
- `experiments/run_stats.py`

These scripts help transform raw experiment outputs into:

- visual figures
- summary tables
- statistics reports
- publication-ready derived artifacts

Even though the paper itself is not included yet, these scripts remain useful.

## 24. Main Commands

Below are the most common commands.

### 24.1 Train the main MC Dropout model

```bash
python train.py --data_dir data/DRIVE --run_name unet_mc_dropout
```

### 24.2 Evaluate the main MC Dropout model

```bash
python evaluate.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout/best_model.pth \
  --n_passes 30 \
  --output_dir results/drive/mc_dropout_fullft
```

### 24.3 Evaluate a deterministic model

```bash
python evaluate.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_deterministic_fullft/best_model.pth \
  --deterministic \
  --output_dir results/drive/deterministic_fullft
```

### 24.4 Train ensemble members

```bash
python train_ensemble.py
```

### 24.5 Evaluate ensemble members

```bash
python evaluate_ensemble.py \
  --data_dir data/DRIVE \
  --checkpoint_dir checkpoints \
  --output_dir results/drive/ensemble
```

### 24.6 Evaluate TTA

```bash
python eval_tta.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout/best_model.pth \
  --output_dir results/drive/tta
```

### 24.7 Run cross-validation

```bash
python train_cv.py --data_dir data/DRIVE --output_dir results/crossval
```

### 24.8 Run cross-dataset evaluation

```bash
python experiments/cross_dataset.py \
  --checkpoint checkpoints/unet_mc_dropout/best_model.pth \
  --stare_dir data/STARE \
  --chase_dir data/CHASE_DB1 \
  --output_dir results/cross_dataset
```

### 24.9 Run ablations

```bash
python experiments/ablation.py \
  --data_dir data/DRIVE \
  --checkpoint checkpoints/unet_mc_dropout/best_model.pth \
  --output_dir results/ablations
```

### 24.10 Generate figures

```bash
python experiments/generate_figures.py
```

### 24.11 Generate tables

```bash
python experiments/generate_tables.py
```

## 25. Results Included In The Repo

The public snapshot includes many result outputs because they are useful and
still lightweight enough for GitHub.

Included result families:

- DRIVE benchmark outputs
- deterministic result outputs
- MC Dropout result outputs
- ensemble result outputs
- cross-dataset outputs
- cross-validation outputs
- ablation outputs
- summary CSV tables
- summary JSON reports
- representative PNG figures
- exported figure artifacts for future paper integration

Excluded result families:

- raw `.npy` artifacts
- heavyweight intermediate training outputs
- stored checkpoints inside result folders

## 26. How To Read The Results Folders

### 26.1 `results/drive/`

Contains in-domain DRIVE evaluation results.

Subfolders include:

- `mc_dropout_fullft/`
- `deterministic_fullft/`
- `ensemble/`

These folders generally contain:

- `config.json`
- `results.json`
- `per_image_metrics.json`
- `runtime.json`
- reliability outputs
- deferral outputs
- selected visual samples

### 26.2 `results/cross_dataset/`

Contains zero-shot transfer results on STARE and CHASE_DB1.

This folder is useful for understanding:

- generalization gaps
- uncertainty behavior on new data
- deferral performance outside DRIVE

### 26.3 `results/crossval/`

Contains fold-wise evaluation outputs and aggregate summaries.

### 26.4 `results/ablations/`

Contains results for sensitivity studies.

### 26.5 `results/summaries/`

Contains compact reporting-oriented files.

These are typically the fastest place to look if you want a summary rather than
the raw experiment folders.

### 26.6 `results/paper_artifacts/`

Contains exported figures derived from experiment outputs.

The paper source is not included, but the figure assets are.

## 27. Important Output Files

This section highlights the most useful files for quick inspection.

### 27.1 Main summary tables

- `results/summaries/drive_main_table.csv`
- `results/summaries/cross_dataset_table.csv`
- `results/summaries/crossval_table.csv`
- `results/summaries/runtime_table.csv`
- `results/summaries/deferral_operating_points.csv`
- `results/summaries/stats_table.csv`
- `results/summaries/bootstrap_table.csv`

### 27.2 Main summary JSON files

- `results/cross_dataset/cross_dataset_results.json`
- `results/cross_dataset/cross_dataset_full_results.json`
- `results/crossval/crossval_summary.json`
- `results/drive/mc_dropout_fullft/results.json`
- `results/drive/deterministic_fullft/results.json`
- `results/drive/ensemble/results.json`

### 27.3 Main figures

- `results/paper_artifacts/drive_method_tradeoff.png`
- `results/paper_artifacts/drive_qualitative_panel.png`
- `results/paper_artifacts/drive_per_image_uncertainty_scatter.png`
- `results/paper_artifacts/mc_dropout_fullft_reliability_diagram.png`
- `results/paper_artifacts/deterministic_fullft_reliability_diagram.png`
- `results/paper_artifacts/temperature_scaling_calibration_comparison.png`
- `results/paper_artifacts/cross_dataset_cross_dataset.png`
- `results/paper_artifacts/cross_dataset_uncertainty_shift.png`
- `results/paper_artifacts/crossval_fold_summary.png`
- `results/paper_artifacts/ablation_summary.png`

## 28. Included Images And Figures

The repository intentionally includes the result images because they make the
project much easier to review from GitHub.

Included image categories:

- deferral curves
- reliability diagrams
- qualitative segmentation examples
- risk-coverage curves
- cross-dataset comparison figures
- uncertainty shift plots
- summary paper-artifact figures

Why keep these:

- they are compact
- they communicate the work quickly
- they do not bloat the repo in the same way as raw checkpoints or arrays

## 29. What The Raw Artifacts Usually Contain

This section explains what was excluded so future readers understand the public
snapshot boundaries.

The removed raw artifact folders generally contain:

- `pred_mean.npy`
- `uncertainty.npy`
- `error_mask.npy`
- `gt_mask.npy`

These are useful if you want to:

- rerender custom visualizations
- perform a new downstream quantitative analysis
- inspect full-resolution stored arrays

They were excluded because:

- they are bulky
- they are intermediate outputs
- they are not needed for most readers

## 30. Reproducibility Notes

This public snapshot is meant to be reproducible with additional downloads.

To reproduce the main workflow, a user will typically need:

- Python 3.11
- dependencies from `requirements.txt`
- the DRIVE dataset
- optionally STARE
- optionally CHASE_DB1

To fully reproduce training-based outputs, a user will also need:

- enough disk space for checkpoints
- enough time for model training
- a working PyTorch device setup

Expected reproducibility boundaries:

- source code reproducibility: yes
- lightweight results review: yes
- exact local environment reproduction: no
- bundled raw data reproduction: no
- bundled checkpoint reproduction: no

## 31. Known Limitations Of This Snapshot

This public snapshot is intentionally useful, but it is not identical to the
full local development folder.

Limitations include:

- raw datasets are not bundled
- trained checkpoints are not bundled
- raw per-pixel `.npy` artifacts are not bundled
- manuscript sources are not bundled
- local W&B runs are not bundled

This means:

- the repo is lighter
- the repo is easier to clone
- the repo is easier to browse
- reproducing every local output requires rerunning experiments

## 32. Known Limitations Of The Research Pipeline

These are limitations of the project itself, not only the public snapshot.

- retinal vessel segmentation is a narrow medical imaging task
- zero-shot cross-dataset transfer remains challenging
- uncertainty quality can depend strongly on thresholding and evaluation setup
- some calibration and uncertainty conclusions may shift with architecture choice
- MC Dropout is a practical approximation, not a perfect Bayesian solution
- the repository currently focuses on binary vessel segmentation, not broader
  multi-class medical segmentation settings

## 33. Practical Tips For Running The Code

- start with reading `results/README.md`
- inspect `results/summaries/*.csv` before diving into raw folders
- use `evaluate.py` first if you want to understand the output structure
- use `train_cv.py` if you want a more rigorous evaluation than a single split
- use the included figures to sanity-check whether your rerun resembles the
  published outputs
- download datasets before trying to run any training or evaluation commands
- do not expect the repo to work if `data/*.py` exists but the raw dataset files
  are absent

## 34. Troubleshooting

### 34.1 Import error for `data.dataset`

Possible cause:

- you are on an older clone that predates the cleaned public update

Fix:

- pull the latest version of the repository

### 34.2 File-not-found errors under `data/DRIVE`

Possible cause:

- the DRIVE dataset has not been downloaded or placed in the expected layout

Fix:

- download the dataset and match the folder structure shown above

### 34.3 Cross-dataset script cannot find STARE or CHASE_DB1

Possible cause:

- those datasets were intentionally not bundled in the repo

Fix:

- download them separately and pass correct `--stare_dir` and `--chase_dir`

### 34.4 No checkpoints found

Possible cause:

- checkpoints are intentionally not versioned here

Fix:

- train models first or provide your own checkpoint paths

### 34.5 GitHub repo does not contain your huge local folders

Possible cause:

- that was intentional for the public snapshot

Fix:

- read the exclusion section in this README

### 34.6 W&B warnings or disabled runs

Possible cause:

- W&B is optional for understanding the public repo

Fix:

- configure W&B if you want tracking, or modify the scripts locally if you want
  offline-only usage

## 35. FAQ

### Q1. Why are datasets not included?

Because raw datasets are downloadable separately, often have their own
distribution terms, and would make the repo much larger.

### Q2. Why are checkpoints not included?

Because they are large generated binaries and were the main reason the original
repo was too heavy to push cleanly.

### Q3. Why are result images included but checkpoints excluded?

Because images are compact and informative, while checkpoints are large and not
useful for quick browsing.

### Q4. Why are some JSON files included?

Because they document the experiment outputs in a lightweight form and are very
useful for review and reproduction.

### Q5. Why are `.npy` artifacts excluded?

Because they are raw intermediate arrays and add a lot of storage overhead.

### Q6. Why is the paper missing?

Because the public snapshot was explicitly requested without the paper for now.

### Q7. Is this repo still useful without the paper?

Yes.

It includes the code and results needed to understand the project structure and
main outputs.

### Q8. Is this repo still useful without checkpoints?

Yes, for reading and understanding.

For exact reruns, you will need to retrain or provide your own checkpoints.

### Q9. Is the data loader code included?

Yes.

The loader source files under `data/` are included.

Only the raw dataset binaries are omitted.

### Q10. Are all results included?

No.

A curated set of useful result outputs is included.

Bulky intermediates and training-stage artifacts are excluded.

### Q11. Why is this README so long?

Because the repository is a curated public snapshot, and the README needs to do
some of the work that a paper or internal notes would normally do.

## 36. Suggested Next Steps

If you are a reader:

- start with `results/summaries/`
- inspect `results/paper_artifacts/`
- skim `evaluate.py` and `train_cv.py`

If you are trying to reproduce the pipeline:

- install dependencies
- download DRIVE
- run a single training experiment
- run one evaluation
- compare your outputs against the provided summary files

If you want to extend the research:

- add new uncertainty methods
- add more cross-dataset settings
- add stronger calibration baselines
- test other architectures
- add more detailed statistical comparisons

If you want to publish the paper later:

- keep using `results/paper_artifacts/`
- add manuscript sources in a later revision

## 37. Public Snapshot Inventory

This section is intentionally verbose.

It gives a quick one-line inventory of the public repo contents.

### 37.1 Top level

- `.gitignore`: ignores bulky and local-only files from future commits
- `README.md`: this long-form project guide
- `requirements.txt`: runtime dependencies
- `pyproject.toml`: formatting and lint settings
- `train.py`: main training loop
- `train_ensemble.py`: ensemble training entry point
- `train_cv.py`: cross-validation training and evaluation pipeline
- `evaluate.py`: main evaluation entry point
- `evaluate_ensemble.py`: ensemble evaluation entry point
- `eval_tta.py`: TTA baseline entry point
- `run_all.sh`: shell helper for larger experiment flows
- `unet_mc.py`: legacy top-level file retained from the original workspace

### 37.2 `configs/`

- `configs/base.yaml`: base config inheritance root
- `configs/drive_mc.yaml`: main MC Dropout config
- `configs/drive_deterministic.yaml`: deterministic baseline config
- `configs/drive_ensemble.yaml`: ensemble config
- `configs/drive_tta.yaml`: TTA config
- `configs/cross_dataset.yaml`: cross-dataset config
- `configs/ablations.yaml`: ablation parameter list config

### 37.3 `data/`

- `data/__init__.py`: package marker
- `data/dataset.py`: top-level dataloader construction wrapper
- `data/drive.py`: DRIVE dataset classes and fold splitting
- `data/stare.py`: STARE loader
- `data/chase.py`: CHASE_DB1 loader
- `data/transforms.py`: augmentation and normalization definitions

### 37.4 `models/`

- `models/__init__.py`: package marker
- `models/unet_mc.py`: MC Dropout model definition
- `models/deterministic_unet.py`: deterministic baseline definition
- `models/tta.py`: TTA wrapper logic
- `models/losses.py`: loss builder
- `models/edl.py`: evidential-style helper module

### 37.5 `utils/`

- `utils/__init__.py`: package marker
- `utils/calibration.py`: calibration helpers
- `utils/checkpoints.py`: checkpoint I/O helpers
- `utils/decision.py`: decision support helpers
- `utils/deferral.py`: deferral policy logic
- `utils/device.py`: device selection helpers
- `utils/failure_analysis.py`: failure mode analysis helpers
- `utils/io.py`: path and JSON helpers
- `utils/mc_dropout.py`: MC Dropout repeated inference helper
- `utils/metrics.py`: segmentation and uncertainty metrics
- `utils/reliability_checks.py`: overconfidence checks
- `utils/seed.py`: seed helpers
- `utils/selective_prediction.py`: selective prediction logic
- `utils/stats.py`: bootstrap and statistical comparison utilities
- `utils/visualization.py`: visualization helpers

### 37.6 `experiments/`

- `experiments/ablation.py`: ablation study runner
- `experiments/compare_methods.py`: unified baseline comparison
- `experiments/cross_dataset.py`: cross-dataset evaluation
- `experiments/generate_figures.py`: render figures from result outputs
- `experiments/generate_tables.py`: generate tables from result outputs
- `experiments/run_stats.py`: run statistical summaries
- `experiments/temperature_scaling.py`: temperature scaling utilities
- `experiments/generate_paper_assets.py`: figure export helper retained for
  future manuscript work

### 37.7 `results/`

- `results/README.md`: compact guide to result folders
- `results/drive/`: main benchmark outputs
- `results/cross_dataset/`: zero-shot transfer outputs
- `results/crossval/`: fold-wise outputs and summary
- `results/ablations/`: sensitivity study outputs
- `results/summaries/`: concise summary tables and reports
- `results/paper_artifacts/`: exported figure set

### 37.8 Example result files worth opening first

- `results/summaries/drive_main_table.csv`
- `results/summaries/cross_dataset_table.csv`
- `results/summaries/runtime_table.csv`
- `results/cross_dataset/cross_dataset_results.json`
- `results/crossval/crossval_summary.json`
- `results/drive/mc_dropout_fullft/results.json`
- `results/drive/deterministic_fullft/results.json`
- `results/paper_artifacts/drive_method_tradeoff.png`
- `results/paper_artifacts/drive_qualitative_panel.png`
- `results/paper_artifacts/crossval_fold_summary.png`

### 37.9 Excluded file groups worth knowing about

- raw datasets under `data/DRIVE`, `data/STARE`, `data/CHASE_DB1`
- checkpoints under `checkpoints/`
- local environments under `venv/` and `.venv/`
- tracking logs under `wandb/`
- raw artifacts under `results/**/artifacts/`
- raw arrays under `results/**/*.npy`
- training-stage outputs under `results/**/patch_train/` and
  `results/**/fullft_train/`
- manuscript sources under `paper/`
- local metadata such as `.DS_Store`

## 38. Closing Notes

This repository is meant to be a usable public research snapshot.

It is not meant to be a complete mirror of every file that ever existed in the
local workspace.

That is intentional.

The goal was to strike a balance between:

- completeness
- clarity
- repository size
- GitHub usability
- reproducibility

The current snapshot keeps:

- the code
- the loader source files
- the experiment scripts
- the useful results
- the useful images
- the lightweight summaries

And it excludes:

- downloadable data
- huge checkpoints
- raw bulky arrays
- local caches
- manuscript sources for now

If you are reading this on GitHub and something seems missing, check the
section titled `What Was Not Added To This Repo And Why`.

That section is the canonical explanation for the cleaned public snapshot.

If the paper is added later, this README can be shortened.

For now, the long-form documentation is intentional and should make the public
repository easier to understand without any additional context.
