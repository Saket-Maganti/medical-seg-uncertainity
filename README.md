# From Prediction to Decision: Uncertainty-Aware Deferral for Reliable Medical Image Segmentation

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-0F766E?style=flat-square)
![Uncertainty](https://img.shields.io/badge/Uncertainty-MC%20Dropout%20%7C%20TTA-7C3AED?style=flat-square)
![Datasets](https://img.shields.io/badge/Datasets-DRIVE%20%7C%20STARE%20%7C%20CHASE__DB1-059669?style=flat-square)
![Best Result](https://img.shields.io/badge/Best-79.5%25%20error%20reduction%20%40%2025%25%20defer-DC2626?style=flat-square)

> Reliable segmentation is not only about predicting masks well.
> It is also about knowing **when to decide automatically** and **when to defer for human review**.

This repository contains the full code, experiments, figures, and LaTeX source for our study on
uncertainty-aware deferral in retinal vessel segmentation. It accompanies a preprint prepared
for arXiv.

---

## Highlights

- **Up to 79.5% error reduction** at 25% deferral coverage on DRIVE.
- **TTA beats MC Dropout** for uncertainty estimation across all three retinal datasets.
- **Three deferral policies** compared head-to-head: global threshold, adaptive per-image, and confidence-aware.
- **Temperature scaling** reduces ECE by 6–8× with no accuracy loss.
- **Cross-dataset evaluation** on DRIVE, STARE, and CHASE\_DB1 with a shared U-Net (ResNet-34) backbone.

---

## Repository Layout

```
.
├── project_page/          # Static project webpage
├── src/                   # Training, evaluation, uncertainty, deferral code
├── configs/               # YAML configs for each experiment
├── checkpoints/           # Saved model weights (local, gitignored)
├── data/                  # DRIVE / STARE / CHASE_DB1 (local, gitignored)
├── results/               # Metrics, plots, risk-coverage curves
└── README.md
```

The LaTeX manuscript and compiled PDF are maintained separately and are
not mirrored in this repository. The preprint will be posted to arXiv;
the link will be added here once available.

---

## Method in One Picture

```
  Input image ──► U-Net (ResNet-34) ──► T stochastic passes
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
                 Mean prediction                 Uncertainty map
                 (soft mask)                    (entropy / MI / var)
                      │                                 │
                      └───────────────┬─────────────────┘
                                      ▼
                              Deferral policy
                     (global τ | adaptive τₙ | conf-aware)
                                      ▼
                     Accept automatically  OR  Defer to expert
```

Two uncertainty sources are compared:

- **MC Dropout** — `T` stochastic forward passes with dropout active at test time.
- **Test-Time Augmentation (TTA)** — `K` geometric augmentations, predictions inverted and averaged.

Three deferral policies decide which pixels (or images) to route to a human:

1. **Global threshold** — single τ on uncertainty.
2. **Adaptive** — per-image τₙ calibrated to match a target coverage.
3. **Confidence-aware** — scores combine uncertainty with prediction margin `2|p − 0.5|`.

---

## Key Results

### Main comparison (DRIVE test set)

| Method            | Dice ↑ | AUC ↑ | ECE ↓  | Unc-AUROC ↑ |
|-------------------|:------:|:-----:|:------:|:-----------:|
| Deterministic     | 0.736  | 0.960 | 0.245  | 0.500       |
| MC Dropout        | 0.764  | 0.981 | 0.038  | 0.842       |
| **TTA**           | **0.771** | **0.984** | **0.031** | **0.867** |
| Deep Ensemble (3) | 0.769  | 0.983 | 0.035  | 0.859       |

### Deferral (DRIVE, 25% coverage deferred)

| Policy             | Residual error ↓ | Error reduction |
|--------------------|:----------------:|:---------------:|
| None (baseline)    | 4.6%             | —               |
| Global             | 1.4%             | 69.6%           |
| Adaptive           | 1.1%             | 76.1%           |
| **Confidence-aware** | **0.9%**       | **79.5%**       |

Full tables, risk-coverage curves, and cross-dataset numbers are in the
preprint (§5).

---

## Quick Start

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.11, PyTorch 2.x, CUDA 12.x (CPU works but is slow).

### 2. Data

Download DRIVE, STARE, and CHASE\_DB1 from their official sites and place them under
`data/` following the structure in `configs/data.yaml`. The datasets are not redistributed here.

### 3. Train

```bash
python -m src.train --config configs/drive_mc_dropout.yaml
python -m src.train --config configs/drive_tta.yaml
```

### 4. Evaluate with uncertainty and deferral

```bash
python -m src.evaluate        --config configs/eval_drive.yaml
python -m src.evaluate_deferral --config configs/deferral_drive.yaml
```

Outputs (metrics, plots, masks) are written to `results/`.

---

## Reproducing the Paper

```bash
bash scripts/reproduce_all.sh
```

This trains the deterministic, MC Dropout, TTA, and ensemble variants; runs cross-dataset
evaluation; generates risk-coverage curves; and saves every figure under `results/`.
Expect multiple GPU-hours on a single A100 for the full sweep.

---

## Datasets

| Dataset    | Images | Resolution | Split (train / val / test) |
|------------|:------:|:----------:|:--------------------------:|
| DRIVE      | 40     | 565×584    | 20 / — / 20                |
| STARE      | 20     | 700×605    | leave-one-out (cross-eval) |
| CHASE\_DB1 | 28     | 999×960    | 20 / 4 / 4                 |

All three are publicly available for research use from their respective maintainers.

---

## Paper

- Title: *From Prediction to Decision: Uncertainty-Aware Deferral for Reliable Medical Image Segmentation*
- Website: `project_page/index.html`
- Preprint: coming soon on arXiv — link will be added here.

The LaTeX source and compiled PDF are maintained outside this repository.

---

## Citation

If you use this code or the findings in your work, please cite:

```bibtex
@article{maganti2026deferral,
  title   = {From Prediction to Decision: Uncertainty-Aware Deferral for Reliable Medical Image Segmentation},
  author  = {Maganti, Saket},
  year    = {2026},
  note    = {Preprint}
}
```

---

## Acknowledgements

Built on [PyTorch](https://pytorch.org/) and
[segmentation\_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).
Thanks to the maintainers of DRIVE, STARE, and CHASE\_DB1 for keeping the datasets available
to the research community.

---

## Contact

Questions, issues, or ideas? Open a GitHub issue or reach out to the maintainer at
[Saket-Maganti](https://github.com/Saket-Maganti).
