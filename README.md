# Rethinking Uncertainty in Segmentation: From Estimation to Decision

![arXiv](https://img.shields.io/badge/arXiv-2604.13262-B31B1B?style=flat-square&logo=arxiv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-0F766E?style=flat-square)
![Uncertainty](https://img.shields.io/badge/Uncertainty-MC%20Dropout%20%7C%20TTA-7C3AED?style=flat-square)
![Datasets](https://img.shields.io/badge/Datasets-DRIVE%20%7C%20STARE%20%7C%20CHASE__DB1-059669?style=flat-square)
![Best Result](https://img.shields.io/badge/Best-~80%25%20error%20reduction%20%40%2025%25%20defer-DC2626?style=flat-square)

> Uncertainty is only useful when it changes a decision.
> We reframe uncertainty in medical image segmentation as a **two-stage problem — estimation, then decision** — and show that optimising uncertainty metrics in isolation does not translate into safer downstream decisions.

This repository contains the full code, experiments, figures and LaTeX source accompanying our
arXiv preprint **[arXiv:2604.13262](https://arxiv.org/abs/2604.13262)**.

---

## Paper

- **Title:** *Rethinking Uncertainty in Segmentation: From Estimation to Decision*
- **Author:** Saket Maganti
- **arXiv:** [2604.13262](https://arxiv.org/abs/2604.13262) (2026)
- **Project page:** [`project_page/index.html`](project_page/index.html)

### Core contribution

We separate the pipeline into **two stages** and study each on its own terms:

1. **Estimation** — how to produce a useful pixel-level uncertainty map (MC Dropout vs Test-Time Augmentation vs deep ensembles).
2. **Decision** — how to translate that map into a concrete *accept / defer* outcome (global threshold, adaptive per-image, confidence-aware).

The central empirical finding: **calibration improvements do not correlate with better decision quality.** Temperature scaling can cut ECE by 6–8× while leaving the risk-coverage curve essentially unchanged. Evaluating uncertainty should therefore prioritise *real-world decision outcomes* (error-vs-coverage) rather than isolated uncertainty / calibration scores.

---

## Highlights

- **Two-stage framework** — uncertainty **estimation** separated from **deferral decision**, evaluated head-to-head.
- **Up to ~80 % error reduction** at 25 % pixel deferral (best method + policy combination).
- **79.5 % error reduction** on DRIVE with TTA + confidence-aware deferral (4.6 % → 0.9 % residual error at 25 % coverage).
- **TTA beats MC Dropout** on Dice, AUC, ECE and uncertainty-AUROC across all three retinal datasets.
- **Calibration ≠ decision quality** — 6–8× ECE reduction from temperature scaling leaves deferral curves essentially unchanged.
- **Confidence-aware deferral rule** — combines epistemic uncertainty with prediction margin `2|p − 0.5|`, outperforming global and adaptive baselines.
- **Cross-dataset consistency** on DRIVE, STARE and CHASE\_DB1 with a shared U-Net (ResNet-34) backbone.

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
not mirrored in this repository. The preprint is available on arXiv at
[arXiv:2604.13262](https://arxiv.org/abs/2604.13262).

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

## Citation

If you use this code or the findings in your work, please cite:

```bibtex
@article{maganti2026rethinking,
  title         = {Rethinking Uncertainty in Segmentation: From Estimation to Decision},
  author        = {Maganti, Saket},
  year          = {2026},
  eprint        = {2604.13262},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2604.13262}
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
