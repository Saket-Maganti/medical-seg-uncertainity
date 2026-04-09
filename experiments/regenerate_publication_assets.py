"""Regenerate publication and website assets from existing result artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.generate_paper_figures import OUT_DIR as PAPER_FIG_DIR
from paper.generate_paper_figures import copy_existing_assets, make_error_reduction_bars, make_pipeline, make_summary_scatter
from utils.comparison_plots import (
    plot_deferral_policy_operating_points,
    plot_method_summary_bars,
    plot_risk_coverage_comparison,
)

COMPARISON_DIR = REPO_ROOT / "results" / "comparison_final" / "comparison"
COMPARISON_JSON = COMPARISON_DIR / "comparison_summary.json"
PROJECT_ASSETS = REPO_ROOT / "project_page" / "assets"
ARXIV_FIGURES = REPO_ROOT / "arxiv_submission" / "figures"


def _load_summary() -> dict:
    with open(COMPARISON_JSON) as f:
        return json.load(f)


def _rc_csv(label: str) -> str:
    return str(REPO_ROOT / "results" / "comparison_final" / label / "risk_coverage" / "risk_coverage.csv")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    summary = _load_summary()

    plot_risk_coverage_comparison(
        methods={
            "MC Dropout": _rc_csv("mc_dropout"),
            "TTA": _rc_csv("tta"),
        },
        output_dir=str(COMPARISON_DIR),
        title="MC Dropout vs TTA — Risk-Coverage (global, no calibration)",
    )

    plot_method_summary_bars(
        results=summary,
        metrics=[
            "dice",
            "auc",
            "global_ece",
            "global_unc_auroc",
            "aucc_dice",
            "adt_error_before",
            "adt_error_after",
            "adt_pct_deferred",
        ],
        output_dir=str(COMPARISON_DIR),
        title="Method × Deferral Mode × Calibration Comparison",
    )

    plot_deferral_policy_operating_points(summary, str(COMPARISON_DIR), method_name="MC Dropout")
    plot_deferral_policy_operating_points(summary, str(COMPARISON_DIR), method_name="TTA")

    make_pipeline()
    make_summary_scatter()
    make_error_reduction_bars()
    copy_existing_assets()

    copies = [
        (PAPER_FIG_DIR / "pipeline.png", PROJECT_ASSETS / "pipeline.png"),
        (PAPER_FIG_DIR / "summary_scatter.png", PROJECT_ASSETS / "summary_scatter.png"),
        (COMPARISON_DIR / "risk_coverage_comparison.png", PROJECT_ASSETS / "risk_coverage.png"),
        (COMPARISON_DIR / "method_comparison_bars.png", PROJECT_ASSETS / "method_comparison_bars.png"),
        (COMPARISON_DIR / "qualitative_comparison.png", PROJECT_ASSETS / "qualitative_comparison.png"),
        (COMPARISON_DIR / "deferral_3mode_tta.png", PROJECT_ASSETS / "deferral_3mode_tta.png"),
        (PAPER_FIG_DIR / "pipeline.png", ARXIV_FIGURES / "pipeline.png"),
        (PAPER_FIG_DIR / "summary_scatter.png", ARXIV_FIGURES / "summary_scatter.png"),
        (COMPARISON_DIR / "risk_coverage_comparison.png", ARXIV_FIGURES / "risk_coverage.png"),
        (PAPER_FIG_DIR / "error_reduction_bars.png", ARXIV_FIGURES / "error_reduction_bars.png"),
    ]
    for src, dst in copies:
        if src.exists():
            _copy(src, dst)
            print(f"Copied {src.name} -> {dst}")


if __name__ == "__main__":
    main()
