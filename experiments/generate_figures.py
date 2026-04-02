from pathlib import Path
import shutil


def collect_figures(results_root: str = "results", out_dir: str = "results/paper_artifacts"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    patterns = [
        "reliability_diagram.png",
        "deferral_pr_curve.png",
        "risk_coverage.png",
        "selective_coverage.png",
        "ablation_results.png",
        "cross_dataset.png",
        "uncertainty_shift.png",
        "failure_taxonomy.png",
        "calibration_comparison.png",
    ]
    for pattern in patterns:
        for path in Path(results_root).rglob(pattern):
            target = out / f"{path.parent.name}_{path.name}"
            shutil.copy2(path, target)


if __name__ == "__main__":
    collect_figures()
