import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.io import dump_json, ensure_dir
from utils.stats import bootstrap_ci_dict, save_stats_report, significance_table, wilcoxon_comparison


def _load_json(path: Path):
    import json

    with path.open() as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    output_dir = ensure_dir(args.output_dir)
    metrics_keys = ["dice", "auc", "ece", "unc_auroc"]

    method_paths = {
        "deterministic": Path(args.deterministic_metrics),
        "mc_dropout": Path(args.mc_metrics),
        "ensemble": Path(args.ensemble_metrics),
    }
    per_image = {name: _load_json(path) for name, path in method_paths.items()}

    bootstrap = {
        name: bootstrap_ci_dict(metrics, metrics_keys, n_bootstrap=args.n_bootstrap)
        for name, metrics in per_image.items()
    }

    significance_table(
        {
            "Deterministic": per_image["deterministic"],
            "MC Dropout": per_image["mc_dropout"],
            "Ensemble": per_image["ensemble"],
        },
        keys=metrics_keys,
        n_bootstrap=args.n_bootstrap,
    )

    comparisons = {
        "mc_vs_deterministic": wilcoxon_comparison(
            per_image["mc_dropout"], per_image["deterministic"], metrics_keys, "MC Dropout", "Deterministic"
        ),
        "mc_vs_ensemble": wilcoxon_comparison(
            per_image["mc_dropout"], per_image["ensemble"], metrics_keys, "MC Dropout", "Ensemble"
        ),
    }

    report = {
        "bootstrap": bootstrap,
        "comparisons": comparisons,
        "paths": {k: str(v) for k, v in method_paths.items()},
    }
    save_stats_report(report, str(output_dir / "stats_report.json"))

    rows = []
    for comparison_name, metrics in comparisons.items():
        for metric_name, payload in metrics.items():
            rows.append(
                {
                    "comparison": comparison_name,
                    "metric": metric_name,
                    "mean_a": payload["mean_a"],
                    "mean_b": payload["mean_b"],
                    "pval": payload["pval"],
                    "sig": payload["sig"],
                    "winner": payload["winner"],
                }
            )
    _write_csv(output_dir / "stats_table.csv", rows)

    bootstrap_rows = []
    for method_name, metrics in bootstrap.items():
        for metric_name, payload in metrics.items():
            bootstrap_rows.append({"method": method_name, "metric": metric_name, **payload})
    _write_csv(output_dir / "bootstrap_table.csv", bootstrap_rows)

    dump_json(output_dir / "stats_inputs.json", {k: str(v) for k, v in method_paths.items()})
    print(f"Saved stats outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deterministic-metrics",
        type=str,
        default="results/drive/deterministic_fullft/per_image_metrics.json",
    )
    parser.add_argument(
        "--mc-metrics",
        type=str,
        default="results/drive/mc_dropout_fullft/per_image_metrics.json",
    )
    parser.add_argument(
        "--ensemble-metrics",
        type=str,
        default="results/drive/ensemble/per_image_metrics.json",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/summaries"))
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()
    main(args)
