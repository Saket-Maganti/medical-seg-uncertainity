import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path):
    if not path.exists():
        return None
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


def build_drive_table(results_root: str = "results/drive", out_path: str = "results/summaries/drive_main_table.csv"):
    root = Path(results_root)
    rows = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        payload = _load_json(method_dir / "results.json")
        if not payload:
            continue
        rows.append({"method": method_dir.name, **payload})
    _write_csv(Path(out_path), rows)
    return rows


def build_deferral_table(results_root: str = "results/drive", out_path: str = "results/summaries/deferral_operating_points.csv"):
    root = Path(results_root)
    rows = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        payload = _load_json(method_dir / "deferral_operating_points.json")
        if not payload:
            continue
        for label, point in payload.get("operating_points", {}).items():
            rows.append({"method": method_dir.name, "operating_point": label, **point})
    _write_csv(Path(out_path), rows)
    return rows


def build_runtime_table(results_root: str = "results/drive", out_path: str = "results/summaries/runtime_table.csv"):
    root = Path(results_root)
    rows = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        payload = _load_json(method_dir / "runtime.json")
        if not payload:
            continue
        rows.append({"method": method_dir.name, **payload})
    _write_csv(Path(out_path), rows)
    return rows


def build_cross_dataset_table(results_root: str = "results/cross_dataset", out_path: str = "results/summaries/cross_dataset_table.csv"):
    payload = _load_json(Path(results_root) / "cross_dataset_results.json")
    if not payload:
        return []
    rows = [{"dataset": dataset, **summary} for dataset, summary in payload.items()]
    _write_csv(Path(out_path), rows)
    return rows


def build_cross_dataset_decision_table(results_root: str = "results/cross_dataset", out_path: str = "results/summaries/cross_dataset_decision_table.csv"):
    payload = _load_json(Path(results_root) / "cross_dataset_full_results.json")
    if not payload:
        return []
    rows = []
    for dataset, result in payload.items():
        op = result.get("deferral_summary", {}).get("operating_points", {})
        selective = result.get("selective_summary", {})
        rows.append(
            {
                "dataset": dataset,
                "balanced_f1_coverage": op.get("balanced_f1", {}).get("coverage"),
                "balanced_f1_precision": op.get("balanced_f1", {}).get("precision"),
                "balanced_f1_recall": op.get("balanced_f1", {}).get("recall"),
                "balanced_f1_dice_acc": op.get("balanced_f1", {}).get("dice_accepted"),
                "high_recall_coverage": op.get("high_recall", {}).get("coverage"),
                "high_recall_dice_acc": op.get("high_recall", {}).get("dice_accepted"),
                "selective_aucc_dice": selective.get("aucc_dice"),
                "selective_aucc_auc": selective.get("aucc_auc"),
            }
        )
    _write_csv(Path(out_path), rows)
    return rows


def build_crossval_table(results_root: str = "results/crossval", out_path: str = "results/summaries/crossval_table.csv"):
    payload = _load_json(Path(results_root) / "crossval_summary.json")
    if not payload:
        return []
    rows = payload.get("folds", [])
    _write_csv(Path(out_path), rows)
    return rows


if __name__ == "__main__":
    build_drive_table()
    build_deferral_table()
    build_runtime_table()
    build_cross_dataset_table()
    build_cross_dataset_decision_table()
    build_crossval_table()
