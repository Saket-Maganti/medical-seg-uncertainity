from pathlib import Path

import numpy as np


def save_prediction_artifacts(
    artifact_dir: str | Path,
    image_id: str,
    pred_mean: np.ndarray,
    uncertainty: np.ndarray,
    error_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict:
    artifact_dir = Path(artifact_dir) / image_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pred_path = artifact_dir / "pred_mean.npy"
    unc_path = artifact_dir / "uncertainty.npy"
    err_path = artifact_dir / "error_mask.npy"
    gt_path = artifact_dir / "gt_mask.npy"

    np.save(pred_path, pred_mean.astype(np.float32))
    np.save(unc_path, uncertainty.astype(np.float32))
    np.save(err_path, error_mask.astype(np.float32))
    np.save(gt_path, gt_mask.astype(np.float32))

    return {
        "pred_mean": str(pred_path),
        "uncertainty": str(unc_path),
        "error_mask": str(err_path),
        "gt_mask": str(gt_path),
    }
