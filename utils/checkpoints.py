from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint}


def load_model_state(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    epoch: int,
    val_dice: float,
    args: dict[str, Any],
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_dice": float(val_dice),
        "args": args,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if extra:
        payload.update(extra)
    torch.save(payload, path)
