from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


DEFAULT_DPI = 320


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "figure.constrained_layout.use": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.labelpad": 10,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#D7DCE3",
            "legend.borderpad": 0.6,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 11,
            "lines.linewidth": 2.2,
            "axes.titlepad": 10,
        }
    )


def style_axes(ax: plt.Axes, *, xpad: int = 8, ypad: int = 8) -> None:
    ax.tick_params(axis="both", which="major", pad=6)
    ax.xaxis.labelpad = xpad
    ax.yaxis.labelpad = ypad


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = DEFAULT_DPI) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def add_adjusted_labels(
    ax: plt.Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    *,
    initial_offsets: Sequence[tuple[float, float]] | None = None,
    text_kwargs: dict | None = None,
    arrow_kwargs: dict | None = None,
    expand_points: tuple[float, float] = (1.3, 1.5),
    expand_text: tuple[float, float] = (1.15, 1.3),
    force_points: float = 0.4,
    force_text: float = 0.6,
) -> list:
    text_kwargs = {
        "fontsize": 9,
        "fontweight": "bold",
        "bbox": {
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#DDE3EA",
            "alpha": 0.92,
        },
        **(text_kwargs or {}),
    }
    text_ha = text_kwargs.pop("ha", "left")
    text_va = text_kwargs.pop("va", "bottom")
    arrow_kwargs = {
        "arrowstyle": "-",
        "color": "#6B7280",
        "lw": 0.8,
        "alpha": 0.85,
        **(arrow_kwargs or {}),
    }

    if initial_offsets is None:
        initial_offsets = [(0.0, 0.0)] * len(labels)
    texts = [
        ax.text(x + dx, y + dy, label, ha=text_ha, va=text_va, zorder=8, **text_kwargs)
        for x, y, label, (dx, dy) in zip(xs, ys, labels, initial_offsets)
    ]
    adjust_text(
        texts,
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        ax=ax,
        arrowprops=arrow_kwargs,
        expand_points=expand_points,
        expand_text=expand_text,
        force_points=force_points,
        force_text=force_text,
        ensure_inside_axes=True,
        avoid_self=True,
        prevent_crossings=True,
        only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
    )
    return texts
