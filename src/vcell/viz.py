"""Shared matplotlib style.

Palette is the validated 3-slot categorical set (blue / orange / aqua): passes
lightness band, chroma floor, CVD separation and normal-vision floor under the
all-pairs list in light mode.  Aqua sits at 2.74:1 against the surface, below the
3:1 bar, so the relief rule applies -- every figure ships with direct labels and
a CSV table of the same numbers next to it in results/.

Figures are light-mode only on purpose: they are destined for a printed Word
submission, not a themable web page.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8c8b84"
GRID = "#e6e5e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (S1, S2, S3)


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.markersize": 5.5,
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def strip(ax, which=("top", "right", "left")) -> None:
    for s in which:
        ax.spines[s].set_visible(False)


def caption(fig, text: str, y: float = -0.02) -> None:
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8, color=MUTED, wrap=True)
