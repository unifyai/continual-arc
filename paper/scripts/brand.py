"""unify's brand tokens for the figures: paper, ink and one green, the green for one highlighted series per figure;
rules, not fills; no shadows, gradients or icons; Instrument Sans for labels, IBM Plex Mono for numerals and meta,
Newsreader italic for caption-like text. The fonts are in paper/fonts/, under the SIL Open Font License.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

PAPER = "#F4F2EC"
PAPER2 = "#EAE8E1"
INK = "#111111"
GRAPHITE = "#2A2A28"
GREEN = "#1F6B46"
GREEN2 = "#185639"
STONE = "#6B6A66"
STONE2 = "#9C9A94"
# The brand's palette: dusk is the accent, the series a figure is about; the deep forms carry a white letter or text at
# small sizes.
DUSK = "#2D3C85"
PERIWINKLE = "#8C9EEA"
CORAL = "#FF7F72"
CORAL_DEEP = "#BC3224"
AQUA_DEEP = "#0D7368"
MAUVE_DEEP = "#744096"
# Deep shades of the spectrum's other hues, for plots in which several lines share a panel: glow's, peach's, rose's and
# a lighter periwinkle's, each at least 3:1 on white
GOLD = "#C08A00"
VERMILLION = "#D55E00"
ROSE = "#CC79A7"
SKY = "#3F8FD6"
SLATE = "#5B6282"
SLATE2 = "#8C92AC"
HAZE = "#F1F2F7"
LINE = (0.067, 0.067, 0.067, 0.12)          # ink at 12%
LINE_STRONG = (0.067, 0.067, 0.067, 0.5)    # ink at 50%

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
SANS = "Instrument Sans"
MONO = "IBM Plex Mono"
SERIF = "Newsreader"


def register_fonts() -> None:
    for f in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))


def apply() -> None:
    register_fonts()
    matplotlib.rcParams.update({
        "font.family": SANS,
        "font.size": 8,
        "text.color": INK,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.labelcolor": STONE,
        "axes.edgecolor": STONE2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "xtick.color": STONE,
        "ytick.color": STONE,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def mono(size: float = 6.5):
    """Font properties for numerals and meta text (IBM Plex Mono)."""
    return font_manager.FontProperties(family=MONO, size=size)


def serif_italic(size: float = 7.5):
    return font_manager.FontProperties(family=SERIF, style="italic", size=size)
