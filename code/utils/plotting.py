"""
Shared matplotlib style for all project figures (paper is printed, light background).

Colours: damage grades are ORDERED, so they use one hue light->dark (blue ordinal
ramp, steps 250..650, lightest step still >= 2:1 contrast on the surface).
Categorical colours (models, structural types) use a fixed slot order, never cycled.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

GRADE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # Grade 1..5
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]

COL_WIDTH_IN = 3.25      # one column of the two-column template
FULL_WIDTH_IN = 6.75


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": MUTED,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
