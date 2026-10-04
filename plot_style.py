"""
Shared fonts, colours, figure sizes and task labels for thesis plots.

Import as `import plot_style as ps` to apply rcParams. Use ps.C_* for colours,
ps.TASK_* for labels and ps.save(fig, path) to save and close figures.

Use captions for figure titles; keep panel titles where needed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fonts and grid
FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family":      FONT,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.titlesize": 12,

    "axes.edgecolor":   "#444444",
    "axes.linewidth":   0.9,
    "axes.grid":        True,
    "axes.axisbelow":   True,
    "grid.color":       "#E4E4E4",
    "grid.linewidth":   0.8,

    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# Figure sizes in inches; shared width, variable height
W          = 9.0
FIG_SINGLE = (W, 5.4)   # one panel
FIG_WIDE   = (W, 4.6)   # two panels side by side
FIG_TALL   = (W, 6.2)   # one panel with many rows, e.g. the coefficient comparison

# Task labels
TASK_CUED = "Cued AAT"
TASK_DP   = "Dual Picture Task"
TASK_SP   = "Single Picture Task"

# Picture role
C_TARGET = "#2F6DA3"
C_DIST   = "#E08A3C"

# Task
C_CUED   = "#4C3C8F"   # indigo
C_DP     = "#9C3D5A"   # wine
C_SP     = "#6E8B3D"   # olive

# Valence
C_POS    = "#3E8E7E"   # teal
C_NEG    = "#A63D5B"   # dark rose

# Congruency
C_CONG   = "#8AA7BF"
C_INCONG = "#2E4A63"

# Significance
C_SIG    = "#1E2749"
C_NS     = "#9AA0AE"

# Bars without categorical colour coding
C_BAR    = "#5A7D9A"

# Participant lines, zero lines, empty bins and text
C_LINES  = "#B9C2CE"
C_ZERO   = "#888888"
C_EMPTY  = "#F2F2F2"
C_TEXT   = "#333333"


def save(fig, path, verbose=True):
    """Save with rcParams settings, close the figure and optionally print its filename."""
    fig.savefig(path)
    plt.close(fig)
    if verbose:
        import os
        print("saved", os.path.basename(path))
