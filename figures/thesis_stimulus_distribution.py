"""
Plot unrounded IAPS valence norms by stimulus valence.

Show the histogram, individual pictures and intensity cutoffs.
Report valence ranges and distances to the high-intensity cutoffs.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stimulus_map import IAPS_LOOKUP, POSITIVE_IDS, NEGATIVE_IDS

OUT = os.path.join(ROOT, "thesis_figures")
os.makedirs(OUT, exist_ok=True)

# Shared style for every figure, see plot_style.py
import plot_style as ps
from plot_style import C_POS, C_NEG, C_EMPTY

pos_vals = np.array(sorted(v for seq, v in IAPS_LOOKUP.items() if seq in POSITIVE_IDS))
neg_vals = np.array(sorted(v for seq, v in IAPS_LOOKUP.items() if seq in NEGATIVE_IDS))

# Valence cutoffs for the intensity bins
CUTS = [1.5, 2.5, 7.5, 8.5]

fig, (ax_h, ax_s) = plt.subplots(
    2, 1, figsize=ps.FIG_SINGLE, sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.13},
)

# Histogram and intensity boundaries
bins = np.arange(1.0, 9.0 + 0.2, 0.2)
for ax in (ax_h, ax_s):
    ax.axvspan(1.0, 1.5, color=C_EMPTY, zorder=0)
    ax.axvspan(8.5, 9.0, color=C_EMPTY, zorder=0)
    for x in CUTS:
        ax.axvline(x, color="#AAAAAA", lw=0.9, ls="--", zorder=1)
    ax.axvline(5.0, color="#888888", lw=1.1, ls="-", zorder=1)

ax_h.hist(neg_vals, bins=bins, color=C_NEG, alpha=0.88,
          edgecolor="white", linewidth=0.5, zorder=3,
          label=f"Unpleasant  (n = {len(neg_vals)})")
ax_h.hist(pos_vals, bins=bins, color=C_POS, alpha=0.88,
          edgecolor="white", linewidth=0.5, zorder=3,
          label=f"Pleasant  (n = {len(pos_vals)})")

ax_h.set_ylabel("Number of pictures")
ax_h.legend(loc="upper center", frameon=False, ncol=2)

ax_h.set_ylim(0, ax_h.get_ylim()[1] * 1.18)   # headroom for the legend
ymax = ax_h.get_ylim()[1]
for x in (1.25, 8.75):
    ax_h.text(x, ymax * 0.42, "no pictures reach here", ha="center",
              va="center", fontsize=8.5, color="#909090", rotation=90)

# Individual picture norms with vertical jitter
rng = np.random.default_rng(0)
ax_s.scatter(neg_vals, 0 + rng.uniform(-0.22, 0.22, len(neg_vals)),
             s=26, color=C_NEG, alpha=0.8, edgecolor="white",
             linewidth=0.4, zorder=3)
ax_s.scatter(pos_vals, 1 + rng.uniform(-0.22, 0.22, len(pos_vals)),
             s=26, color=C_POS, alpha=0.8, edgecolor="white",
             linewidth=0.4, zorder=3)
ax_s.set_ylim(-0.6, 1.6)
ax_s.set_yticks([0, 1])
ax_s.set_yticklabels(["Unpl.", "Pleas."], fontsize=10)
ax_s.set_xlabel("Valence norm, IAPS Table 1 / all subjects "
                "(1 = most unpleasant, 9 = most pleasant)")

ax_s.set_xlim(1.0, 9.0)
ax_s.set_xticks(np.arange(1, 10))

ps.save(fig, f"{OUT}/thesis_stimulus_valence_distribution.png")
print("Caption: Unrounded IAPS Table 1 valence norms (all subjects; Lang et al., 2008). "
      "Dashed lines mark intensity cutoffs; shading marks high-intensity ranges.")

# Descriptive statistics
print(f"  pleasant    n={len(pos_vals):2d}  range {pos_vals[0]:.2f} - {pos_vals[-1]:.2f}"
      f"  M={pos_vals.mean():.2f}  SD={pos_vals.std(ddof=1):.2f}")
print(f"  unpleasant  n={len(neg_vals):2d}  range {neg_vals[0]:.2f} - {neg_vals[-1]:.2f}"
      f"  M={neg_vals.mean():.2f}  SD={neg_vals.std(ddof=1):.2f}")
print(f"  gap to the '+1' bin cutoff: pleasant {8.5 - pos_vals[-1]:.2f}, "
      f"unpleasant {neg_vals[0] - 1.5:.2f}")
print("  High-bin occupancy depends on the stimulus norms and the chosen cutoffs.")
