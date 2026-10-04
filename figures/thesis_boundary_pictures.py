"""
Figure of the six negative pictures at the 2.5 rounding boundary (Cued AAT).

Normalise RT within participants, average by picture and plot picture RT
against intensity (|valence - 5|), highlighting the six boundary pictures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"                / "results" / "cuedtask_trial_table_preprocessed.csv"
OUT          = ROOT / "thesis_figures"
OUT.mkdir(exist_ok=True)

for p in [CUEDTASK_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run preprocessing scripts first")

import plot_style as ps
from plot_style import C_NEG
NEUTRAL_MIDPOINT = 5.0


def picture_means(df):
    """Return picture means after centring participant RTs and restoring the grand mean."""
    d = df[df["target_valence"].isin(["positive", "negative"])].copy()
    d = d[d["target_valence_iaps"].notna()]
    grand = d["rt_ms"].mean()
    d["rt_norm"] = d["rt_ms"] - d.groupby("subject")["rt_ms"].transform("mean") + grand
    out = (d.groupby(["target_pic", "target_valence"])
             .agg(valence=("target_valence_iaps", "first"),
                  rt=("rt_norm", "mean"),
                  n_trials=("rt_norm", "size"))
             .reset_index())
    out["intensity"] = (out["valence"] - NEUTRAL_MIDPOINT).abs()
    return out


# Sequence IDs with valence norms from 2.41 to 2.47
BOUNDARY_SEQ = [75, 76, 71, 63, 84, 44]


def highlight_boundary(df, fname):
    """Highlight six negative pictures near the valence cutoff of 2.5."""
    pics = picture_means(df)
    neg = pics[pics["target_valence"] == "negative"].copy()
    fig, ax = plt.subplots(figsize=ps.FIG_SINGLE)

    rest = neg[~neg["target_pic"].isin(BOUNDARY_SEQ)]
    band = neg[neg["target_pic"].isin(BOUNDARY_SEQ)]

    ax.scatter(rest["intensity"], rest["rt"], s=40, color=C_NEG, alpha=0.55,
               edgecolor="white", linewidth=0.6, zorder=3, label="other negative pictures")
    ax.scatter(band["intensity"], band["rt"], s=90, color=ps.C_SIG, alpha=0.95,
               edgecolor="white", linewidth=1.2, zorder=5,
               label="6 pictures right at the 2.5 cutoff")
    ax.axvline(0.0, color="#999", ls=":", lw=0.8)
    ax.axhline(neg["rt"].mean(), color="#999", ls="--", lw=1, zorder=1,
               label=f"mean over all negative pictures ({neg['rt'].mean():.0f} ms)")
    ax.set_xlabel("Intensity: distance from neutral midpoint (|valence - 5|)")
    ax.set_ylabel("Mean RT (ms), normalised within participant")
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95)
    plt.tight_layout()
    ps.save(fig, OUT / fname, verbose=False)
    print(f"saved {fname}  (boundary pictures {band['rt'].mean():.1f} ms vs all negative {neg['rt'].mean():.1f} ms)")


cued = pd.read_csv(CUEDTASK_CSV)
highlight_boundary(cued, "thesis_intensity_boundary_highlight_cued.png")
