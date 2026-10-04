"""
Negative-target intensity - Boundary sensitivity checks

1) Compare participant effects across alternative valence cutoffs
2) Regress picture RT on intensity within bins and across negative pictures
3) Repeat the categorical contrast without six near-boundary pictures

Exploratory analyses; results do not alter the main models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# === Paths and settings ===
ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"                / "results" / "cuedtask_trial_table_preprocessed.csv"
DUALPIC_CSV  = ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"
RESULTS_DIR  = ROOT / "results_combined"
RESULTS_DIR.mkdir(exist_ok=True)

for p in [CUEDTASK_CSV, DUALPIC_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run preprocessing scripts first")

NEUTRAL = 5.0
# Sequence IDs with valence 2.41–2.47, just below the 2.5 boundary
BOUNDARY_BAND = [75, 76, 71, 63, 84, 44]

out = []


def say(line=""):
    print(line)
    out.append(line)


def categorical_effect(d):
    """participant-level medium-minus-low contrast, the model's own comparison"""
    piv = d.groupby(["subject", "target_intensity_e"])["log_rt"].mean().unstack("target_intensity_e")
    if -1.0 not in piv.columns or 0.0 not in piv.columns:
        return None
    diff = (piv[0.0] - piv[-1.0]).dropna()
    t, p = scipy_stats.ttest_1samp(diff.values, 0.0)
    return (np.exp(diff.mean()) - 1) * 100, p, len(diff)


def split_effect(d, cutoff):
    """Test more-minus-less intense contrasts at a given valence cutoff."""
    d = d.copy()
    d["grp"] = np.where(d["target_valence_iaps"] < cutoff, "more", "less")
    piv = d.groupby(["subject", "grp"])["log_rt"].mean().unstack("grp")
    if "more" not in piv.columns or "less" not in piv.columns:
        return None
    diff = (piv["more"] - piv["less"]).dropna()
    if len(diff) < 5:
        return None
    t, p = scipy_stats.ttest_1samp(diff.values, 0.0)
    return (np.exp(diff.mean()) - 1) * 100, p


def picture_table(d):
    """Normalize RT within participant, then average trials by picture."""
    grand = d["rt_ms"].mean()
    d = d.copy()
    d["rt_norm"] = d["rt_ms"] - d.groupby("subject")["rt_ms"].transform("mean") + grand
    pics = (d.groupby("target_pic")
              .agg(valence=("target_valence_iaps", "first"),
                   rt=("rt_norm", "mean"),
                   cat=("target_intensity_e", "first"))
              .reset_index())
    pics["intensity"] = (pics["valence"] - NEUTRAL).abs()
    return pics

# === Run checks for negative targets in each task ===
for name, path in [("CuedAAT", CUEDTASK_CSV), ("DualPicture", DUALPIC_CSV)]:
    df = pd.read_csv(path)
    neg = df[(df["target_valence"] == "negative") & df["target_valence_iaps"].notna()].copy()

    say(f"{name}, NEGATIVE targets")

    base = categorical_effect(neg)
    say(f"  Participant contrast (medium - low): {base[0]:+.2f}%, p={base[1]:.4f}, n={base[2]}")

    say("\n  [1] Alternative valence cutoffs (exploratory, unadjusted p-values)")
    say("    cutoff   effect      p")
    for c in [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]:
        r = split_effect(neg, c)
        n_more = neg.loc[neg["target_valence_iaps"] < c, "target_pic"].nunique()
        n_less = neg.loc[neg["target_valence_iaps"] >= c, "target_pic"].nunique()
        if r and n_more >= 3 and n_less >= 3:
            tag = "   <-- the rounding boundary the model uses" if abs(c - 2.5) < 1e-9 else ""
            say(f"     {c:.2f}    {r[0]:+6.2f}%   p={r[1]:.4f}{tag}")

    say("\n  [2] Picture-level RT slopes by intensity")
    pics = picture_table(neg)
    for c, lab in [(0.0, "more intense bin (code 0)"),
                   (-1.0, "less intense bin (code -1)")]:
        sub = pics[pics["cat"] == c]
        lr = scipy_stats.linregress(sub["intensity"], sub["rt"])
        say(f"    {lab:<36} slope={lr.slope:+7.1f} ms/point, r={lr.rvalue:+.3f}, p={lr.pvalue:.3f}, {len(sub)} pics")
    lr = scipy_stats.linregress(pics["intensity"], pics["rt"])
    say(f"    {'all negative pictures':<36} slope={lr.slope:+7.1f} ms/point, r={lr.rvalue:+.3f}, p={lr.pvalue:.3f}, {len(pics)} pics")

    say("\n  [3] Exclude the six near-boundary pictures")
    band = pics[pics["target_pic"].isin(BOUNDARY_BAND)]
    say(f"    Boundary-band mean RT minus all-negative-picture mean: {band['rt'].mean() - pics['rt'].mean():+.1f} ms")
    wo = categorical_effect(neg[~neg["target_pic"].isin(BOUNDARY_BAND)])
    say(f"    effect with those six removed: {wo[0]:+.2f}%, p={wo[1]:.4f}  (was {base[0]:+.2f}%, p={base[1]:.4f})")
    say()

# === Save results ===
result_path = RESULTS_DIR / "intensity_boundary_checks.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"results saved: {result_path}")
