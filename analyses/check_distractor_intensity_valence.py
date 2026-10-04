""" 
Distractor intensity - Split by target valence

Compute participant medium-minus-low log-RT effects within each target
valence. Test both simple effects and their paired difference against zero.

Each pair contains opposite valences: a positive target has a negative
distractor, and vice versa. Tests use participants with both simple effects.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# === Paths and required columns ===
ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"                / "results" / "cuedtask_trial_table_preprocessed.csv"
DUALPIC_CSV  = ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"
RESULTS_DIR  = ROOT / "results_combined"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_PATH = RESULTS_DIR / "distractor_intensity_valence_split.txt"

TERM = "distractor_intensity_e"
NEEDED = ["subject", "log_rt", TERM, "target_valence"]

for p in [CUEDTASK_CSV, DUALPIC_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run the preprocessing scripts first")

out = []


def say(line=""):
    print(line)
    out.append(line)


def pct(log_diff):
    return (np.exp(log_diff) - 1.0) * 100.0


def effect_within(df, valence):
    """each participant's own distractor-intensity effect within one valence category"""
    sub = df[df["target_valence"] == valence]
    piv = sub.groupby(["subject", TERM])["log_rt"].mean().unstack(TERM)
    if -1.0 not in piv.columns or 0.0 not in piv.columns:
        return pd.Series(dtype=float)
    return (piv[0.0] - piv[-1.0]).dropna()


say("Distractor intensity split by target valence")
say("Positive target = negative distractor; negative target = positive distractor.")
say()

# === Test simple effects and their paired difference ===
for name, path in [("CuedAAT", CUEDTASK_CSV), ("DualPicture", DUALPIC_CSV)]:
    df = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{name}: columns missing from {path.name}: {missing}")
    df = df.dropna(subset=NEEDED).copy()

    eff_pos = effect_within(df, "positive")   # negative distractor
    eff_neg = effect_within(df, "negative")   # positive distractor
    if eff_pos.empty or eff_neg.empty:
        raise SystemExit(f"{name}: one valence category has only one occupied intensity level")

    common = eff_pos.index.intersection(eff_neg.index)
    a, b = eff_pos.loc[common], eff_neg.loc[common]
    diff = b - a
    t, p = scipy_stats.ttest_1samp(diff.values, 0.0)
    n = len(diff)

    # Test both simple effects in the same matched participant sample
    t_a, p_a = scipy_stats.ttest_1samp(a.values, 0.0)
    t_b, p_b = scipy_stats.ttest_1samp(b.values, 0.0)

    say(f"[{name}] target valence x distractor intensity")
    say(f"  Positive targets: {pct(a.mean()):+.2f}% "
        f"RT change   t({len(a)-1})={t_a:+.3f}, p={p_a:.4f}   (n={len(eff_pos)})")
    say(f"  Negative targets: {pct(b.mean()):+.2f}% "
        f"RT change   t({len(b)-1})={t_b:+.3f}, p={p_b:.4f}   (n={len(eff_neg)})")
    say(f"  interaction (paired diff, n={n}): t({n-1})={t:.3f}, p={p:.4f}")
    if p_a >= 0.05 and p_b >= 0.05:
        say("  neither simple effect reaches p < .05.")

    same_sign = (a.mean() > 0) == (b.mean() > 0)
    if p < 0.05:
        say("  Evidence of a distractor-intensity effect difference between target valences.")
    elif same_sign:
        say("  No significant difference; both estimates are positive or both non-positive.")
    else:
        say("  No significant difference; one estimate is positive and the other non-positive.")
    say()

say("This split supplements H1; non-significant tests do not establish absent effects.")

# === Save results ===
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\nresults saved: {OUT_PATH}")
