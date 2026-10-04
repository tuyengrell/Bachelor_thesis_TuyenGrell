"""
Participant-level target-intensity interaction checks.

Compare medium-minus-low log-RT effects across congruency and target valence,
then across congruency within each valence. Test paired effect differences
with one-sample t-tests. Separate from the main models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"                / "results" / "cuedtask_trial_table_preprocessed.csv"
DUALPIC_CSV  = ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"
RESULTS_DIR  = ROOT / "results_combined"
RESULTS_DIR.mkdir(exist_ok=True)

for p in [CUEDTASK_CSV, DUALPIC_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run preprocessing scripts first")


def intensity_effect_within(df, split_col, split_val, val_col="log_rt"):
    """Return each participant's medium-minus-low effect within a subgroup."""
    sub = df[df[split_col] == split_val]
    piv = sub.groupby(["subject", "target_intensity_e"])[val_col].mean().unstack("target_intensity_e")
    if -1.0 not in piv.columns or 0.0 not in piv.columns:
        return pd.Series(dtype=float)
    return (piv[0.0] - piv[-1.0]).dropna()


def interaction_test(df, split_col, val_a, val_b, label, task_name, out_lines):
    eff_a = intensity_effect_within(df, split_col, val_a)
    eff_b = intensity_effect_within(df, split_col, val_b)
    common = eff_a.index.intersection(eff_b.index)
    diff = eff_b.loc[common] - eff_a.loc[common]
    t, p = scipy_stats.ttest_1samp(diff.values, 0.0)
    # Descriptive effects use all available participants in each subgroup
    pct_a = (np.exp(eff_a.mean()) - 1) * 100
    pct_b = (np.exp(eff_b.mean()) - 1) * 100
    n = len(diff)
    line1 = f"\n[{task_name}] {label}"
    line2 = f"  intensity effect within {val_a!r}: {pct_a:+.2f}% RT change (n={len(eff_a)} participants)"
    line3 = f"  intensity effect within {val_b!r}: {pct_b:+.2f}% RT change (n={len(eff_b)} participants)"
    line4 = f"  interaction (paired diff, n={n}): t({n-1})={t:.3f}, p={p:.4f}"
    for l in (line1, line2, line3, line4):
        print(l)
        out_lines.append(l)
    return p


cued = pd.read_csv(CUEDTASK_CSV)
dp   = pd.read_csv(DUALPIC_CSV)
dp   = dp.rename(columns={"congruency": "condition"})  # align column name with CuedTask

out_lines = ["Target-intensity interaction checks (congruency, valence)",
             "not part of the main model, see note at top of file\n"]

print("CONGRUENCY x TARGET INTENSITY")
out_lines.append(" CONGRUENCY x TARGET INTENSITY ")
interaction_test(cued, "condition", "congruent", "incongruent",
                  "Congruency x Target Intensity", "CuedAAT", out_lines)
interaction_test(dp, "condition", "congruent", "incongruent",
                  "Congruency x Target Intensity", "DualPicture", out_lines)

print("\nVALENCE x TARGET INTENSITY")
out_lines.append("\nVALENCE x TARGET INTENSITY ")
interaction_test(cued, "target_valence", "positive", "negative",
                  "Valence x Target Intensity", "CuedAAT", out_lines)
interaction_test(dp, "target_valence", "positive", "negative",
                  "Valence x Target Intensity", "DualPicture", out_lines)

# Repeat the congruency comparison within each target valence
print("\nCONGRUENCY x TARGET INTENSITY, WITHIN EACH VALENCE")
out_lines.append("\nCONGRUENCY x INTENSITY, WITHIN VALENCE ")
for task_name, df in [("CuedAAT", cued), ("DualPicture", dp)]:
    for val in ("positive", "negative"):
        interaction_test(df[df["target_valence"] == val], "condition", "congruent", "incongruent",
                          f"Congruency x Target Intensity, {val} targets only", task_name, out_lines)

result_path = RESULTS_DIR / "intensity_interaction_checks.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print(f"\nresults saved: {result_path}")
