""" 
H2 - Target vs distractor intensity effects

1) Test b_target - b_distractor = 0 using a REML model contrast
2) Compare participant-level intensity effects with a paired t-test

Both tests are two-sided. Directional support requires checking the sign.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

# === Paths and model ===
ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"                / "results" / "cuedtask_trial_table_preprocessed.csv"
DUALPIC_CSV  = ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"
RESULTS_DIR  = ROOT / "results_combined"
RESULTS_DIR.mkdir(exist_ok=True)

FORMULA = ("log_rt ~ cond_e + target_valence_e + target_intensity_e "
           "+ distractor_intensity_e + target_side_e + z_trial")
# Derive cond_e from condition (Cued) or congruency (Dual Picture)
NEEDED = ["subject", "log_rt", "target_valence", "target_valence_e",
          "target_intensity_e", "distractor_intensity_e", "target_side_e", "z_trial"]


def add_cond_e(df, task_name):
    col = "condition" if "condition" in df.columns else (
          "congruency" if "congruency" in df.columns else None)
    if col is None:
        raise SystemExit(f"{task_name}: neither 'condition' nor 'congruency' in the csv")
    df = df[df[col].isin(["congruent", "incongruent"])].copy()
    df["cond_e"] = df[col].map({"congruent": -0.5, "incongruent": 0.5})
    df["subject"] = df["subject"].astype(str)
    return df

for p in [CUEDTASK_CSV, DUALPIC_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run the preprocessing scripts first")

out = []


def say(line=""):
    print(line)
    out.append(line)


def pct(log_diff):
    """Convert a log-RT difference to percentage RT change."""
    return (np.exp(log_diff) - 1.0) * 100.0


def model_contrast(df, task_name):
    """Test the target-minus-distractor coefficient contrast."""
    res = smf.mixedlm(FORMULA, df, groups=df["subject"]).fit(
        reml=True, method="powell", maxiter=200
    )
    names = list(res.model.exog_names)
    i_t = names.index("target_intensity_e")
    i_d = names.index("distractor_intensity_e")

    # Contrast matrix covers fixed effects only
    c = np.zeros((1, len(names)))
    c[0, i_t] = 1.0
    c[0, i_d] = -1.0

    tt = res.t_test(c)
    est = float(np.ravel(tt.effect)[0])
    se = float(np.ravel(tt.sd)[0])
    z = float(np.ravel(tt.statistic)[0])
    p = float(np.ravel(tt.pvalue)[0])
    lo, hi = [float(v) for v in np.ravel(tt.conf_int())[:2]]

    b_t = float(res.params["target_intensity_e"])
    b_d = float(res.params["distractor_intensity_e"])

    say(f"  Model contrast (REML, full covariance)")
    say(f"      b target     = {b_t:+.4f}")
    say(f"      b distractor = {b_d:+.4f}")
    say(f"      difference   = {est:+.4f}  SE={se:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    say(f"      z = {z:+.3f}   p = {p:.4f}   ({pct(est):+.2f}% in RT terms)")
    say(f"      converged: {res.converged}")
    return p


def participant_level(df, task_name):
    """Compare medium-minus-low effects, averaging available valence cells equally."""
    def role_effect(term):
        cell = df.groupby(["subject", term, "target_valence"])["log_rt"].mean().reset_index()
        cell = cell.groupby(["subject", term])["log_rt"].mean().reset_index()  # valence-balanced
        piv = cell.pivot(index="subject", columns=term, values="log_rt")
        if -1.0 not in piv.columns or 0.0 not in piv.columns:
            raise SystemExit(f"{task_name}: {term} does not have both occupied levels")
        return (piv[0.0] - piv[-1.0]).dropna()

    eff_t = role_effect("target_intensity_e")
    eff_d = role_effect("distractor_intensity_e")
    common = eff_t.index.intersection(eff_d.index)
    eff_t, eff_d = eff_t.loc[common], eff_d.loc[common]

    diff = eff_t - eff_d
    t, p = scipy_stats.ttest_rel(eff_t.values, eff_d.values)
    n = len(diff)
    n_pos = int((diff > 0).sum())

    say(f"  Participant level effects (paired, valence-balanced)")
    say(f"      target effect     = {pct(eff_t.mean()):+.2f}%")
    say(f"      distractor effect = {pct(eff_d.mean()):+.2f}%")
    say(f"      difference        = {pct(diff.mean()):+.2f}%   "
        f"t({n-1}) = {t:+.3f}   p = {p:.4f}   ({n_pos}/{n} participants in that direction)")
    return p


# === Run both contrasts for each task ===
say("H2: target-minus-distractor intensity contrast (two-sided tests)")
say( )

for name, path in [("Cued AAT", CUEDTASK_CSV), ("Dual Picture Task", DUALPIC_CSV)]:
    df = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{name}: columns missing from {path.name}: {missing}")
    df = add_cond_e(df, name)
    df = df.dropna(subset=NEEDED + ["cond_e"]).copy()

    say(f"{name}: {len(df)} trials, {df['subject'].nunique()} participants")
    say("-" * 78)
    p1 = model_contrast(df, name)
    say()
    p2 = participant_level(df, name)
    say()
    if p1 < 0.05 and p2 < 0.05:
        say("  Both approaches agree: the two roles differ. H2 is supported.")
    elif p1 < 0.05 or p2 < 0.05:
        say("  Only one test detects a role difference; report both results.")
    else:
        say("  No evidence that the two roles differ at p < .05. H2 is not supported here.")
    say()

say("Report the direct contrasts, their direction, and the individual effects.")

# === Save results ===
result_path = RESULTS_DIR / "h2_role_contrast.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\nresults saved: {result_path}")
