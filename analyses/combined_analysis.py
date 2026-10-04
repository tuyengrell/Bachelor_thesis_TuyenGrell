""" 
Cued vs Dual Picture Task - Combined mixed-effects analysis

Steps:
1) Harmonize trial tables and prefix subject IDs by task
2) Summarize valence effects by task
3) Test H3a–H3c with drop-one ML likelihood-ratio tests
4) Remove H3 interactions with p >= .05 by backward elimination
5) Refit the selected model with REML
6) Test exploratory task interactions separately and save the results

All seven main effects remain in the model.
H3a: task × target valence
H3b: task × target intensity
H3c: task × distractor intensity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

# === Paths and data ===
ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"               / "results" / "cuedtask_trial_table_preprocessed.csv"
DUALPIC_CSV  = ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"
RESULTS_DIR  = ROOT / "results_combined"
RESULTS_DIR.mkdir(exist_ok=True)

# Shared participant-level summaries and plot style
from analysis_common import subject_desc, subject_means


print("Combined analysis: Cued AAT vs Dual Picture Task")

for p in [CUEDTASK_CSV, DUALPIC_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run preprocessing scripts first")

df_cued = pd.read_csv(CUEDTASK_CSV)
df_dual = pd.read_csv(DUALPIC_CSV)

# Harmonize congruency coding across tasks
df_cued["cond_e"] = df_cued["condition"].map({"congruent": -0.5, "incongruent": 0.5})

# Keep Dual Picture trials with valid movement and congruency
df_dual = df_dual[df_dual["movement"].isin(["push", "pull"])].copy()
df_dual = df_dual[df_dual["congruency"].isin(["congruent", "incongruent"])].copy()
df_dual["cond_e"] = df_dual["congruency"].map({"congruent": -0.5, "incongruent": 0.5})

print(f"  CuedTask:    {len(df_cued)} trials, {df_cued['subject'].nunique()} subjects")
print(f"  DualPicture: {len(df_dual)} trials, {df_dual['subject'].nunique()} subjects")

needed = ["subject", "log_rt", "rt_ms", "z_trial", "cond_e",
          "target_valence_e", "target_intensity_e", "distractor_intensity_e", "target_side_e"]
for col in needed:
    for name, df_ in [("CuedAAT", df_cued), ("DualPicture", df_dual)]:
        if col not in df_.columns:
            raise SystemExit(f"missing column '{col}' in {name} -- re-run preprocessing")

# Prefix subject IDs to avoid overlap between tasks
df_cued_h = df_cued[needed].copy()
df_cued_h["task"]    = "CuedAAT"
df_cued_h["subject"] = "CT_" + df_cued_h["subject"].astype(str)

df_dual_h = df_dual[needed].copy()
df_dual_h["task"]    = "DualPicture"
df_dual_h["subject"] = "DP_" + df_dual_h["subject"].astype(str)

df_all = pd.concat([df_cued_h, df_dual_h], ignore_index=True)
df_all = df_all[df_all["target_valence_e"].notna()].copy()
df_all["task_e"] = df_all["task"].map({"CuedAAT": -0.5, "DualPicture": +0.5})

print(f"  combined:    {len(df_all)} trials, {df_all['subject'].nunique()} subjects (between-subjects)")

# === Descriptive RT statistics (participant cell means) ===
print("\nRT by task x target valence (participant M/SD/Mdn):")
desc = subject_desc(df_all, ["task", "target_valence_e"])
print(desc.to_string())

for task in ["CuedAAT", "DualPicture"]:
    cells = subject_means(df_all[df_all["task"] == task], "target_valence_e")
    print(f"  {task}: neg - pos target = {cells[+0.5] - cells[-0.5]:+.1f} ms "
          "(per-participant means)")

# === Combined mixed-effects models ===
print("\nfitting combined models...")

df_model = df_all.copy()
df_model["subject"] = df_model["subject"].astype(str)


def fit(formula, reml):
    """Fit a subject random-intercept model."""
    return smf.mixedlm(formula, data=df_model, groups=df_model["subject"]).fit(
        reml=reml, method="powell", maxiter=200
    )


def lrt(m0, m1):
    """Compare nested models using a chi-square likelihood-ratio test."""
    chi2 = 2 * (m1.llf - m0.llf)
    df   = m1.df_modelwc - m0.df_modelwc
    p    = scipy_stats.chi2.sf(chi2, df)
    return chi2, df, p


def make_formula(base, interactions):
    return base if not interactions else base + " + " + " + ".join(interactions)


# Retain all seven main effects throughout model selection
base_formula = ("log_rt ~ task_e + cond_e + target_valence_e + target_intensity_e"
                " + distractor_intensity_e + target_side_e + z_trial")

CANDIDATE_INTERACTIONS = [
    "task_e:target_valence_e",       # H3a
    "task_e:target_intensity_e",     # H3b
    "task_e:distractor_intensity_e", # H3c
]
H3_LABELS = {
    "task_e:target_valence_e":       "H3a (task x valence)",
    "task_e:target_intensity_e":     "H3b (task x target intensity)",
    "task_e:distractor_intensity_e": "H3c (task x distractor intensity)",
}

M_full = fit(make_formula(base_formula, CANDIDATE_INTERACTIONS), reml=False)

print(f"\nhypothesis tests (LRT, ML), each interaction vs the full model without it:")
h3_stats = {}
for term in CANDIDATE_INTERACTIONS:
    reduced   = [t for t in CANDIDATE_INTERACTIONS if t != term]
    m_reduced = fit(make_formula(base_formula, reduced), reml=False)
    chi2, ddf, p = lrt(m_reduced, M_full)
    h3_stats[term] = (chi2, ddf, p)
    print(f"  {H3_LABELS[term]:34s} chi2={chi2:.3f}  df={ddf}  p={p:.4f}")

# === Backward elimination of H3 interactions only ===
# Reuse the full model and initial drop-one LRTs
print(f"\n[backward elimination on task interactions only] main effects always kept:")
current_interactions = list(CANDIDATE_INTERACTIONS)
elimination_log = []
step = 1
m_current = M_full
pvals = {term: h3_stats[term][2] for term in CANDIDATE_INTERACTIONS}
while True:
    if not current_interactions:
        print("  no interaction terms left, stopping.")
        break
    worst_term = max(pvals, key=pvals.get)
    worst_p    = pvals[worst_term]
    print(f"\n  step {step}: current model = {make_formula(base_formula, current_interactions)}")
    for term in current_interactions:
        flag = "  <-- least significant" if term == worst_term else ""
        print(f"    drop {term:34s} LRT p={pvals[term]:.4f}{flag}")
    if worst_p >= 0.05:
        print(f"  DROP {worst_term}  (p={worst_p:.4f} >= .05, not significant)")
        elimination_log.append((step, worst_term, worst_p))
        current_interactions = [t for t in current_interactions if t != worst_term]
        step += 1
        m_current = fit(make_formula(base_formula, current_interactions), reml=False)
        pvals = {}
        for term in current_interactions:
            reduced   = [t for t in current_interactions if t != term]
            m_reduced = fit(make_formula(base_formula, reduced), reml=False)
            _, _, p   = lrt(m_reduced, m_current)
            pvals[term] = p
    else:
        print(f"  all remaining interactions are significant (worst p={worst_p:.4f}). stopping.")
        break

minimal_interactions = current_interactions
minimal_formula = make_formula(base_formula, minimal_interactions)
print(f"\nminimal combined model: {minimal_formula}")
if elimination_log:
    print("interactions dropped, in order:")
    for step_n, term, p in elimination_log:
        print(f"  step {step_n}: {term}  (p={p:.4f})")
else:
    print("nothing was dropped, all three task interactions were significant.")

print(f"\nfinal model (REML): {minimal_formula}")
Mfinal = fit(minimal_formula, reml=True)
print(Mfinal.summary())

# === Exploratory task interactions (outside H3 selection) === 
print(f"\n[exploratory, post-hoc] task interactions not in the pre-specified H1-H3 set:")
EXPLORATORY_INTERACTIONS = {
    "task_e:cond_e":        "does the congruency effect differ by task?",
    "task_e:target_side_e": "does the side effect differ by task (sign flip seen in single-task models)?",
    "task_e:z_trial":       "does the trial/practice slope differ by task (n.s. in DualPicture alone)?",
}
# Reuse the selected ML model as the exploratory baseline
m_minimal_ml = m_current
explor_stats = {}
for term, question in EXPLORATORY_INTERACTIONS.items():
    m_with = fit(make_formula(minimal_formula, [term]), reml=False)
    chi2, ddf, p = lrt(m_minimal_ml, m_with)
    explor_stats[term] = (chi2, ddf, p)
    print(f"  {term:24s} chi2={chi2:.3f}  df={ddf}  p={p:.4f}   ({question})")

# Preserve H3a/H3b/H3c result lines: thesis_plots.py parses this format
chi2_h3a, df_h3a, p_h3a = h3_stats["task_e:target_valence_e"]
chi2_h3b, df_h3b, p_h3b = h3_stats["task_e:target_intensity_e"]
chi2_h3c, df_h3c, p_h3c = h3_stats["task_e:distractor_intensity_e"]

# === Save model results and combine trial table ===
result_path = RESULTS_DIR / "combined_lmer_results.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("Combined analysis: Cued AAT vs Dual Picture Task\n")
    f.write(f"CuedTask:    {df_cued_h['subject'].nunique()} subjects, {len(df_cued_h)} trials\n")
    f.write(f"DualPicture: {df_dual_h['subject'].nunique()} subjects, {len(df_dual_h)} trials\n")
    f.write("Between-subjects design.\n\n")
    f.write("effect coding:\n")
    f.write("  task_e:                   CuedTask=-0.5  DualPicture=+0.5\n")
    f.write("  cond_e:                   congruent=-0.5  incongruent=+0.5\n")
    f.write("  target_valence_e:         positive=-0.5  negative=+0.5\n")
    f.write("  target_intensity_e/distractor_intensity_e: -1=low 0=medium +1=high (rounded IAPS, role-based)\n")
    f.write("    NOTE: under the IAPS all-subjects norms (Table 1) no picture rounds into the +1\n")
    f.write("    bin, so this predictor has only two occupied levels (48 low vs 39 medium\n")
    f.write("    pictures, out of the 87 in the set).\n\n")
    f.write("hypotheses:\n")
    f.write("  H3a: valence effect moderated by task (task x target_valence_e)\n")
    f.write("  H3b: target-intensity effect moderated by task (task x target_intensity_e)\n")
    f.write("  H3c: distractor-intensity effect moderated by task (task x distractor_intensity_e)\n\n")
    f.write("note on model selection: main effects (task_e, cond_e, target_valence_e,\n")
    f.write("target_intensity_e, distractor_intensity_e, target_side_e, z_trial) always stay in\n")
    f.write("the model, same reasoning as the single-task scripts. backward elimination is used\n")
    f.write("ONLY on the three task-interaction terms above: each\n")
    f.write("interaction is dropped if not significant, one at a time, refitting each step.\n\n")
    f.write("descriptive RT (participant level: M/SD/Mdn across participant cell means,\n")
    f.write("n_trials = trials in the cell, n_subj = participants in the cell):\n")
    f.write(desc.to_string())
    f.write("\n\nhypothesis tests (LRT, ML), each vs the full model without that interaction:\n")
    f.write(f"  H3a: chi2={chi2_h3a:.3f}  df={df_h3a}  p={p_h3a:.4f}\n")
    f.write(f"  H3b: chi2={chi2_h3b:.3f}  df={df_h3b}  p={p_h3b:.4f}\n")
    f.write(f"  H3c: chi2={chi2_h3c:.3f}  df={df_h3c}  p={p_h3c:.4f}\n\n")
    f.write("backward elimination (interactions only):\n")
    if elimination_log:
        for step_n, term, p in elimination_log:
            f.write(f"  step {step_n}: dropped {term}  (p={p:.4f})\n")
    else:
        f.write("  nothing dropped, all three interactions were significant.\n")
    f.write(f"\nfinal (minimal) model: {minimal_formula} [REML]\n\n")
    f.write(Mfinal.summary().as_text())
    f.write("\n\nexploratory post-hoc checks (outside H1-H3):\n")
    f.write("each tested as: final model + the one interaction term, vs. final model, LRT (ML).\n")
    for term, question in EXPLORATORY_INTERACTIONS.items():
        chi2, ddf, p = explor_stats[term]
        f.write(f"  {term:24s} chi2={chi2:.3f}  df={ddf}  p={p:.4f}   ({question})\n")

df_all.to_csv(RESULTS_DIR / "combined_trial_table.csv", index=False)
print(f"\nresults saved: {result_path}")
print("done.")
