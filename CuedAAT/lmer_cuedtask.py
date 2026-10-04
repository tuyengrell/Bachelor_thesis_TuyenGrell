"""
Cued AAT - Linear mixed-effects model

Steps:
1) Summarize RT by condition, target valence and intensity
2) Compare nested models by likelihood ratio test (LRT) to see which terms are needed
3) Fit final model (REML) and report coefficients, SE, p-values; retain all six predictors
4) Check intensity effects with random-slope model and participant-level paired test
5) Save results and plot mean RT by intensity

Intensity is coded by picture role (target/distractor).
All predictors are pre-specified (theory-driven) and are kept in the final model regardless of their own p-value.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from pathlib import Path

# === Paths and data ===
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
CSV_PATH    = RESULTS_DIR / "cuedtask_trial_table_preprocessed.csv"

# Shared descriptive helpers and plot style
sys.path.insert(0, str(SCRIPT_DIR.parent))
from analysis_common import subject_desc, subject_means, subject_cells

import plot_style as ps

print("Cued AAT LMER")

if not CSV_PATH.exists():
    raise SystemExit(f"csv not found: {CSV_PATH} -- run preprocess first")

df = pd.read_csv(CSV_PATH)
print(f"loaded {len(df)} trials from {df['subject'].nunique()} subjects")

for col in ("target_valence_e", "target_intensity_e", "distractor_intensity_e", "target_side_e"):
    if col not in df.columns:
        raise SystemExit(f"missing column: {col} -- re-run preprocess_lmer_cuedtask.py")

# === Descriptive RT statistics ===
desc_full = subject_desc(df, ["condition", "target_valence"])
print("\nRT by condition x target valence (participant-level M/SD/Mdn):")
print(desc_full.to_string())

cong_cells = subject_means(df, "condition")
val_cells  = subject_means(df, "target_valence")
cong_rt, incong_rt = cong_cells["congruent"], cong_cells["incongruent"]
pos_tv,  neg_tv    = val_cells["positive"],  val_cells["negative"]
print(f"\ncongruency effect:  {incong_rt - cong_rt:+.1f} ms  (per-participant means)")
print(f"target valence:     {neg_tv - pos_tv:+.1f} ms (neg - pos, per-participant means)")

# Trial counts and mean RT by intensity
print("\ntarget_intensity_e distribution (strength of the CUED image):")
print(df["target_intensity_e"].value_counts().sort_index())
print("mean RT by target_intensity_e (per-participant means):")
print(subject_means(df, "target_intensity_e").round(1))
print("\ndistractor_intensity_e distribution (strength of the UNCUED image):")
print(df["distractor_intensity_e"].value_counts().sort_index())
print("mean RT by distractor_intensity_e (per-participant means):")
print(subject_means(df, "distractor_intensity_e").round(1))

# === Prepare data for LMER ===
df_model = df.copy()
df_model["subject"] = df_model["subject"].astype(str)
df_model["cond_e"]  = df_model["condition"].map({"congruent": -0.5, "incongruent": 0.5})

print(f"\nmodel dataset: {len(df_model)} trials, {df_model['subject'].nunique()} subjects")


def fit(formula, reml):
    """Fit a subject random-intercept model."""
    return smf.mixedlm(
        formula, data=df_model, groups=df_model["subject"]
    ).fit(reml=reml, method="powell", maxiter=200)


def lrt(m_simple, m_complex):
    """Compare two nested models by likelihood ratio test (LRT) and return chi2, df, p-value."""
    chi2 = 2 * (m_complex.llf - m_simple.llf)
    df_  = m_complex.df_modelwc - m_simple.df_modelwc
    p    = scipy_stats.chi2.sf(chi2, df_)
    return chi2, df_, p


# === Compare nested models by LRT (ML) but retain all predictors ===
print("\nfitting models (stepwise, for reporting -- nothing gets dropped)...")
M0 = fit("log_rt ~ 1", reml=False)
M1 = fit("log_rt ~ cond_e", reml=False)
M2 = fit("log_rt ~ cond_e + target_valence_e", reml=False)
M3 = fit("log_rt ~ cond_e + target_valence_e + target_intensity_e", reml=False)
M4 = fit("log_rt ~ cond_e + target_valence_e + target_intensity_e + distractor_intensity_e", reml=False)
M5 = fit("log_rt ~ cond_e + target_valence_e + target_intensity_e + distractor_intensity_e + target_side_e", reml=False)
M6 = fit("log_rt ~ cond_e + target_valence_e + target_intensity_e + distractor_intensity_e + target_side_e + z_trial", reml=False)

chi2_01, df_01, p_01 = lrt(M0, M1)
chi2_12, df_12, p_12 = lrt(M1, M2)
chi2_23, df_23, p_23 = lrt(M2, M3)
chi2_34, df_34, p_34 = lrt(M3, M4)
chi2_45, df_45, p_45 = lrt(M4, M5)
chi2_56, df_56, p_56 = lrt(M5, M6)

print(f"\nmodel building (LRT, ML) -- each row = adding one term to the row above:")
print(f"  M0 intercept only               AIC={M0.aic:.2f}")
print(f"  M1 + cond_e                     AIC={M1.aic:.2f}  chi2={chi2_01:.3f}  df={df_01}  p={p_01:.4f}")
print(f"  M2 + target_valence_e           AIC={M2.aic:.2f}  chi2={chi2_12:.3f}  df={df_12}  p={p_12:.4f}")
print(f"  M3 + target_intensity_e         AIC={M3.aic:.2f}  chi2={chi2_23:.3f}  df={df_23}  p={p_23:.4f}")
print(f"  M4 + distractor_intensity_e     AIC={M4.aic:.2f}  chi2={chi2_34:.3f}  df={df_34}  p={p_34:.4f}")
print(f"  M5 + target_side_e              AIC={M5.aic:.2f}  chi2={chi2_45:.3f}  df={df_45}  p={p_45:.4f}")
print(f"  M6 + z_trial                    AIC={M6.aic:.2f}  chi2={chi2_56:.3f}  df={df_56}  p={p_56:.4f}")
print("  (a term having p >= .05 here does NOT remove it from the final model, see note at top of file)")

# === Fit final model (REML) and report coefficients, SE, p-values ===
final_formula = ("log_rt ~ cond_e + target_valence_e + target_intensity_e"
                 " + distractor_intensity_e + target_side_e + z_trial")
print(f"\nfinal model (REML), all pre-specified terms kept:")
print(f"  {final_formula}")
Mfinal_reml = fit(final_formula, reml=True)
print(Mfinal_reml.summary())

# === Intensity robustness checks: random-slope model and participant-level paired test ===
robust_lines = []


def _robustness(term):
    out = [f"\n[robustness] {term}"]
    fixed = Mfinal_reml.params.get(term, float("nan"))
    se    = Mfinal_reml.bse.get(term, float("nan"))
    pv    = Mfinal_reml.pvalues.get(term, float("nan"))
    out.append(f"  random intercept only : b={fixed:+.4f}  SE={se:.4f}  p={pv:.4f}")
    # Add a by-subject random slope for this intensity term
    try:
        m_slope = smf.mixedlm(final_formula, data=df_model,
                              groups=df_model["subject"],
                              re_formula=f"~{term}").fit(reml=True, method="powell",
                                                         maxiter=400)
        if m_slope.converged:
            out.append(f"  + random slope        : b={m_slope.params[term]:+.4f}  "
                       f"SE={m_slope.bse[term]:.4f}  p={m_slope.pvalues[term]:.4f}")
            ch, dfree, pl = lrt(Mfinal_reml, m_slope)
            out.append(f"  slope variance needed?  LRT chi2={ch:.3f}  df={dfree}  p={pl:.4f}")
        else:
            out.append("  + random slope        : did not converge (too little data "
                       "to estimate between-participant variation in this effect)")
    except Exception as exc:                                   # noqa: BLE001
        out.append(f"  + random slope        : failed ({exc})")
    # Average across valence cells; paired t-test of participant-level means
    cell = df_model.groupby(["subject", term, "target_valence"])["log_rt"].mean().reset_index()
    cell = cell.groupby(["subject", term])["log_rt"].mean().reset_index()
    piv  = cell.pivot(index="subject", columns=term, values="log_rt").dropna()
    if piv.shape[1] == 2:
        a, b = sorted(piv.columns)
        d = piv[b] - piv[a]
        t_stat, p_t = scipy_stats.ttest_rel(piv[b], piv[a])
        out.append(f"  participant-level     : {(np.exp(d.mean())-1)*100:+.2f}%  "
                   f"t({len(d)-1})={t_stat:.2f}  p={p_t:.4f}  "
                   f"({int((d>0).sum())}/{len(d)} participants in that direction)")
    return out


for _term in ("target_intensity_e", "distractor_intensity_e"):
    robust_lines += _robustness(_term)
print("\n".join(robust_lines))

# === Save results ===
result_path = RESULTS_DIR / "cuedtask_lmer_results.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("Cued AAT LMER results\n")
    f.write(f"N={df['subject'].nunique()} subjects, {len(df)} trials\n")
    f.write("RT filter: >= 150ms, outlier: +/-2.5 SD per subject\n\n")
    f.write("effect coding:\n")
    f.write("  cond_e:                   congruent=-0.5  incongruent=+0.5\n")
    f.write("  target_valence_e:         positive=-0.5   negative=+0.5\n")
    f.write("  target_side_e:            left=-0.5  right=+0.5\n")
    f.write("  target_intensity_e:       intensity of the CUED image, round(IAPS), +1=most extreme\n")
    f.write("  distractor_intensity_e:   intensity of the UNCUED image, round(IAPS), +1=most extreme\n\n")
    f.write("note on model selection: no significance-based backward elimination was used.\n")
    f.write("all six terms are pre-specified (theory-driven) and are kept in the final model\n")
    f.write("regardless of their own p-value. dropping a non-significant hypothesis term would\n")
    f.write("remove the answer to the research question instead of reporting it, and stepwise\n")
    f.write("selection by p-value is itself a known source of biased post-selection estimates\n")
    f.write("(Whittingham et al., 2006). significance-based elimination is only used later, for\n")
    f.write("the task-interaction terms in the Cued AAT vs Dual Picture Task comparison.\n\n")
    f.write("descriptive RT (participant level: M/SD/Mdn across participant cell means,\n")
    f.write("n_trials = trials in the cell, n_subj = participants in the cell):\n")
    f.write(desc_full.to_string())
    f.write(f"\n\ncongruency effect: {incong_rt - cong_rt:+.1f} ms (per-participant means)\n")
    f.write(f"target valence effect: {neg_tv - pos_tv:+.1f} ms (neg - pos, per-participant means)\n\n")
    f.write("model building LRT (ML), informative only, nothing dropped based on this:\n")
    f.write(f"  M0  AIC={M0.aic:.2f}\n")
    f.write(f"  M1  AIC={M1.aic:.2f}  chi2={chi2_01:.3f}  df={df_01}  p={p_01:.4f}\n")
    f.write(f"  M2  AIC={M2.aic:.2f}  chi2={chi2_12:.3f}  df={df_12}  p={p_12:.4f}\n")
    f.write(f"  M3  AIC={M3.aic:.2f}  chi2={chi2_23:.3f}  df={df_23}  p={p_23:.4f}\n")
    f.write(f"  M4  AIC={M4.aic:.2f}  chi2={chi2_34:.3f}  df={df_34}  p={p_34:.4f}\n")
    f.write(f"  M5  AIC={M5.aic:.2f}  chi2={chi2_45:.3f}  df={df_45}  p={p_45:.4f}\n")
    f.write(f"  M6  AIC={M6.aic:.2f}  chi2={chi2_56:.3f}  df={df_56}  p={p_56:.4f}\n\n")
    f.write(f"final model: {final_formula} [REML]\n\n")
    f.write(Mfinal_reml.summary().as_text())
    f.write("\n\nIntensity robustness checks (final model unchanged):\n")
    f.write("\n".join(robust_lines))
    f.write("\n")

print(f"results saved: {result_path}")

# === Plot mean RT by intensity (target vs distractor) ===
# Approximate 95% confidence intervals across participants means
fig, ax = plt.subplots(figsize=ps.FIG_SINGLE)
present = sorted(set(df_model["target_intensity_e"].dropna()) |
                 set(df_model["distractor_intensity_e"].dropna()))
for col, label, color in [
    ("target_intensity_e",     "Target (cued) image",     ps.C_TARGET),
    ("distractor_intensity_e", "Distractor (uncued) image", ps.C_DIST),
]:
    cells = subject_cells(df_model, col)
    g     = cells.groupby(col)["rt_ms"].agg(["mean", "std", "size"])
    x     = [v for v in present if v in g.index]
    y     = [g.loc[v, "mean"] for v in x]
    yerr  = [1.959964 * g.loc[v, "std"] / np.sqrt(g.loc[v, "size"]) for v in x]
    ax.errorbar(x, y, yerr=yerr, label=label, color=color,
                marker="o", capsize=5, linewidth=2, markersize=7)
# Display all intensity levels; shade empty bins
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels(["low" if -1 in present else "low\n(no pictures)",
                    "medium" if 0 in present else "medium\n(no pictures)",
                    "high" if 1 in present else "high\n(no pictures)"])
ax.set_xlim(-1.5, 1.5)
for v in (-1, 0, 1):
    if v not in present:
        ax.axvspan(v - 0.45, v + 0.45, color=ps.C_EMPTY, zorder=0)
ax.set_xlabel("Intensity (rounded IAPS, role-based)")
ax.set_ylabel("Mean RT (ms)")

ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plot_path = RESULTS_DIR / "cuedtask_target_distractor_intensity.png"
ps.save(fig, plot_path, verbose=False)
print(f"plot saved: {plot_path}")
