""" 
H3a - Exploratory picture-selection check

Steps:
1) Normalize RT within participant and average trials by picture
2) Match target pictures across Cued and Dual Picture tasks
3) Compare target counts and availability-adjusted selection rates
4) Correlate Cued RT with selection rates, overall and within valence
5) Plot picture RTs and target counts; save the report

Uses retained trials from the preprocessed CSVs. Associations are descriptive;
unequal availability and filtering can affect target counts and rates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# === Paths ===
ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"                / "results" / "cuedtask_trial_table_preprocessed.csv"
DUALPIC_CSV  = ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"
RESULTS_DIR  = ROOT / "results_combined"
FIG_DIR      = ROOT / "thesis_figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

for p in [CUEDTASK_CSV, DUALPIC_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run preprocessing scripts first")

# Shared style for every figure in the thesis
import plot_style as ps
from plot_style import C_POS, C_NEG, C_CUED, C_DP, TASK_CUED, TASK_DP

out = []


def say(line=""):
    print(line)
    out.append(line)


def load(path):
    df = pd.read_csv(path)
    df = df[df["target_valence"].isin(["positive", "negative"])].copy()
    df["subject"] = df["subject"].astype(str)
    return df


def picture_rt(df):
    """Mean RT per picture, normalised within participant first."""
    grand = df["rt_ms"].mean()
    d = df.copy()
    d["rt_norm"] = d["rt_ms"] - d.groupby("subject")["rt_ms"].transform("mean") + grand
    return (d.groupby("target_pic")
              .agg(rt=("rt_norm", "mean"),
                   valence=("target_valence", "first"),
                   iaps=("target_valence_iaps", "first"),
                   n_as_target=("rt_norm", "size"))
              .reset_index())


def availability(df):
    """Count target and distractor appearances in the retained trials.
    """
    tgt = df["target_pic"].value_counts()
    dis = df["distractor_pic"].value_counts()
    return tgt.add(dis, fill_value=0).astype(int)


def corr_block(label, x, y):
    """Pearson plus Spearman. Spearman guards against one or two pictures
    dragging the line; if the two disagree, do not trust the Pearson value."""
    if len(x) < 4:
        say(f"    {label:<42} too few pictures ({len(x)})")
        return
    r, p_r = scipy_stats.pearsonr(x, y)
    rho, p_s = scipy_stats.spearmanr(x, y)
    flag = "  <-- significant" if min(p_r, p_s) < 0.05 else ""
    say(f"    {label:<42} r={r:+.3f} (p={p_r:.4f})   rho={rho:+.3f} (p={p_s:.4f})   "
        f"n={len(x)}{flag}")


def uniformity(label, counts):
    """Compare target counts with equal counts using chi-square; report the coefficient of variation."""
    counts = np.asarray(counts, dtype=float)
    exp = counts.mean()
    chi2 = ((counts - exp) ** 2 / exp).sum()
    dof = len(counts) - 1
    p = scipy_stats.chi2.sf(chi2, dof)
    cv = counts.std(ddof=1) / counts.mean()
    say(f"    {label:<42} CV={cv:.3f}  chi2({dof})={chi2:.1f}  p={p:.4g}")


# === Load data and summarize pictures ===
cued = load(CUEDTASK_CSV)
dp   = load(DUALPIC_CSV)

pc_cued = picture_rt(cued)
pc_dp   = picture_rt(dp)

avail_cued = availability(cued)
avail_dp   = availability(dp)

# One row per picture, both tasks side by side
pics = pc_cued.rename(columns={"rt": "rt_cued", "n_as_target": "n_cued"}).merge(
    pc_dp[["target_pic", "rt", "n_as_target"]].rename(
        columns={"rt": "rt_dp", "n_as_target": "n_dp"}),
    on="target_pic", how="inner")
pics["avail_cued"] = pics["target_pic"].map(avail_cued).astype(float)
pics["avail_dp"]   = pics["target_pic"].map(avail_dp).astype(float)
pics["rate_cued"]  = pics["n_cued"] / pics["avail_cued"]
pics["rate_dp"]    = pics["n_dp"]   / pics["avail_dp"]
pics = pics.sort_values("rt_cued").reset_index(drop=True)
pics["rank"] = np.arange(1, len(pics) + 1)

say("=" * 78)
say("Selection bias check for H3a")
say("=" * 78)
say(f"  pictures present in both tasks: {len(pics)} "
    f"({(pics.valence == 'positive').sum()} pleasant, {(pics.valence == 'negative').sum()} unpleasant)")
say(f"  CuedTask   {cued['subject'].nunique()} participants, {len(cued)} trials")
say(f"  DualPicture {dp['subject'].nunique()} participants, {len(dp)} trials")
say("  RT per picture is normalised within participant before averaging.")

# === Compare target-count distributions ===
say("\n  [1] Target-count uniformity")
say("      (CuedTask is experimenter-controlled and should be close to flat;")
say("       any strong deviation in DualPicture is participant choice)")
uniformity("CuedTask, times cued", pics["n_cued"])
uniformity("DualPicture, times chosen", pics["n_dp"])

# === Correlate Cued RT with target-selection rates ===
say("\n  [2] Cued RT vs Dual Picture selection rate")
say("      x = picture's mean RT in the CuedTask, y = selection rate in DualPicture")
corr_block("all pictures", pics["rt_cued"], pics["rate_dp"])
for val, lab in [("positive", "pleasant only"), ("negative", "unpleasant only")]:
    sub = pics[pics["valence"] == val]
    corr_block(lab + "  (valence held constant)", sub["rt_cued"], sub["rate_dp"])
say(" Within-valence results account for category differences; they do not establish causation.")

say("\n  [3] Control: Cued RT vs cueing rate")
corr_block("all pictures", pics["rt_cued"], pics["rate_cued"])

# === Test participant negative-target shares against 0.5 ===
say("\n  [5] Negative-target share by participant (test against 50%)")
for name, d in [("CuedTask (cued)", cued), ("DualPicture (chosen)", dp)]:
    per_subj = d.groupby("subject")["target_valence"].apply(lambda s: (s == "negative").mean())
    t, p = scipy_stats.ttest_1samp(per_subj.values, 0.5)
    say(f"    {name:<42} {100*per_subj.mean():.1f}%  "
        f"t({len(per_subj)-1})={t:+.2f}  p={p:.4f}")

say()


# === Plot picture RTs and target counts ===
fig, axes = plt.subplots(3, 1, figsize=(ps.W, 7.4), sharex=True,
                         gridspec_kw={"height_ratios": [2.4, 1, 1]})
x = pics["rank"].to_numpy()
colours = pics["valence"].map({"positive": C_POS, "negative": C_NEG})

ax = axes[0]
ax.scatter(x, pics["rt_cued"], marker="o", s=46, color=C_CUED, zorder=3,
           edgecolor="white", linewidth=0.7, label=f"{TASK_CUED} (experimenter cues)")
ax.scatter(x, pics["rt_dp"], marker="s", s=46, color=C_DP, zorder=3, alpha=0.9,
           edgecolor="white", linewidth=0.7, label=f"{TASK_DP} (participant chooses)")
ax.set_ylabel("Mean RT per picture (ms)\nnormalised within participant")
ax.grid(color="#E9E9E9", lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

for ax, col, lab in [(axes[1], "n_dp",   f"{TASK_DP}\ntimes chosen"),
                     (axes[2], "n_cued", f"{TASK_CUED}\ntimes cued")]:
    expected = pics[col].mean()
    ax.bar(x, pics[col], color=colours, width=0.85)
    ax.axhline(expected, color="#444", ls="--", lw=1.2)
    ax.set_ylabel(lab)
    ax.grid(axis="y", color="#E9E9E9", lw=0.8)
    ax.set_axisbelow(True)
    # Show valence colors and the mean-count reference in both count panels
    ax.legend(handles=[Patch(facecolor=C_POS, label="pleasant"),
                       Patch(facecolor=C_NEG, label="unpleasant"),
                       Line2D([0], [0], color="#444", ls="--", lw=1.2,
                              label=f"mean = {expected:.0f}")],
              loc="upper left", fontsize=9, framealpha=0.95, ncol=3)

axes[2].set_xlabel(f"Pictures, sorted by their mean reaction time in the {TASK_CUED} "
                   "(left = fastest, right = slowest)")
axes[2].set_xlim(0.2, len(pics) + 0.8)

plt.tight_layout()
ps.save(fig, FIG_DIR / "selection_bias_h3a.png", verbose=False)
say("saved selection_bias_h3a.png")

# === Save diagnostic report ===
result_path = RESULTS_DIR / "selection_bias_h3a.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"results saved: {result_path}")
