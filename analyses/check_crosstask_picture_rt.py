"""
Cross-task comparison of picture RTs, congruency and valence effects.

Compute participant-normalised picture means and cross-task correlations.
Plot the picture-level correlations. Task effects are compared descriptively;
picture correlations include p-values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

ROOT  = Path(__file__).resolve().parents[1]
PATHS = {
    "Cued AAT":           ROOT / "CuedAAT" / "results" / "cuedtask_trial_table_preprocessed.csv",
    "Dual Picture Task":  ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv",
    "Single Picture Task": ROOT / "SinglePictureTask" / "results" / "singlepicture_trial_table_preprocessed.csv",
}
RESULTS_DIR = ROOT / "results_combined"
FIG_DIR     = ROOT / "thesis_figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

for name, p in PATHS.items():
    if not p.exists():
        raise SystemExit(f"missing preprocessed csv for {name}: {p}\n"
                         f"run that task's preprocessing script first")

# shared style for every figure, see plot_style.py
import plot_style as ps
from plot_style import C_POS, C_NEG, C_SIG as NAVY
DISPLAY = {"Cued AAT": ps.TASK_CUED,
           "Dual Picture Task": ps.TASK_DP,
           "Single Picture Task": ps.TASK_SP}

out = []


def say(line=""):
    print(line)
    out.append(line)


def load(name, path):
    df = pd.read_csv(path)
    # Align condition column names
    if "condition" not in df.columns and "instructed_condition" in df.columns:
        df = df.rename(columns={"instructed_condition": "condition"})
    df = df[df["target_valence"].isin(["positive", "negative"])].copy()
    df["subject"] = df["subject"].astype(str)
    return df


def picture_rt(df):
    """Centre RT within participant, restore the grand mean, then average by picture."""
    grand = df["rt_ms"].mean()
    d = df.copy()
    d["rt_norm"] = d["rt_ms"] - d.groupby("subject")["rt_ms"].transform("mean") + grand
    return (d.groupby("target_pic")
              .agg(rt=("rt_norm", "mean"),
                   valence=("target_valence", "first"))
              .reset_index())


def subject_effect(df, col, hi, lo):
    """Return the mean, SE and participant count for paired hi-minus-lo RTs."""
    g = df.groupby(["subject", col])["rt_ms"].mean().unstack()
    if hi not in g.columns or lo not in g.columns:
        return np.nan, np.nan, np.nan
    diff = (g[hi] - g[lo]).dropna()
    m = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return m, se, len(diff)


data = {name: load(name, p) for name, p in PATHS.items()}

say("CROSS-TASK COMPARISON  (descriptive only, three different samples)")
for name, df in data.items():
    say(f"  {name:22} N={df['subject'].nunique():3}  trials={len(df):5}  "
        f"mean RT={df['rt_ms'].mean():7.1f} ms")

say("1. Congruency and valence effect per task (participant-level means)")

for name, df in data.items():
    cong, cong_se, n_c = subject_effect(df, "condition", "incongruent", "congruent")
    val,  val_se,  n_v = subject_effect(df, "target_valence", "negative", "positive")
    say(f"  {name:22} congruency = {cong:+7.1f} ms (SE {cong_se:5.1f})   "
        f"valence (neg-pos) = {val:+6.1f} ms (SE {val_se:5.1f})   n = {n_c}")

say("  Between-task effect differences are not tested.")

say("2. Picture-level RT stability across tasks")

pics = {name: picture_rt(df).set_index("target_pic") for name, df in data.items()}

say("\n  Pearson and Spearman on the 87 picture means:")
pair_stats = {}
for (n1, a), (n2, b) in itertools.combinations(pics.items(), 2):
    j = pd.concat([a["rt"].rename("a"), b["rt"].rename("b")], axis=1).dropna()
    r, p_r = scipy_stats.pearsonr(j["a"], j["b"])
    rho, p_s = scipy_stats.spearmanr(j["a"], j["b"])
    pair_stats[(n1, n2)] = (r, p_r, rho, p_s, len(j))
    say(f"    {n1:20} vs {n2:20}  r={r:+.3f} (p={p_r:.2e})  "
        f"rho={rho:+.3f} (p={p_s:.2e})  n={len(j)}")

say("\n  Within valence group (removes the pleasant/unpleasant confound):")
for (n1, a), (n2, b) in itertools.combinations(pics.items(), 2):
    for grp in ["positive", "negative"]:
        ia = a[a["valence"] == grp]["rt"].rename("a")
        ib = b[b["valence"] == grp]["rt"].rename("b")
        j = pd.concat([ia, ib], axis=1).dropna()
        if len(j) < 4:
            continue
        r, p_r = scipy_stats.pearsonr(j["a"], j["b"])
        say(f"    {n1[:12]:12} vs {n2[:12]:12}  {grp:9}  r={r:+.3f} (p={p_r:.4f})  n={len(j)}")


fig, axes = plt.subplots(1, 3, figsize=(ps.W, 3.6))
for ax, ((n1, a), (n2, b)) in zip(axes, itertools.combinations(pics.items(), 2)):
    j = pd.concat([a["rt"].rename("a"), b["rt"].rename("b"),
                   a["valence"].rename("v")], axis=1).dropna()
    for grp, colour, lab in [("positive", C_POS, "pleasant"),
                             ("negative", C_NEG, "unpleasant")]:
        s = j[j["v"] == grp]
        ax.scatter(s["a"], s["b"], s=34, alpha=0.8, color=colour,
                   edgecolor="white", linewidth=0.7, label=lab, zorder=3)
    r, p_r, rho, p_s, n = pair_stats[(n1, n2)]
    xs = np.linspace(j["a"].min(), j["a"].max(), 50)
    sl, ic = np.polyfit(j["a"], j["b"], 1)
    ax.plot(xs, ic + sl * xs, color=NAVY, lw=2.1, zorder=4)
    ax.set_xlabel(f"{DISPLAY.get(n1, n1)}\nmean RT per picture (ms)")
    ax.set_ylabel(f"{DISPLAY.get(n2, n2)}\nmean RT per picture (ms)")
    ax.set_title(f"r = {r:+.2f}", loc="left", fontweight="bold", color=NAVY)
    ax.grid(color="#E4E4E4", lw=0.8)
    ax.set_axisbelow(True)
axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)
plt.tight_layout()
ps.save(fig, FIG_DIR / "crosstask_picture_rt.png", verbose=False)
say("\nsaved crosstask_picture_rt.png")

result_path = RESULTS_DIR / "crosstask_picture_rt.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\nresults saved: {result_path}")
