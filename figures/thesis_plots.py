"""
Thesis figures for the Cued AAT and the Dual Picture Task.

Plot model coefficients, participant effects and intensity effects from
preprocessed CSVs and saved model results. Save figures to thesis_figures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import re, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "thesis_figures")
os.makedirs(OUT, exist_ok=True)
CUED_CSV = f"{ROOT}/CuedAAT/results/cuedtask_trial_table_preprocessed.csv"
DP_CSV   = f"{ROOT}/DualPictureTask/results/dualpicture_trial_table_preprocessed.csv"
CUED_TXT = f"{ROOT}/CuedAAT/results/cuedtask_lmer_results.txt"
DP_TXT   = f"{ROOT}/DualPictureTask/results/dualpicture_lmer_results.txt"


import plot_style as ps
from plot_style import (C_CUED, C_DP, C_POS, C_NEG,
                        C_SIG, C_NS, C_CONG, C_INCONG,
                        C_LINES, C_ZERO,
                        TASK_CUED, TASK_DP)

LABELS = {
 "cond_e":"Congruency (incongruent > congruent)",
 "target_valence_e":"Target valence (negative > positive)",
 "target_intensity_e":"Target intensity (main effect)",
 "distractor_intensity_e":"Distractor intensity (main effect)",
 "target_side_e":"Target side (right > left)",
 "z_trial":"Trial number (learning effect)",
}
ORDER=["cond_e","target_valence_e","target_intensity_e","distractor_intensity_e",
       "target_side_e","z_trial"]

def parse_coeffs(txt):
    """Read fixed-effect estimates and CIs from a saved model summary."""
    s=open(txt,encoding="utf-8").read()
    m=re.search(r"Coef\.\s+Std\.Err\.\s+z\s+P>\|z\|\s+\[0\.025\s+0\.975\]\n-+\n(.*?)\nGroup Var", s, re.S)
    block=m.group(1); out={}
    for line in block.strip().split("\n"):
        parts=line.split()
        name=" ".join(parts[:-6]) if len(parts)>6 else parts[0]
        nums=parts[-6:]
        try:
            coef,se,z,p,lo,hi=[float(x) for x in nums]
        except ValueError:
            continue
        out[name]={"coef":coef,"se":se,"z":z,"p":p,"lo":lo,"hi":hi}
    return out

def forest(coeffs, fname):
    terms=[t for t in ORDER if t in coeffs]
    ys=np.arange(len(terms))[::-1]
    fig,ax=plt.subplots(figsize=ps.FIG_SINGLE)
    ax.axvline(0,color=C_ZERO,lw=1.1,ls="--",zorder=1)
    for y,t in zip(ys,terms):
        c=coeffs[t]; sig=c["p"]<0.05; col=C_SIG if sig else C_NS
        ax.plot([c["lo"],c["hi"]],[y,y],color=col,lw=2.4,zorder=2,solid_capstyle="round")
        ax.scatter([c["coef"]],[y],s=95,color=col,zorder=3,edgecolor="white",linewidth=1.1)
        ptxt = "p < .001" if c["p"]<0.001 else f"p = {c['p']:.3f}".replace("0.",".")
        # RT change per contrast, intensity step or 1 SD of trial number.
        pct = (np.exp(c["coef"]) - 1.0) * 100.0
        ax.text(1.02,y,f"b = {c['coef']:+.3f} ({pct:+.1f}%)   {ptxt}",
                transform=ax.get_yaxis_transform(),
                va="center",ha="left",fontsize=10.3,color="#333")
    ax.set_yticks(ys); ax.set_yticklabels([LABELS[t] for t in terms],fontsize=11.5)
    ax.set_xlabel("Effect on log(RT), coefficient with 95% CI\n"
                  "(% = RT change per one coefficient unit: a full contrast for the\n"
                  "±0.5-coded terms, one intensity step, or 1 SD for z_trial)")
    ax.set_xlim(min(-0.05,min(coeffs[t]["lo"] for t in terms)-0.02),
               max(0.14, max(coeffs[t]["hi"] for t in terms)+0.02))
    ax.margins(y=0.08); ax.grid(axis="y",visible=False)
    leg=[Patch(facecolor=C_SIG,label="significant (p < .05)"),
         Patch(facecolor=C_NS,label="not significant")]
    ax.legend(handles=leg,loc="lower right",frameon=True,fontsize=10,framealpha=.95)
    plt.subplots_adjust(right=0.62,left=0.30,top=0.9,bottom=0.12)
    ps.save(fig, f"{OUT}/{fname}")

def ws_ci(df, cond, val="rt_ms", subj="subject"):
    """Compute normalised within-subject CIs with the Morey correction."""
    cell=df.groupby([subj,cond])[val].mean().reset_index()
    sm=cell.groupby(subj)[val].transform("mean"); grand=cell[val].mean()
    cell["norm"]=cell[val]-sm+grand
    g=cell.groupby(cond)["norm"].agg(mean="mean", sem=lambda x:x.std(ddof=1)/np.sqrt(len(x)))
    M=cell[cond].nunique(); corr=np.sqrt(M/(M-1))
    g["ci95"]=g["sem"]*corr*1.959964
    return g, cell


# Retain all designed levels, including empty bins
INT_LEVELS = [-1.0, 0.0, 1.0]
INT_NAMES  = {-1.0: "low", 0.0: "medium", 1.0: "high"}


def intensity_axis(ax, dfs, col):
    """Draw the three designed intensity levels, greying out any that hold no
    pictures. Returns the levels that are actually populated."""
    present = set()
    for d in dfs:
        present |= set(d[col].dropna().unique())
    labels = [INT_NAMES[v] if v in present else f"{INT_NAMES[v]}\n(no pictures)"
              for v in INT_LEVELS]
    for v in INT_LEVELS:
        if v not in present:
            ax.axvspan(v-0.45, v+0.45, color=ps.C_EMPTY, zorder=0)
    ax.set_xticks(INT_LEVELS); ax.set_xticklabels(labels)
    ax.set_xlim(-1.5, 1.5)
    return [v for v in INT_LEVELS if v in present]


def ws_ci_geom(df, cond, balance=None, val="log_rt", subj="subject"):
    """Normalise log RT within subject, then back-transform means and CIs.

    If requested, weight available balance cells equally within each condition.
    """
    keys = [subj, cond] + ([balance] if balance else [])
    cell = df.groupby(keys)[val].mean().reset_index()
    if balance:
        cell = cell.groupby([subj, cond])[val].mean().reset_index()
    sm = cell.groupby(subj)[val].transform("mean"); grand = cell[val].mean()
    cell["norm"] = cell[val] - sm + grand
    g = cell.groupby(cond)["norm"].agg(mean="mean",
                                       sem=lambda x: x.std(ddof=1)/np.sqrt(len(x)))
    M = cell[cond].nunique()
    g["ci95"] = g["sem"] * (np.sqrt(M/(M-1)) if M > 1 else 1.0) * 1.959964
    g["geo"] = np.exp(g["mean"])
    g["lo"]  = np.exp(g["mean"] - g["ci95"])
    g["hi"]  = np.exp(g["mean"] + g["ci95"])
    return g


def intensity_series(ax, df, col, levels, label, color, ls, mk, balance="target_valence"):
    
    if balance and df[balance].nunique() < 2:
        balance = None
    g = ws_ci_geom(df, col, balance=balance)
    xs = [v for v in levels if v in g.index]
    if not xs:
        return
    ys  = [g.loc[v, "geo"] for v in xs]
    lo  = [g.loc[v, "geo"] - g.loc[v, "lo"] for v in xs]
    hi  = [g.loc[v, "hi"]  - g.loc[v, "geo"] for v in xs]
    ax.errorbar(xs, ys, yerr=[lo, hi], label=label, color=color, ls=ls, marker=mk,
                capsize=5, lw=2.3, ms=8, mec="white", mew=1.1)

def delta_inset(ax, cell, cond, hi, lo, color, loc):
    # small inset with the paired per-participant difference on its own scale 
    # the main panel spans ~700-1600ms, so a real ~16ms effect would otherwise
    # be invisible even when significant.
    piv = cell.pivot(index="subject", columns=cond, values="rt_ms")
    if hi not in piv.columns or lo not in piv.columns:
        return
    d = (piv[hi] - piv[lo]).dropna().values
    if len(d) < 3:
        return
    ins = ax.inset_axes(loc, zorder=6)
    ins.set_facecolor("white"); ins.patch.set_alpha(1.0)
    rng = np.random.default_rng(7)
    ins.scatter(rng.normal(0, 0.035, len(d)), d, s=11, color=color, alpha=.5,
                zorder=3, edgecolor="white", linewidth=.4)
    m = d.mean(); ci = 1.959964*d.std(ddof=1)/np.sqrt(len(d))
    ins.errorbar(0, m, yerr=ci, fmt="o", color="#1E2749", ms=6.5, capsize=5,
                 lw=2, zorder=4, mec="white", mew=1)
    ins.axhline(0, color=C_ZERO, ls="--", lw=1)
    ins.set_xticks([]); ins.set_xlim(-0.30, 0.30)
    ins.yaxis.tick_right(); ins.tick_params(axis="y", labelsize=8, pad=1.5)
    ins.yaxis.set_major_locator(plt.MaxNLocator(4))
    ins.set_title("per participant", fontsize=8.5, color="#444", pad=3)
    ins.grid(axis="y", alpha=.3)
    for s in ins.spines.values():
        s.set_color("#BBB")


def main_effects(df, fname):
    fig,axes=plt.subplots(1,2,figsize=ps.FIG_WIDE)
    order_c=["congruent","incongruent"]; labels_c=["congruent","incongruent"]
    g,cell=ws_ci(df[df["condition"].isin(order_c)],"condition")
    ax=axes[0]; piv=cell.pivot(index="subject",columns="condition",values="rt_ms")
    for _,r in piv.iterrows():
        if all(c in piv.columns for c in order_c):
            ax.plot([0,1],[r[order_c[0]],r[order_c[1]]],color=C_LINES,lw=0.7,alpha=.5,zorder=1)
    for i,c in enumerate(order_c):
        ax.errorbar(i,g.loc[c,"mean"],yerr=g.loc[c,"ci95"],fmt="o",ms=11,
            color=[C_CONG,C_INCONG][i],capsize=6,lw=2.4,zorder=3,mec="white",mew=1.3)
    ax.set_xticks([0,1]); ax.set_xticklabels(labels_c); ax.set_xlim(-0.4,1.4)
    ax.set_ylabel("Mean RT (ms)"); ax.set_title("Congruency effect")
    d=g.loc["incongruent","mean"]-g.loc["congruent","mean"]
    ax.text(.5,.02,f"delta = +{d:.0f} ms",transform=ax.transAxes,ha="center",fontsize=11,color="#333")
    delta_inset(ax, cell, "condition", "incongruent", "congruent",
                C_INCONG, [0.06, 0.62, 0.17, 0.30])
    order_v=["positive","negative"]; labels_v=["positive","negative"]
    g2,cell2=ws_ci(df[df["target_valence"].isin(order_v)],"target_valence")
    ax=axes[1]; piv2=cell2.pivot(index="subject",columns="target_valence",values="rt_ms")
    for _,r in piv2.iterrows():
        if all(c in piv2.columns for c in order_v):
            ax.plot([0,1],[r[order_v[0]],r[order_v[1]]],color=C_LINES,lw=0.7,alpha=.5,zorder=1)
    for i,c in enumerate(order_v):
        ax.errorbar(i,g2.loc[c,"mean"],yerr=g2.loc[c,"ci95"],fmt="o",ms=11,
            color=[C_POS,C_NEG][i],capsize=6,lw=2.4,zorder=3,mec="white",mew=1.3)
    ax.set_xticks([0,1]); ax.set_xticklabels(labels_v); ax.set_xlim(-0.4,1.4)
    ax.set_ylabel("Mean RT (ms)"); ax.set_title("Target valence effect")
    d2=g2.loc["negative","mean"]-g2.loc["positive","mean"]
    ax.text(.5,.02,f"delta = +{d2:.0f} ms",transform=ax.transAxes,ha="center",fontsize=11,color="#333")
    delta_inset(ax, cell2, "target_valence", "negative", "positive",
                C_NEG, [0.06, 0.62, 0.17, 0.30])
    plt.tight_layout()
    ps.save(fig, f"{OUT}/{fname}")

def val_delta(df):
    cell=df[df["target_valence"].isin(["positive","negative"])].groupby(
        ["subject","target_valence"])["rt_ms"].mean().unstack()
    return (cell["negative"]-cell["positive"]).dropna()

def combined_valence(dc, dd, fname):
    dcd=val_delta(dc).values; ddd=val_delta(dd).values
    fig,ax=plt.subplots(figsize=ps.FIG_SINGLE)
    data=[dcd,ddd]; colors=[C_CUED,C_DP]
    vp=ax.violinplot(data,positions=[0,1],widths=0.75,showmeans=False,showextrema=False)
    for pc,c in zip(vp["bodies"],colors):
        pc.set_facecolor(c); pc.set_alpha(.22); pc.set_edgecolor(c); pc.set_linewidth(1.2)
    rng=np.random.default_rng(3)
    for i,(vals,c) in enumerate(zip(data,colors)):
        x=rng.normal(i,0.045,len(vals))
        ax.scatter(x,vals,color=c,alpha=.55,s=28,zorder=3,edgecolor="white",linewidth=.5)
        m=vals.mean(); ci=1.959964*vals.std(ddof=1)/np.sqrt(len(vals))
        ax.errorbar(i,m,yerr=ci,fmt="o",color="#1E2749",ms=10,capsize=8,lw=2.2,zorder=4,mec="white",mew=1.2)
        ax.annotate(f"M = {m:+.0f} ms",(i,m),xytext=(14,0),textcoords="offset points",
                    va="center",fontsize=11.5,fontweight="bold",color=c)
    ax.axhline(0,color=C_ZERO,ls="--",lw=1.1,zorder=1)
    ax.set_xticks([0,1]); ax.set_xticklabels([f"{TASK_CUED}\n(N = {dc['subject'].nunique()})",
                                              f"{TASK_DP}\n(N = {dd['subject'].nunique()})"])
    ax.set_xlim(-0.6,1.6)
    ax.set_ylabel("Valence effect per participant:\nRT(negative) minus RT(positive)  [ms]")
    fig.tight_layout()
    ps.save(fig, f"{OUT}/{fname}")

def combined_intensity(dc, dd, fname):
    fig,axes=plt.subplots(1,2,figsize=ps.FIG_WIDE,sharey=True)
    n_c, n_d = dc["subject"].nunique(), dd["subject"].nunique()
    for ax,(col,lab) in zip(axes,[("target_intensity_e","Target intensity"),
                                  ("distractor_intensity_e","Distractor intensity")]):
        levels = intensity_axis(ax, [dc, dd], col)
        for df,name,n,c,ls,mk in [(dc,TASK_CUED,n_c,C_CUED,"-","o"),
                                  (dd,TASK_DP,n_d,C_DP,"--","s")]:
            intensity_series(ax, df, col, levels, f"{name} (N = {n})", c, ls, mk)
        ax.set_xlabel("Intensity (rounded IAPS, role-based)"); ax.set_title(lab)
    axes[0].set_ylabel("Geometric mean RT (ms)"); axes[0].legend(loc="best",frameon=True,fontsize=11)
    plt.tight_layout()
    ps.save(fig, f"{OUT}/{fname}")

# Load trials and model results
dc=pd.read_csv(CUED_CSV); dc["condition"]=dc["condition"].astype(str)
dd=pd.read_csv(DP_CSV)
dd=dd[dd["movement"].isin(["push","pull"])]
dd=dd[dd["congruency"].isin(["congruent","incongruent"])].copy()
dd=dd.rename(columns={"congruency":"condition"}); dd["condition"]=dd["condition"].astype(str)

cc=parse_coeffs(CUED_TXT); dpc=parse_coeffs(DP_TXT)

forest(cc, "thesis_cued_coefficients.png")
main_effects(dc, "thesis_cued_maineffects.png")
forest(dpc, "thesis_dp_coefficients.png")
main_effects(dd, "thesis_dp_maineffects.png")
combined_valence(dc, dd, "thesis_combined_valence.png")
combined_intensity(dc, dd, "thesis_combined_intensity.png")
print("done")
