"""
Compare target-intensity effects across the three tasks.

Compute valence-balanced participant contrasts and paired t-tests for the
Cued AAT, the Dual Picture Task and the Single Picture Task.

The categorical contrast does not test a continuous intensity gradient.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("Cued AAT",            ROOT / "CuedAAT" / "results" / "cuedtask_trial_table_preprocessed.csv"),
    ("Dual Picture Task",   ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"),
    ("Single Picture Task", ROOT / "SinglePictureTask" / "results" / "singlepicture_trial_table_preprocessed.csv"),
]
RESULTS_DIR = ROOT / "results_combined"
RESULTS_DIR.mkdir(exist_ok=True)

for name, path in TASKS:
    if not path.exists():
        raise SystemExit(f"missing preprocessed csv for {name}: {path}\n"
                         f"run that task's preprocessing script first")

out = []


def say(line=""):
    print(line)
    out.append(line)


def participant_level(df, term="target_intensity_e"):
    """Test paired log-RT means, weighting available valence cells equally."""
    d = df[df["target_valence"].isin(["positive", "negative"])]
    cell = d.groupby(["subject", term, "target_valence"])["log_rt"].mean().reset_index()
    cell = cell.groupby(["subject", term])["log_rt"].mean().reset_index()
    piv  = cell.pivot(index="subject", columns=term, values="log_rt").dropna()
    if piv.shape[1] != 2:
        raise SystemExit(f"expected exactly 2 occupied intensity levels, got {list(piv.columns)}")
    lo, hi = sorted(piv.columns)
    diff = piv[hi] - piv[lo]
    t_stat, p_val = scipy_stats.ttest_rel(piv[hi], piv[lo])
    pct = (np.exp(diff.mean()) - 1) * 100
    return pct, float(t_stat), len(diff) - 1, float(p_val), int((diff > 0).sum()), len(diff), diff


say("Target-intensity effects across tasks")
say("  Valence-balanced paired tests on log RT; higher minus lower occupied level.")
say("  Participant counts show positive differences / paired participants.")
say()

results = {}
for name, path in TASKS:
    df = pd.read_csv(path)
    pct, t, dfree, p, k, n, diff = participant_level(df)
    results[name] = dict(pct=pct, t=t, df=dfree, p=p, k=k, n=n, diff=diff)
    say(f"  {name:20} {pct:+6.2f}%   t({dfree:2d}) = {t:+5.2f}   p = {p:.4f}   "
        f"({k}/{n} participants)")

sp = results["Single Picture Task"]
say(f"\nSingle Picture effect: {sp['pct']:+.2f}%, p = {sp['p']:.4f}.")
say("  This tests the categorical contrast without a distractor, not a continuous gradient.")
say("  Differences between tasks are not tested here.")

result_path = RESULTS_DIR / "intensity_replication.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\nresults saved: {result_path}")
