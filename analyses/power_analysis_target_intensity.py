"""
Target-intensity power analysis for the Cued and Dual-Picture tasks.

1) Approximate sample size from the observed coefficient and SE
2) Repeat with half the observed effect as a sensitivity scenario
3) Estimate power by simulating and refitting random-intercept models
4) Save a text summary

Estimates depend on the fitted effects, variance components and trial design.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

ROOT   = Path(__file__).resolve().parents[1]
CUEDTASK_CSV = ROOT / "CuedAAT"                / "results" / "cuedtask_trial_table_preprocessed.csv"
DUALPIC_CSV  = ROOT / "DualPictureTask" / "results" / "dualpicture_trial_table_preprocessed.csv"
RESULTS_DIR  = ROOT / "results_combined"
RESULTS_DIR.mkdir(exist_ok=True)

for p in [CUEDTASK_CSV, DUALPIC_CSV]:
    if not p.exists():
        raise SystemExit(f"csv not found: {p} -- run preprocessing scripts first")

FORMULA = ("log_rt ~ cond_e + target_valence_e + target_intensity_e"
           " + distractor_intensity_e + target_side_e + z_trial")
TERM    = "target_intensity_e"
ALPHA   = 0.05

# 500 simulations per sample size; 3,500 refits per task.
N_SIM   = 500
N_GRID  = [15, 25, 40, 55, 80, 110, 150]
RNG     = np.random.default_rng(20260807)

out = []

def say(line=""):
    print(line)
    out.append(line)


def load(csv_path, cond_col):
    df = pd.read_csv(csv_path)
    if cond_col == "congruency":
        df = df[df["movement"].isin(["push", "pull"])].copy()
        df = df[df["congruency"].isin(["congruent", "incongruent"])].copy()
        df["cond_e"] = df["congruency"].map({"congruent": -0.5, "incongruent": 0.5})
    else:
        df = df[df["condition"].isin(["congruent", "incongruent"])].copy()
        df["cond_e"] = df["condition"].map({"congruent": -0.5, "incongruent": 0.5})
    df = df[df["target_valence"].isin(["positive", "negative"])].copy()
    df["subject"] = df["subject"].astype(str)
    return df


def fit(df):
    return smf.mixedlm(FORMULA, data=df, groups=df["subject"]).fit(
        reml=True, method="powell", maxiter=400)


def n_for_power(b, se, n_obs, power=0.80, alpha=ALPHA):
    """Approximate N assuming SE scales with 1/sqrt(N)."""
    z_obs    = abs(b) / se
    z_needed = scipy_stats.norm.ppf(1 - alpha / 2) + scipy_stats.norm.ppf(power)
    return n_obs * (z_needed / z_obs) ** 2, z_obs, z_needed


def simulate_power(df, m, n_subj, effect, n_sim=N_SIM):
    """Resample participant designs, simulate log RT and refit the model.
    Use fitted variance components and fixed effects, replacing only TERM.
    Exceptions count as misses; convergence flags are not checked.
    """
    params  = m.params
    sd_u    = float(np.sqrt(m.cov_re.iloc[0, 0]))
    sd_e    = float(np.sqrt(m.scale))
    terms   = [t for t in params.index if t not in ("Group Var", "Intercept")]
    subjects = [g for _, g in df.groupby("subject")]

    n_hit = 0
    for i in range(n_sim):
        picks = RNG.integers(0, len(subjects), size=n_subj)
        parts = []
        for new_id, k in enumerate(picks):
            block = subjects[k].copy()
            block["subject"] = f"s{new_id}"
            parts.append(block)
        sim = pd.concat(parts, ignore_index=True)

        mu = np.full(len(sim), params["Intercept"], dtype=float)
        for t in terms:
            beta = effect if t == TERM else params[t]
            mu += beta * sim[t].to_numpy(dtype=float)
        u = RNG.normal(0.0, sd_u, size=n_subj)
        sim["log_rt"] = mu + np.repeat(u, [len(p) for p in parts]) + RNG.normal(0.0, sd_e, size=len(sim))

        try:
            mm = smf.mixedlm(FORMULA, data=sim, groups=sim["subject"]).fit(
                reml=True, method="powell", maxiter=400)
            if mm.pvalues[TERM] < ALPHA:
                n_hit += 1
        except Exception:
            pass          # Fits raising an exception count as misses
        if (i + 1) % 50 == 0:
            print(f"      ... {i + 1}/{n_sim} runs at N={n_subj}")
    return n_hit / n_sim


def run_task(csv_path, cond_col, task_name):
    df = load(csv_path, cond_col)
    m  = fit(df)
    b  = float(m.params[TERM])
    se = float(m.bse[TERM])
    p  = float(m.pvalues[TERM])
    n_obs = df["subject"].nunique()

    say(f"{task_name}: N = {n_obs} participants, {len(df)} trials")
    say(f"  {TERM}: b = {b:+.4f}, SE = {se:.4f}, p = {p:.4f}")
    say(f"  RT magnitude, 100 * (exp(|b|) - 1): {100 * (np.exp(abs(b)) - 1):.2f}% per intensity step")
    say(f"  SD: random intercept = {np.sqrt(m.cov_re.iloc[0,0]):.4f}, "
        f"residual = {np.sqrt(m.scale):.4f}")

    say("\n  Approximate N (same trials per participant)")
    for power in (0.80, 0.90, 0.95):
        n_need, z_obs, z_needed = n_for_power(b, se, n_obs, power)
        say(f"    {int(power*100)}% power: N = {int(np.ceil(n_need)):>4}   "
            f"(observed |z| = {z_obs:.2f}, needed |z| = {z_needed:.2f})")

    say("\n  Sensitivity scenario: half the observed effect")
    for power in (0.80, 0.90):
        n_need, _, _ = n_for_power(b / 2.0, se, n_obs, power)
        say(f"    {int(power*100)}% power: N = {int(np.ceil(n_need)):>4}")

    say(f"\n  Simulated power: {N_SIM} runs per N, true effect fixed at the observed b")
    say("    N     power")
    for n_subj in N_GRID:
        pw = simulate_power(df, m, n_subj, b)
        mark = "  <-- observed sample" if n_subj == n_obs else ""
        say(f"    {n_subj:>4}   {pw:.3f}{mark}")

    say()


run_task(CUEDTASK_CSV, "condition", "Cued AAT")
run_task(DUALPIC_CSV, "congruency", "Dual Picture Task")

result_path = RESULTS_DIR / "power_analysis_target_intensity.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\nresults saved: {result_path}")
