""" 
Single Picture Task - Behavioral Preprocessing

Steps:
1) Load CSV data; validate picture IDs and congruency
2) Remove practice trials, picture 36, and missing RTs
3) Remove incorrect responses and subjects below 90% accuracy
4) Remove RT <150 ms and within-subject outliers (±2.5 SD)
5) Exclude subjects with <70% valid trials
6) Log-transform RT and standardize retained-trial order
7) Save the trial table, diagnostic plots, and print summaries

Picture mapping: 1–35 unchanged; 36 excluded; 37–88 shifted down by one.
Practice rows: 1–4 and 45–48 (one-based); two columns per subject.
No distractor, side, or gaze variables are used.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR    = Path(__file__).parent
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Shared stimulus mapping and plot style
sys.path.insert(0, str(DATA_DIR.parent))
from stimulus_map import POSITIVE_IDS, NEGATIVE_IDS, IAPS_LOOKUP, SEQ_TO_IAPS

import plot_style as ps

# === Filtering settings ===
N_SD              = 2.5      # per-subject outlier threshold
MIN_VALID_PCT     = 70.0     # Exclude subject below this valid-trial percentage
ACCURACY_FILTER   = True
ACCURACY_MIN_PCT  = 90.0     # exclude subject if accuracy < this (Koch 2025)
DROP_ERROR_TRIALS = True     # remove incorrect-response trials
MIN_RT_MS         = 150.0    # anticipatory-response floor

# === Task structure ===
N_TRIALS_PER_COL = 88
N_BLOCKS         = 2
TEST_TRIAL_ROWS  = set(range(0, 4)) | set(range(44, 48))   # 0-based
HALF_SPLIT_ROW   = 44        # rows 0-43 = first instruction, 44-87 = second
INVALID_PIC_ID   = 36        # the picture the other two tasks do not use
CONG_FIRST_ORDERS = {1, 3}   # blockOrder values that start congruent


def single_id_to_seq(pic_id):
    """Map task IDs to shared sequence IDs; return NaN for picture 36."""
    if pic_id == INVALID_PIC_ID:
        return np.nan
    return pic_id if pic_id < INVALID_PIC_ID else pic_id - 1


def round_code_pos(iaps_val):
    """Map rounded positive valence: <=7 to -1, 8 to 0, >=9 to +1."""
    v = float(iaps_val)
    if np.isnan(v):
        return np.nan
    r = int(round(v))
    if r <= 7:
        return -1.0
    elif r == 8:
        return 0.0
    return 1.0


def round_code_neg(iaps_val):
    """Map rounded negative valence: <=1 to +1, 2 to 0, >=3 to -1."""
    v = float(iaps_val)
    if np.isnan(v):
        return np.nan
    r = int(round(v))
    if r <= 1:
        return 1.0
    elif r == 2:
        return 0.0
    return -1.0


# === Load CSV data ===
print("\nLoading .csv files...")

REQUIRED = ["pictureSequence", "reactionTime", "correctReaction",
            "actualReaction", "blockOrder"]
raw = {}
for name in REQUIRED:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    raw[name] = pd.read_csv(path, header=None)
    print(f"  {name:18} shape={raw[name].shape}")

pic_seq   = raw["pictureSequence"].values
rt_sec    = raw["reactionTime"].values.astype(float)
correct_r = raw["correctReaction"].values.astype(str)
actual_r  = raw["actualReaction"].values.astype(str)
block_ord = raw["blockOrder"].values.ravel().astype(int)

N_COLS = pic_seq.shape[1]
if pic_seq.shape[0] != N_TRIALS_PER_COL:
    raise SystemExit(f"expected {N_TRIALS_PER_COL} trials per column, got {pic_seq.shape[0]}")
for name, arr in [("reactionTime", rt_sec), ("correctReaction", correct_r),
                  ("actualReaction", actual_r)]:
    if arr.shape != pic_seq.shape:
        raise SystemExit(f"{name} shape {arr.shape} != pictureSequence {pic_seq.shape}")
if len(block_ord) != N_COLS:
    raise SystemExit(f"blockOrder has {len(block_ord)} entries, expected {N_COLS}")
if N_COLS % N_BLOCKS != 0:
    raise SystemExit(f"{N_COLS} columns is not divisible by {N_BLOCKS} blocks")

N_SUBJECTS = N_COLS // N_BLOCKS
print(f"  -> {N_SUBJECTS} participants x {N_BLOCKS} blocks = {N_COLS} columns")


# === Validate picture mapping and congruency ===
print("\nChecking picture IDs mapping against stimulus_map.py...")

present = sorted({int(v) for v in np.unique(pic_seq)})
if present != list(range(1, 89)):
    raise SystemExit(f"expected picture IDs 1..88, got {len(present)} distinct values")

mapped = sorted({int(single_id_to_seq(p)) for p in present if p != INVALID_PIC_ID})
assert mapped == list(range(1, 88)), \
    f"mapping must produce sequence IDs 1..87, got {len(mapped)} values"
assert set(mapped) == POSITIVE_IDS | NEGATIVE_IDS, \
    "mapped sequence IDs do not match stimulus_map.py"

# Check documented valence ranges against the shared mapping
doc_pos = {int(single_id_to_seq(p)) for p in range(1, 45) if p != INVALID_PIC_ID}
doc_neg = {int(single_id_to_seq(p)) for p in range(45, 89)}
assert doc_pos == POSITIVE_IDS, "pleasant range does not match POSITIVE_IDS after mapping"
assert doc_neg == NEGATIVE_IDS, "unpleasant range does not match NEGATIVE_IDS after mapping"
print(f"  sequence IDs 1..87 complete, no gaps, no duplicates")
print(f"  pleasant  {len(doc_pos)} pictures -> matches POSITIVE_IDS")
print(f"  unpleasant {len(doc_neg)} pictures -> matches NEGATIVE_IDS")
print(f"  picture ID {INVALID_PIC_ID} excluded ({int((pic_seq == INVALID_PIC_ID).sum())} trials)")

# Compare block-order congruency with the expected response
_pos_mask  = pic_seq <= 44
_cong_from_reaction = np.where(_pos_mask, correct_r == "pull", correct_r == "push")
_first_half_cong = np.isin(block_ord, list(CONG_FIRST_ORDERS))
_cong_from_order = np.empty_like(_cong_from_reaction)
_cong_from_order[:HALF_SPLIT_ROW, :] = _first_half_cong
_cong_from_order[HALF_SPLIT_ROW:, :] = ~_first_half_cong
_agree = float((_cong_from_reaction == _cong_from_order).mean()) * 100
print(f"  congruency from blockOrder vs from correctReaction: {_agree:.1f}% agreement")
if _agree < 99.9:
    raise SystemExit("congruency coding is inconsistent, do not trust blockOrder")


# === Build trial table; exclude practice trials and picture 36 ===
print(f"\nBuilding trial table "
      f"({N_SUBJECTS} subjects x {N_BLOCKS} blocks x {N_TRIALS_PER_COL} trials)...")

rows = []
n_test_trials   = 0
n_invalid_pic   = 0

for col in range(N_COLS):
    subject = col // N_BLOCKS + 1
    block   = col % N_BLOCKS + 1
    order   = int(block_ord[col])
    cong_first = order in CONG_FIRST_ORDERS

    for trial_idx in range(N_TRIALS_PER_COL):
        if trial_idx in TEST_TRIAL_ROWS:
            n_test_trials += 1
            continue

        pic_raw = int(pic_seq[trial_idx, col])
        seq_id  = single_id_to_seq(pic_raw)
        if np.isnan(seq_id):
            n_invalid_pic += 1
            continue
        seq_id = int(seq_id)

        in_first_half = trial_idx < HALF_SPLIT_ROW
        condition = ("congruent" if (in_first_half == cong_first) else "incongruent")

        if seq_id in POSITIVE_IDS:
            target_valence = "positive"
        elif seq_id in NEGATIVE_IDS:
            target_valence = "negative"
        else:
            target_valence = "unknown"

        iaps_id  = SEQ_TO_IAPS.get(seq_id, np.nan)
        iaps_val = IAPS_LOOKUP.get(seq_id, np.nan)

        if target_valence == "positive":
            target_intensity_e = round_code_pos(iaps_val)
        elif target_valence == "negative":
            target_intensity_e = round_code_neg(iaps_val)
        else:
            target_intensity_e = np.nan

        move_expected = str(correct_r[trial_idx, col]).strip().lower()
        move_done     = str(actual_r[trial_idx, col]).strip().lower()
        if move_expected in ("pull", "push") and move_done in ("pull", "push"):
            correct = 1.0 if move_expected == move_done else 0.0
        else:
            correct = np.nan

        rt_s = float(rt_sec[trial_idx, col])

        rows.append({
            "subject"             : subject,
            "block"               : block,
            "trial"               : trial_idx + 1,
            "block_order"         : order,
            "condition"           : condition,
            "cond_e"              : -0.5 if condition == "congruent" else 0.5,
            "target_pic"          : seq_id,
            "target_pic_single_id": pic_raw,
            "target_valence"      : target_valence,
            "target_valence_e"    : (-0.5 if target_valence == "positive"
                                     else (0.5 if target_valence == "negative" else np.nan)),
            "target_valence_iaps" : iaps_val,
            "target_iaps_id"      : iaps_id,
            "target_intensity_e"  : target_intensity_e,
            "movement_expected"   : move_expected,
            "movement_performed"  : move_done,
            "correct"             : correct,
            "rt_s"                : rt_s,
            "rt_ms"               : rt_s * 1000.0,
        })

df = pd.DataFrame(rows)
n_raw = len(df)

# Track remaining trials from the original total
n_raw_all      = N_COLS * N_TRIALS_PER_COL
n_after_test   = n_raw_all - n_test_trials
n_after_pic36  = n_after_test - n_invalid_pic
print(f"  Raw trials (all columns):       {n_raw_all}")
print(f"  Test trials skipped:            {n_test_trials}")
print(f"  Picture {INVALID_PIC_ID} trials skipped:        {n_invalid_pic}")
print(f"  Trials in table:                {n_raw}")
print(f"  Subjects:                       {df['subject'].nunique()}")
print(f"\n  condition distribution:          {df['condition'].value_counts().to_dict()}")
print(f"  target_valence distribution:     {df['target_valence'].value_counts().to_dict()}")
print(f"  target_intensity_e distribution: {df['target_intensity_e'].value_counts().sort_index().to_dict()}")

n_nan_rt = int(df["rt_ms"].isna().sum())
if n_nan_rt:
    df = df[df["rt_ms"].notna()].copy()
    print(f"  Removed {n_nan_rt} trials with missing RT")
n_after_table = len(df)


# === Apply accuracy exclusions ===
acc_low_subj  = []
n_err_removed = 0
n_after_acc   = None
if ACCURACY_FILTER and df["correct"].notna().any():
    subj_acc    = df.groupby("subject")["correct"].mean() * 100
    acc_low_subj = subj_acc[subj_acc < ACCURACY_MIN_PCT].index.tolist()
    overall_acc = df["correct"].mean() * 100

    print(f"\nAccuracy threshold: {ACCURACY_MIN_PCT}%")
    print(f"  Overall accuracy: {overall_acc:.1f}%")
    if overall_acc < 60.0:
        print("  WARNING: accuracy is close to chance; check the alignment of actualReaction")
        print("  with correctReaction.")
    for s in sorted(subj_acc.index):
        flag = "  <-- EXCLUDE" if subj_acc[s] < ACCURACY_MIN_PCT else ""
        print(f"    Subj {s:>3}: {subj_acc[s]:5.1f}%{flag}")

    n_err = int((df["correct"] == 0.0).sum())
    if DROP_ERROR_TRIALS:
        df = df[df["correct"] != 0.0].copy()
        n_err_removed = n_err
        print(f"  Removed {n_err} incorrect-response trials")
    if acc_low_subj:
        df = df[~df["subject"].isin(acc_low_subj)].copy()
        print(f"  Excluded {len(acc_low_subj)} subject(s) < {ACCURACY_MIN_PCT}%: {acc_low_subj}")
    n_after_acc = len(df)
    print(f"  After accuracy filter: {len(df)} trials, {df['subject'].nunique()} subjects")


# === Apply RT floor ===
print(f"\nRemove trials with RT < {MIN_RT_MS:.0f} ms")
before = len(df)
df = df[df["rt_ms"] >= MIN_RT_MS].copy()
n_after_cutoff = len(df)
print(f"  Removed: {before - len(df)} trials | Remaining: {len(df)}")


# === Remove within-subject RT outliers ===
print(f"\nPer-subject outlier removal (mean +/- {N_SD} SD)...")
stats = df.groupby("subject")["rt_ms"].agg(subj_mean="mean", subj_sd="std").reset_index()
df = df.merge(stats, on="subject")
df["outlier"] = (
    (df["rt_ms"] < df["subj_mean"] - N_SD * df["subj_sd"]) |
    (df["rt_ms"] > df["subj_mean"] + N_SD * df["subj_sd"])
)
n_out = int(df["outlier"].sum())
n_after_sd = len(df) - n_out
print(f"  Flagged: {n_out} trials ({100 * n_out / len(df):.1f}%)")


# === Exclude participants with insufficient valid trials ===
print(f"\nMinimum valid trials: {MIN_VALID_PCT}%")
EXPECTED_TRIALS_PER_SUBJ = N_BLOCKS * (N_TRIALS_PER_COL - len(TEST_TRIAL_ROWS))
kept_per_subj = df[~df["outlier"]].groupby("subject").size()
low_subj = []
for s in sorted(df["subject"].unique()):
    pct = 100 * kept_per_subj.get(s, 0) / EXPECTED_TRIALS_PER_SUBJ
    if pct < MIN_VALID_PCT:
        low_subj.append(s)
        print(f"    Subj {s:>3}: {pct:5.1f}%  <-- EXCLUDE")
print(f"  {len(low_subj)} subject(s) below {MIN_VALID_PCT}%"
      f"{': ' + str(low_subj) if low_subj else ''}")

df_clean = df[~df["outlier"] & ~df["subject"].isin(low_subj)].copy()
n_final = len(df_clean)
print(f"\n  Remaining after outlier removal + subject exclusion: {n_final} trials"
      f", {df_clean['subject'].nunique()} subjects")


# === Log-transform RT and standardize retained-trial order ===
print("\nLog transform and z_trial...")
df_clean["log_rt"] = np.log(df_clean["rt_ms"])
df_clean = df_clean.sort_values(["subject", "block", "trial"]).copy()
df_clean["trial_num"] = df_clean.groupby("subject").cumcount() + 1
df_clean["z_trial"] = df_clean.groupby("subject")["trial_num"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(f"  z_trial range: [{df_clean['z_trial'].min():.2f}, {df_clean['z_trial'].max():.2f}]")
print(f"  Mean RT (ms):  {df_clean['rt_ms'].mean():.1f}")
print(f"  Mean log(RT):  {df_clean['log_rt'].mean():.4f}")


# === Diagnostic plots ===
print("\nSaving diagnostic plots...")

# RT outliers by subject scatter 
fig, ax = plt.subplots(figsize=(14, 5))
colors = df["outlier"].map({False: "#4285F4", True: "#E8401C"})
kept_rt = df.loc[~df["outlier"], "rt_ms"]
y_cap = kept_rt.max() * 1.15
n_above = int((df["rt_ms"] > y_cap).sum())
ax.scatter(df["subject"], df["rt_ms"].clip(upper=y_cap), c=colors, s=5, alpha=0.45)
ax.set_ylim(0, y_cap)
ax.set_xlabel("Subject")
ax.set_ylabel("RT (ms)")
ax.set_title(
    f"Single Picture Task -- Per-subject outlier removal (mean +/- {N_SD} SD)\n"
    f"y-axis capped at {y_cap:.0f} ms  |  {n_above} extreme values clipped "
    f"(all flagged as outliers)")
ax.legend(handles=[mpatches.Patch(color="#4285F4", label="kept"),
                   mpatches.Patch(color="#E8401C", label="outlier")])
plt.tight_layout()
fig.savefig(RESULTS_DIR / "singlepicture_01_outlier_removal.png", dpi=150)
plt.close()
print("  Saved: singlepicture_01_outlier_removal.png")

# Cleaned RT and log-RT distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(df_clean["rt_ms"], bins=60, color="#4285F4", edgecolor="white", alpha=0.85)
axes[0].set_title("RT (ms) -- after cleaning")
axes[0].set_xlabel("RT (ms)")
axes[1].hist(df_clean["log_rt"], bins=60, color="#E8401C", edgecolor="white", alpha=0.85)
axes[1].set_title("log(RT) -- after transform")
axes[1].set_xlabel("log(RT)")
for a in axes:
    a.set_ylabel("Trials")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "singlepicture_02_rt_distribution.png", dpi=150)
plt.close()
print("  Saved: singlepicture_02_rt_distribution.png")

# Remaining trials after each filtering step
funnel_vals   = [n_raw_all, n_after_test, n_after_pic36]
funnel_labels = ["Raw", "Test trials", f"Picture {INVALID_PIC_ID}"]
if n_nan_rt:
    funnel_vals.append(n_after_table)
    funnel_labels.append("NaN / no response")
if n_after_acc is not None:
    funnel_vals.append(n_after_acc)
    funnel_labels.append(f"Error trials\n(+ subj < {ACCURACY_MIN_PCT:.0f}% accuracy)")
funnel_vals   += [n_after_cutoff, n_after_sd]
funnel_labels += [f"RT < {MIN_RT_MS:.0f} ms", f"±{N_SD} SD outliers"]
if n_final != n_after_sd:
    funnel_vals.append(n_final)
    funnel_labels.append(f"Subjects < {MIN_VALID_PCT:.0f}% valid")

fig, ax = plt.subplots(figsize=(ps.W, 4.6))
bars = ax.bar(range(len(funnel_vals)), funnel_vals,
              color=ps.C_BAR, edgecolor="white", width=0.6)
for b, v in zip(bars, funnel_vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center",
            va="bottom", fontsize=9)
ax.set_xticks(range(len(funnel_labels)))
ax.set_xticklabels(funnel_labels, fontsize=9, rotation=30, ha="right")
ax.set_ylabel("Number of trials")

ax.set_ylim(0, max(funnel_vals) * 1.12)
plt.tight_layout()
ps.save(fig, RESULTS_DIR / "singlepicture_00_preprocessing_funnel.png", verbose=False)
print("  Saved: singlepicture_00_preprocessing_funnel.png")


# === Save trial table and report counts ===
out_csv = RESULTS_DIR / "singlepicture_trial_table_preprocessed.csv"
df_clean.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv.name}")
print(f"    {len(df_clean)} trials, {df_clean['subject'].nunique()} subjects, "
      f"{len(df_clean.columns)} columns")

print("\n" + "=" * 70)
print("EXCLUSION SUMMARY")
print("=" * 70)
print(f"  Raw trials ({N_COLS} columns x {N_TRIALS_PER_COL})        | {n_raw_all:>5}")
print(f"  After test-trial removal                | {n_after_test:>5}")
print(f"  After picture-{INVALID_PIC_ID} removal                 | {n_after_pic36:>5}")
if n_after_acc is not None:
    print(f"  After accuracy filter                   | {n_after_acc:>5} "
          f"({n_err_removed} error trials, {len(acc_low_subj)} subjects)")
print(f"  After RT<{MIN_RT_MS:.0f}ms removal                  | {n_after_cutoff:>5}")
print(f"  After mean+/-{N_SD}SD outlier removal      | {n_after_sd:>5}")
print(f"  Final                                   | {n_final:>5}")
print("=" * 70)
