"""
    Cued AAT - preprocessing
    
    Steps:
    1) Load trial and gaze data; code valence and intensity
    2) Exclude trials using fixation and post-onset gaze checks (Koch, 2025)
    3) Drop error trials and subjects below 90% accuracy
    4) Remove RT < 150ms and +/-2.5 SD outliers
    5) Exclude subjects with <70% valid trials after all exclusions 
    6) Log-transform RT, code target side, add z_trial, make plots
    7) Save the preprocessed trial table
    
 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
from pathlib import Path

# === Settings ===

DATA_DIR = Path(__file__).parent
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SD = 2.5             # within-subject RT cutoff (SD)
MIN_VALID_PCT = 70.0   # exclude subjects below this percentage

# Accuracy exclusion (Koch 2025)

ACCURACY_FILTER   = True
ACCURACY_MIN_PCT  = 90.0
DROP_ERROR_TRIALS = True

# shared stimulus mapping and plot style
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from stimulus_map import POSITIVE_IDS, NEGATIVE_IDS, IAPS_LOOKUP

# Shared style for every figure in the thesis, see plot_style.py
import plot_style as ps

# === Intensity coding from rounded IAPS valence ===
def round_code_pos(iaps_val):
    """Map rounded positive valence: -1 = least positive (rounded <=7), 0 = medium (8), +1 = most positive (9)."""
    v = float(iaps_val)
    if np.isnan(v):
        return np.nan
    r = int(round(v))
    if r <= 7:
        return -1.0
    elif r == 8:
        return 0.0
    else:
        return 1.0


def round_code_neg(iaps_val):
    """Map rounded negative valence: +1 = most negative (rounded 1), 0 = medium (2), -1 = least negative (rounded >=3)."""
    v = float(iaps_val)
    if np.isnan(v):
        return np.nan
    r = int(round(v))
    if r <= 1:
        return 1.0
    elif r == 2:
        return 0.0
    else:
        return -1.0


# === Eye-tracking trial exclusion (Koch 2025) ===

ET_EXCLUSION_ENABLED  = True
VALID_ET_IDX          = list(range(5, 49)) + list(range(54, 98))   # 88 valid ET indices, 0-based
# also drop a trial if the first dwell is on the uncued side (looked at the
# distractor, not the target)
EXCLUDE_WRONG_DIRECTION = True
SCREEN_CENTER_PX      = 960    # px
CROSS_ROI_HALF        = 100    # Fixation cross ROI: center +/- 100 px = 860-1060 px
LEAVE_SAMPLES         = 50     # End dwell after >50 samples outside
VEL_THRESHOLD         = 5      # Saccade velocity threshold (px/sample)
MIN_PIC_STABLE        = 20     # Stability window after picture onset (samples); at least 80% valid 
MAX_VAR_PX            = 50     # Max positional variability within stability window (px)
POST_ONSET_MS         = 3000   # Search window after picture onset (ms)
EARLY_SACCADE_MS      = 100    # Exclude first detected dwell at <=100ms


def find_trial_center(x_win):
    """Estimate gaze center from the longest center-band dwell; use a median fallback."""
    n = len(x_win)
    if n < 5:
        return float(SCREEN_CENTER_PX)
    in_roi = (~np.isnan(x_win)) & (np.abs(x_win - SCREEN_CENTER_PX) <= CROSS_ROI_HALF)
    if not in_roi.any():
        valid_x = x_win[~np.isnan(x_win)]
        if len(valid_x) == 0:
            return float(SCREEN_CENTER_PX)
        fb = float(np.median(valid_x))
        in_roi2 = (~np.isnan(x_win)) & (np.abs(x_win - fb) <= CROSS_ROI_HALF)
        return float(np.nanmedian(x_win[in_roi2])) if in_roi2.any() else fb
    best_start = best_end = 0
    best_len   = 0
    cur_start  = None
    out_count  = 0
    for i in range(n):
        if in_roi[i]:
            if cur_start is None:
                cur_start = i
            out_count = 0
        else:
            if cur_start is not None:
                out_count += 1
                if out_count > LEAVE_SAMPLES:
                    end = i - out_count
                    if end - cur_start > best_len:
                        best_len = end - cur_start
                        best_start, best_end = cur_start, end
                    cur_start = None
                    out_count = 0
    if cur_start is not None:
        end = n - 1
        if end - cur_start + 1 > best_len:
            best_start, best_end = cur_start, end
    return float(np.nanmedian(x_win[best_start : best_end + 1]))


def had_premature_gaze(x_arr, t_arr, cross_onset_ms, picture_onset_ms, center_x=SCREEN_CENTER_PX):
    """Detect a stable shift from the center band during fixation."""
    mask = (t_arr >= cross_onset_ms) & (t_arr < picture_onset_ms)
    if mask.sum() < MIN_PIC_STABLE + 2:
        return False
    x_win = x_arr[mask]
    n = len(x_win)
    for i in range(n - MIN_PIC_STABLE - 1):
        if np.isnan(x_win[i]) or np.isnan(x_win[i + 1]):
            continue
        if (abs(x_win[i + 1] - x_win[i]) > VEL_THRESHOLD
                and abs(x_win[i] - center_x) <= CROSS_ROI_HALF        # starts in center
                and abs(x_win[i + 1] - center_x) > CROSS_ROI_HALF):  # lands outside center
            seg   = x_win[i + 1 : i + 1 + MIN_PIC_STABLE]
            valid = seg[~np.isnan(seg)]
            if (len(valid) >= int(MIN_PIC_STABLE * 0.8)
                    and np.max(valid) - np.min(valid) <= MAX_VAR_PX):
                return True
    return False


def has_picture_dwell(x_arr, t_arr, picture_onset_ms, center_x):
    """Return the first stable lateral dwell: left, right, early (<=100ms), or none."""
    mask = (t_arr >= picture_onset_ms) & (t_arr < picture_onset_ms + POST_ONSET_MS)
    if mask.sum() < MIN_PIC_STABLE + 2:
        return "none"
    x_win = x_arr[mask]
    t_win = t_arr[mask]
    n     = len(x_win)
    for i in range(n - MIN_PIC_STABLE - 1):
        if np.isnan(x_win[i]) or np.isnan(x_win[i + 1]):
            continue
        if abs(x_win[i + 1] - x_win[i]) > VEL_THRESHOLD:
            pos_x = x_win[i + 1]
            if pos_x < center_x - CROSS_ROI_HALF:
                side = "left"
            elif pos_x > center_x + CROSS_ROI_HALF:
                side = "right"
            else:
                continue  # velocity spike but still inside center band
            seg   = x_win[i + 1 : i + 1 + MIN_PIC_STABLE]
            valid = seg[~np.isnan(seg)]
            if (len(valid) >= int(MIN_PIC_STABLE * 0.8)
                    and np.max(valid) - np.min(valid) <= MAX_VAR_PX):
                if t_win[i] - picture_onset_ms <= EARLY_SACCADE_MS:
                    return "early"   # first stable saccade is anticipatory -> exclude
                return side   # "left" or "right"
    return "none"


N_BLOCKS     = 2
N_TRIALS     = 88    # 44 congruent + 44 incongruent
FIRST_SUB_ID = 48    # First participant ID

# === Load behavioral and eye-tracking data ===

print("Cued AAT -- Preprocessing")
print(f"\nData directory: {DATA_DIR.resolve()}")
print(f"Results:        {RESULTS_DIR.resolve()}")
print(f"\n Loading .mat files...")

def load_mat(name):
    """Load a numeric variable from a .mat file."""
    path = DATA_DIR / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(
            f"\n  ERROR: {path} not found.\n"
            f"  Please ensure all .mat files are in:\n  {DATA_DIR.resolve()}"
        )
    data = sio.loadmat(str(path))
    key  = [k for k in data if not k.startswith("_")][0]
    return data[key]


def load_block_order():
    """Load block order (A/B per column) from exported CSV."""
    bo_csv = DATA_DIR / "results" / "block_order.csv"
    df = pd.read_csv(bo_csv)
    orders = df.sort_values("col_idx")["order"].tolist()
    print(f"  blockOrder: {len(orders)} values from block_order.csv")
    return orders


reaction_time      = load_mat("reactionTime")
cue_side           = load_mat("cueSide")
picture_sequence   = load_mat("pictureSequence")

# Decode push/pull strings from compressed MATLAB MCOS data
def _decode_reaction_performed(path, n_rows, n_cols):
    import re as _re, zlib as _zlib
    raw = open(path, "rb").read()
    streams = [i for i in range(len(raw) - 1)
               if raw[i] == 0x78 and raw[i + 1] in (0x9C, 0xDA, 0x01, 0x5E)]
    big = ""
    for pos in streams:
        try:
            dec = _zlib.decompress(raw[pos:])
        except Exception:
            continue
        for m in _re.finditer(rb"(?:[\x20-\x7e]\x00){8,}", dec):
            ss = m.group().decode("utf-16-le")
            if _re.fullmatch("(push|pull)+", ss) and len(ss) > len(big):
                big = ss
    toks = _re.findall("push|pull", big)
    if len(toks) < n_rows * n_cols:
        return None
    # Restore MATLAB column-major trial order
    
    return np.array(toks[:n_rows * n_cols], dtype=object).reshape(n_cols, n_rows).T

reaction_performed = _decode_reaction_performed(
    DATA_DIR / "reactionPerformed.mat", reaction_time.shape[0], reaction_time.shape[1])
if reaction_performed is None:
    print("  WARNING: reactionPerformed decode failed - accuracy filter disabled")
    ACCURACY_FILTER = False
else:
    print(f"  reactionPerformed decoded OK: {reaction_performed.shape}")
block_order_list   = load_block_order()

N_COLS     = reaction_time.shape[1]
N_SUBJECTS = N_COLS // N_BLOCKS

# print data dimensions
print(f"  reactionTime:    {reaction_time.shape}")
print(f"  cueSide:         {cue_side.shape}")
print(f"  pictureSequence: {picture_sequence.shape}")
print(f"  --> Auto-detected: {N_SUBJECTS} subjects x {N_BLOCKS} blocks = {N_COLS} columns")
print(f"      Subject IDs: {FIRST_SUB_ID} to {FIRST_SUB_ID + N_SUBJECTS - 1}")

# Load eye-tracking data for trial exclusion
_et_available   = False
_et_pic_idx     = None
_et_fix_idx     = None
_et_x_left      = None
_et_x_right     = None
_et_time_vec_c  = None

if ET_EXCLUSION_ENABLED:
    print("\n[ET] Loading eye-tracking data...")
    try:
        def _load_cell(name):
            d = sio.loadmat(str(DATA_DIR / f"{name}.mat"),
                            squeeze_me=False, struct_as_record=False)
            k = [x for x in d if not x.startswith("_")][0]
            return d[k]

        _et_pic_idx    = _load_cell("pictureTimeIdx")
        _et_fix_idx    = _load_cell("fixationTimeIdx")
        _et_x_left     = _load_cell("xPositionLeftContinuos")
        _et_x_right    = _load_cell("xPositionRightContinuos")
        _et_time_vec_c = _load_cell("timeVectorContinuos")

        _et_available = True
    except Exception as _e:
        print(f"  WARNING: gaze exclusion skipped; eye-tracking data could not be loaded: {_e}")


def get_valence(pic_id):
    if int(pic_id) in POSITIVE_IDS:
        return "positive"
    elif int(pic_id) in NEGATIVE_IDS:
        return "negative"
    return "unknown"


def get_iaps_valence(pic_id):
    """Look up IAPS valence (0-based seq ID). NaN if not found."""
    return IAPS_LOOKUP.get(int(pic_id), np.nan)


def get_block_order(col):
    return block_order_list[col] if col < len(block_order_list) else "?"


# === Build trial table and apply gaze exclusion ===

print(f"\nBuilding trial table ({N_SUBJECTS} subjects x {N_BLOCKS} blocks x {N_TRIALS} trials = {N_COLS} columns)...")

rows = []
n_et_excluded = 0              # trials excluded: no picture dwell (or only early saccade)
early_saccade_by_subj = {}     # per-subject count of trials excluded by 100ms filter
n_premature_excluded = 0       # trials excluded: saccade from center to lateral during fixation cross
premature_by_subj = {}         # per-subject count
n_wrong_dir_excluded = 0       # trials excluded: first dwell on the uncued (wrong) side
wrong_dir_by_subj = {}         # per-subject count

for subj_idx in range(N_SUBJECTS):
    subj_id = FIRST_SUB_ID + subj_idx

    for block_idx in range(N_BLOCKS):
        col   = subj_idx * N_BLOCKS + block_idx      # column index in the 88 x 146 data
        order = get_block_order(col)                  # 'A' or 'B'

        # Preload eye-tracking arrays once per column
        _x_l_col = _x_r_col = _onsets_col = _fix_idx_col = _t_vec_col = None
        if _et_available:
            try:
                # Mask gaze outside the screen
                _xl_raw        = _et_x_left[0, col].flatten().astype(float)
                _xr_raw        = _et_x_right[0, col].flatten().astype(float)
                _x_l_col       = np.where((_xl_raw < 0) | (_xl_raw > 1920), np.nan, _xl_raw)
                _x_r_col       = np.where((_xr_raw < 0) | (_xr_raw > 1920), np.nan, _xr_raw)
                _onsets_col    = _et_pic_idx[0, col].flatten().astype(float)
                if col < _et_fix_idx.shape[1]:
                    _fix_idx_col = _et_fix_idx[0, col].flatten().astype(float)
                else:
                    _fix_idx_col = None
                _t_vec_col     = _et_time_vec_c[0, col].flatten().astype(float)
            except Exception:
                pass

        for trial_idx in range(N_TRIALS):
            rt_s = float(reaction_time[trial_idx, col])

            if np.isnan(rt_s) or rt_s == 0:
                continue

            # A = congruent first; B: incongruent first (44 trials per half)
            
            in_first_half = trial_idx < 44
            if order == "A":
                condition = "congruent" if in_first_half else "incongruent"
            elif order == "B":
                condition = "incongruent" if in_first_half else "congruent"
            else:
                condition = "unknown"

            
            cue    = int(cue_side[trial_idx, col])
            lp     = picture_sequence[trial_idx, 0, col]
            rp     = picture_sequence[trial_idx, 1, col]

            if np.isnan(lp) or np.isnan(rp):
                continue

            left_pic  = int(lp)
            right_pic = int(rp)

            # Target = cued picture; distractor = uncued picture
            if cue == 0:   # cue points left
                target_pic, distractor_pic = left_pic, right_pic
                cue_dir = "left"
            else:           # cue points right
                target_pic, distractor_pic = right_pic, left_pic
                cue_dir = "right"

            target_valence     = get_valence(target_pic)
            distractor_valence = get_valence(distractor_pic)

            # Congruent: positive = pull, negative = push ; incongruent: reversed
            move_perf = reaction_performed[trial_idx, col] if reaction_performed is not None else None
            if target_valence == "positive":
                exp_move = "pull" if condition == "congruent" else "push"
            elif target_valence == "negative":
                exp_move = "push" if condition == "congruent" else "pull"
            else:
                exp_move = None
            is_correct = (float(move_perf == exp_move)
                          if (move_perf is not None and exp_move is not None) else np.nan)

            # Intensity codes by valence category and picture role
            
            target_iaps     = get_iaps_valence(target_pic)
            distractor_iaps = get_iaps_valence(distractor_pic)
            if target_valence == "positive":
                target_intensity_e, distractor_intensity_e = round_code_pos(target_iaps), round_code_neg(distractor_iaps)
                pos_intensity_e, neg_intensity_e = target_intensity_e, distractor_intensity_e
            elif target_valence == "negative":
                target_intensity_e, distractor_intensity_e = round_code_neg(target_iaps), round_code_pos(distractor_iaps)
                pos_intensity_e, neg_intensity_e = distractor_intensity_e, target_intensity_e
            else:
                target_intensity_e = distractor_intensity_e = pos_intensity_e = neg_intensity_e = np.nan

            if (ET_EXCLUSION_ENABLED and _et_available
                    and _x_l_col is not None and _t_vec_col is not None
                    and _onsets_col is not None and _fix_idx_col is not None
                    and trial_idx < len(VALID_ET_IDX)
                    and VALID_ET_IDX[trial_idx] < len(_onsets_col)
                    and VALID_ET_IDX[trial_idx] < len(_fix_idx_col)):
                try:
                    _et_i     = VALID_ET_IDX[trial_idx]
                    t_pic_idx = int(_onsets_col[_et_i])
                    t_fix_idx = int(_fix_idx_col[_et_i])
                    if t_pic_idx < len(_t_vec_col) and t_fix_idx < len(_t_vec_col):
                        picture_onset_ms = _t_vec_col[t_pic_idx]
                        cross_onset_ms   = _t_vec_col[t_fix_idx]
                        x_avg = np.where(
                            ~np.isnan(_x_l_col) & ~np.isnan(_x_r_col),
                            (_x_l_col + _x_r_col) / 2.0,
                            np.where(~np.isnan(_x_l_col), _x_l_col, _x_r_col)
                        )
                        # Estimate gaze center during fixation cross (Koch 2025)
                        cross_mask = ((_t_vec_col >= cross_onset_ms)
                                      & (_t_vec_col < picture_onset_ms))
                        center_x   = find_trial_center(x_avg[cross_mask])
                        # Exclude trials with a deliberate saccade away from center during the fixation cross
                        if had_premature_gaze(x_avg, _t_vec_col, cross_onset_ms, picture_onset_ms, center_x):
                            n_premature_excluded += 1
                            premature_by_subj[subj_id] = premature_by_subj.get(subj_id, 0) + 1
                            continue
                        _dwell = has_picture_dwell(x_avg, _t_vec_col, picture_onset_ms, center_x)
                        if _dwell not in ("left", "right"):
                            n_et_excluded += 1
                            if _dwell == "early":
                                early_saccade_by_subj[subj_id] = early_saccade_by_subj.get(subj_id, 0) + 1
                            continue
                        # Exclude first dwell on the uncued side
                        if EXCLUDE_WRONG_DIRECTION and _dwell != cue_dir:
                            n_et_excluded += 1
                            n_wrong_dir_excluded += 1
                            wrong_dir_by_subj[subj_id] = wrong_dir_by_subj.get(subj_id, 0) + 1
                            continue
                except Exception:
                    pass

            rows.append({
                "subject"                 : subj_id,
                "block"                   : block_idx + 1,
                "trial"                   : trial_idx + 1,
                "block_order"             : order,
                "condition"               : condition,
                "cue_side"                : cue_dir,
                "target_pic"              : target_pic,
                "distractor_pic"          : distractor_pic,
                "target_valence"          : target_valence,
                "target_valence_e"        : -0.5 if target_valence == "positive" else (0.5 if target_valence == "negative" else np.nan),
                "movement_performed"      : move_perf,
                "correct"                 : is_correct,
                "distractor_valence"      : distractor_valence,
                "target_valence_iaps"     : target_iaps,
                "distractor_valence_iaps" : distractor_iaps,
                "pos_intensity_e"         : pos_intensity_e,
                "neg_intensity_e"         : neg_intensity_e,
                "target_intensity_e"      : target_intensity_e,
                "distractor_intensity_e"  : distractor_intensity_e,
                "rt_s"                    : rt_s,
                "rt_ms"                   : rt_s * 1000.0,
            })

df = pd.DataFrame(rows)
n_raw             = N_SUBJECTS * N_BLOCKS * N_TRIALS                      # maximum possible trials
n_after_nan_rp    = len(df) + n_premature_excluded + n_et_excluded        # after NaN/no-response
n_after_premature = len(df) + n_et_excluded                               # after pre-onset exclusion
n_after_et        = len(df)                                                # after picture dwell exclusion
print(f"  Total rows: {len(df)}")
if ET_EXCLUSION_ENABLED and _et_available:
    pct_premature = 100 * n_premature_excluded / max(n_after_nan_rp, 1)
    print(f"  Pre-onset gaze:  {n_premature_excluded} trials ({pct_premature:.1f}%) -- EXCLUDED")
    if n_premature_excluded > 0:
        print("    Per subject:")
        for s in sorted(premature_by_subj):
            print(f"      Subj {s}: {premature_by_subj[s]} trials")
    n_early_total = sum(early_saccade_by_subj.values())
    print(f"  ET exclusions (no dwell + wrong side): {n_et_excluded} trials")
    print(f"    Early first dwell(<=100ms): {n_early_total} trials")
    print(f"    First dwell on uncued side:  {n_wrong_dir_excluded} trials")
    if n_early_total > 0:
        print("    Per subject (100ms filter):")
        for s in sorted(early_saccade_by_subj):
            print(f"      Subj {s}: {early_saccade_by_subj[s]} trials")
print(f"  Subjects:   {sorted(df.subject.unique())}")
print(f"  Conditions: {df.condition.value_counts().to_dict()}")
print(f"  Valence (distractor): {df.distractor_valence.value_counts().to_dict()}")
iaps_nan = df["target_valence_iaps"].isna().sum()
if iaps_nan > 0:
    print(f"  WARNING: {iaps_nan} trials have NaN IAPS valence (check picture IDs)")

print(f"\n  target_intensity_e distribution:     {df['target_intensity_e'].value_counts().sort_index().to_dict()}")
print(f"  distractor_intensity_e distribution: {df['distractor_intensity_e'].value_counts().sort_index().to_dict()}")

# === Apply accuracy exclusion after gaze filtering (Koch 2025) ===
acc_low_subj = []
n_after_acc   = None           
if ACCURACY_FILTER and "correct" in df.columns and df["correct"].notna().any():
    subj_acc = df.groupby("subject")["correct"].mean() * 100
    acc_low_subj = subj_acc[subj_acc < ACCURACY_MIN_PCT].index.tolist()
    overall_acc = df["correct"].mean() * 100
    if overall_acc < 60.0:
        print(f"\n  WARNING: overall accuracy is {overall_acc:.1f}%, close to the 50% chance level.")
        print(f"  Check trial alignment in _decode_reaction_performed() before trusting the accuracy filter.")
    print(f"\nAccuracy filter (Koch 2025, threshold {ACCURACY_MIN_PCT}%):")
    for s in sorted(subj_acc.index):
        flag = " <-- EXCLUDE" if subj_acc[s] < ACCURACY_MIN_PCT else ""
        print(f"    Subj {s}: {subj_acc[s]:.1f}%{flag}")
    n_err = int((df["correct"] == 0.0).sum())
    if DROP_ERROR_TRIALS:
        df = df[df["correct"] != 0.0].copy()
        print(f"  Error trials removed: {n_err}")
    if acc_low_subj:
        df = df[~df["subject"].isin(acc_low_subj)].copy()
        print(f"  Excluded {len(acc_low_subj)} subjects < {ACCURACY_MIN_PCT}% accuracy: {acc_low_subj}")
    n_after_acc = len(df)
    print(f"  After accuracy filter: {len(df)} trials, {df['subject'].nunique()} subjects")

# === Remove RT < 150ms and within-subject +/-2.5 SD outliers ===
before = len(df)
df = df[df["rt_ms"] >= 150].copy()
n_after_cutoff = len(df)
print(f"  Removed: {before - len(df)} trials | Remaining: {len(df)}")

subj_stats = (
    df.groupby("subject")["rt_ms"]
    .agg(subj_mean="mean", subj_sd="std")
    .reset_index()
)
df = df.merge(subj_stats, on="subject")

# flag outliers
df["outlier"] = (
    (df["rt_ms"] < df["subj_mean"] - N_SD * df["subj_sd"]) |
    (df["rt_ms"] > df["subj_mean"] + N_SD * df["subj_sd"])
)

# how many flagged
n_out = df["outlier"].sum()
n_after_sd = len(df) - n_out
print(f"  Flagged: {n_out} trials ({100 * n_out / len(df):.1f}%)")

# === Exclude subjects with insufficient valid trials (<70% after all exclusions) ===

# Use expected trial counts to account for missing sessions
EXPECTED_TRIALS_PER_SUBJ = N_BLOCKS * N_TRIALS

kept_per_subj = df[~df["outlier"]].groupby("subject").size()

all_subjects   = df["subject"].unique()
kept_full      = kept_per_subj.reindex(all_subjects, fill_value=0)
pct_valid      = (kept_full / EXPECTED_TRIALS_PER_SUBJ * 100).round(1)

print(f"  Expected trials per subject: {EXPECTED_TRIALS_PER_SUBJ} ({N_BLOCKS} blocks x {N_TRIALS} trials)")
print("\n  Subject | Valid %")
for subj, pct in sorted(pct_valid.items()):
    flag = " <-- EXCLUDE" if pct < MIN_VALID_PCT else ""
    print(f"  {subj:6d}  | {pct:6.1f}%{flag}")

low_subj = pct_valid[pct_valid < MIN_VALID_PCT].index.tolist()
if low_subj:
    print(f"\n  Excluded {len(low_subj)} subjects below {MIN_VALID_PCT}% valid trials: {low_subj}")
else:
    print(f"\n  All subjects >= {MIN_VALID_PCT}% valid trials.")

# === Plot RT outliers (capped y-axis) ===
kept_rt = df.loc[~df["outlier"], "rt_ms"]
y_cap   = np.percentile(kept_rt, 99.5) * 1.15
n_above = (df["rt_ms"] > y_cap).sum()

# plot: per-subject rt scatter with outliers flagged
fig, ax = plt.subplots(figsize=(16, 5))
jitter = np.random.uniform(-0.3, 0.3, len(df))
colors = df["outlier"].map({False: "#4285F4", True: "#E8401C"})
ax.scatter(df["subject"] + jitter, df["rt_ms"].clip(upper=y_cap),
           c=colors, alpha=0.2, s=5, linewidths=0)
ax.set_ylim(0, y_cap)
ax.set_xlabel("Subject ID", fontsize=12)
ax.set_ylabel("RT (ms)", fontsize=12)
ax.set_title(
    f"Cued AAT -- Per-subject outlier removal (mean +/- {N_SD} SD)\n"
    f"y-axis capped at {y_cap:.0f} ms  |  {n_above} extreme values clipped (all flagged as outliers)",
    fontsize=11
)
ax.legend(handles=[mpatches.Patch(color="#4285F4", label="kept"),
                   mpatches.Patch(color="#E8401C", label="outlier")],
          loc="upper right")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "cuedtask_01_outlier_removal.png", dpi=150)
plt.close()

# Keep only valid trials from subjects with sufficient data
df_clean = df[~df["outlier"] & ~df["subject"].isin(low_subj)].copy()
n_final   = len(df_clean)
# === Report final counts and exclusions ===
if low_subj:
    print(f"\n  Excluded subjects: {low_subj}")
print(f"\n  Remaining after outlier removal + subject exclusion: {n_final} trials"
      f" ({df_clean['subject'].nunique()} subjects)")

# Preprocessing summary table
print("  Preprocessing step                      |     N ")
print(f"  Raw (subjects x blocks x trials)        | {n_raw:>5} ")
print(f"  After NaN / no-response removal         | {n_after_nan_rp:>5} ")
if ET_EXCLUSION_ENABLED and _et_available:
    print(f"  After pre-onset gaze exclusion          | {n_after_premature:>5} ")
    print(f"  After no-picture-dwell exclusion (ET)   | {n_after_et:>5} ")
if n_after_acc is not None:
    print(f"  After accuracy filter (errors + <90%)   | {n_after_acc:>5} ")
print(f"  After RT<150ms removal                  | {n_after_cutoff:>5} ")
print(f"  After mean+/-2.5SD outlier removal      | {n_after_sd:>5} ")
if low_subj:
    print(f"  After excluding {len(low_subj)} subject(s) <70% valid | {n_final:>5} ")
print(f"  FINAL (for LMER)                        | {n_final:>5} ")

# === Plot trial exclusion funnel ===

funnel_labels = ["Raw", "NaN / no response"]
funnel_vals   = [n_raw, n_after_nan_rp]
if ET_EXCLUSION_ENABLED and _et_available and n_premature_excluded > 0:
    funnel_labels.append("Pre-onset gaze")
    funnel_vals.append(n_after_premature)
if ET_EXCLUSION_ENABLED and _et_available and n_et_excluded > 0:
    funnel_labels.append("No picture dwell")
    funnel_vals.append(n_after_et)


if n_after_acc is not None and n_after_acc < n_after_et:
    funnel_labels.append("Accuracy\n(errors + subj < 90%)")
    funnel_vals.append(n_after_acc)
funnel_labels += ["RT < 150 ms", "±2.5 SD outliers"]
funnel_vals   += [n_after_cutoff, n_after_sd]
if low_subj:
    funnel_labels.append("Subjects < 70% valid")
    funnel_vals.append(n_final)

fig, ax = plt.subplots(figsize=(ps.W, 4.6))
bars = ax.bar(range(len(funnel_vals)), funnel_vals,
              color=ps.C_BAR, edgecolor="white", width=0.6)
for bar, val in zip(bars, funnel_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(funnel_labels)))
ax.set_xticklabels(funnel_labels, fontsize=9, rotation=30, ha="right")
ax.set_ylabel("Number of trials")

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
ps.save(fig, RESULTS_DIR / "cuedtask_02_exclusion_funnel.png", verbose=False)

# === Prepare LMER variables ===
print("\nLog-transform RT + add LMER columns...")
df_clean = df_clean.copy()
df_clean["log_rt"] = np.log(df_clean["rt_ms"])

# Target side: left = -0.5, right = 0.5
df_clean["target_side_e"] = df_clean["cue_side"].map({"left": -0.5, "right": 0.5})

# Standardize trial number within each subject for LMER
df_clean = df_clean.sort_values(["subject", "block", "trial"]).copy()
df_clean["trial_num"] = df_clean.groupby("subject").cumcount() + 1
df_clean["z_trial"] = df_clean.groupby("subject")["trial_num"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(f"  z_trial range: [{df_clean['z_trial'].min():.2f}, {df_clean['z_trial'].max():.2f}]")
print(f"  Mean RT (ms):  {df_clean['rt_ms'].mean():.1f}")
print(f"  Mean log(RT):  {df_clean['log_rt'].mean():.3f}")
print(f"  SD   log(RT):  {df_clean['log_rt'].std():.3f}")

# === Plot RT distributions before/after log-transform ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df_clean["rt_ms"],  bins=60, color="#4285F4", edgecolor="white", alpha=0.85)
axes[0].set_title("RT (ms) -- before transform")
axes[0].set_xlabel("RT (ms)")
axes[1].hist(df_clean["log_rt"], bins=60, color="#E8401C", edgecolor="white", alpha=0.85)
axes[1].set_title("log(RT) -- after transform")
axes[1].set_xlabel("log(RT)")
for ax in axes:
    ax.set_ylabel("Count")
plt.suptitle("Cued AAT -- RT distributions", fontsize=13)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "cuedtask_03_logrt_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# === Plot per-subject mean RT (sorted), to spot unusually slow/fast subjects ===
subj_means = df_clean.groupby("subject")["rt_ms"].mean().sort_values()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(subj_means)), subj_means.values, color="steelblue", alpha=0.8, edgecolor="white")
ax.axhline(subj_means.mean(), color="tomato", linestyle="--", linewidth=2,
           label=f"Grand mean = {subj_means.mean():.0f} ms")
ax.set_xlabel("Subject (sorted by mean RT)", fontsize=12)
ax.set_ylabel("Mean RT (ms)", fontsize=12)
ax.set_title("Cued AAT: Per-subject Mean RT", fontsize=13)
ax.set_xticks(range(len(subj_means)))
ax.set_xticklabels([str(s) for s in subj_means.index], rotation=45, fontsize=8)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "cuedtask_04_per_subject_rt.png", dpi=150, bbox_inches="tight")
plt.close()

# === Save preprocessed trial table ===
out_csv = RESULTS_DIR / "cuedtask_trial_table_preprocessed.csv"
df_clean.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv.name} ({len(df_clean)} trials, {df_clean['subject'].nunique()} subjects)")

print("\nDone. Next: run lmer_cuedtask.py")
