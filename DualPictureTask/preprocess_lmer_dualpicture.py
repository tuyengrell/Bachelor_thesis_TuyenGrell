"""
Dual Picture Task - Preprocessing

Steps:
1) Load behavioral and gaze data; identify target and distractor
2) Code instructed congruency, response accuracy, and valence intensity
3) Apply gaze checks and remove instruction errors
4) Remove RT < 150ms and +/-2.5 SD outliers per participant
5) Exclude participants with <70% valid trials
6) Log-transform RT, code target side, and standardize trial number
7) Report descriptive statistics, make plots and save the trial table

The 90% accuracy threshold is reported for sensitivity checks only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
from pathlib import Path

# === Settings ===
DATA_DIR    = Path(__file__).parent          # DualPictureTask/ folder
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Shared participant-level descriptive helpers
import sys
sys.path.insert(0, str(DATA_DIR.parent))
from analysis_common import subject_desc, subject_means

# Within-participant RT cutoff (SD)
N_SD = 2.5

# Exclude participants with <70% valid trials
MIN_VALID_PCT = 70.0

# Assign congruency by block half
# Disable both flags to use executed congruency without removing instruction errors
CONGRUENCY_FROM_INSTRUCTION = True   # condition from the block half, not the movement
DROP_INSTRUCTION_ERRORS     = True   # remove trials that violate the instructed mapping

# A/C: congruent first, B/D: incongruent first
CONGRUENT_FIRST_ORDERS = ("A", "C")
HALF_SWITCH_TRIAL      = 48          # trials 1-48 = first half, 49-96 = second half

# Accuracy threshold for sensitivity reporting only; not a main-analysis exclusion
SENSITIVITY_ACC_PCT = 90.0


def instructed_condition(block_order_letter, trial_index_0based):
    """Return instructed congruency from block order and trial half."""
    first_half = trial_index_0based < HALF_SWITCH_TRIAL
    congruent_first = block_order_letter in CONGRUENT_FIRST_ORDERS
    if congruent_first:
        return "congruent" if first_half else "incongruent"
    return "incongruent" if first_half else "congruent"

# Shared stimulus mapping and plot style
from stimulus_map import POSITIVE_IDS, NEGATIVE_IDS, IAPS_LOOKUP

import plot_style as ps

# === Intensity coding from rounded IAPS valence ===
def round_code_pos(iaps_val):
    """Map rounded positive valence: -1=least positive (rounded 7 or below), 0=medium (rounded 8), +1=most positive (rounded 9)."""
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
    """Map rounded negative valence: +1=most negative (rounded 1), 0=medium (rounded 2), -1=least negative (rounded 3 or above)."""
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
ET_EXCLUSION_ENABLED = True
SCREEN_CENTER_PX     = 960    # horizontal screen center in pixels
CROSS_ROI_HALF       = 100    # fixation cross region: center +/- 100px
LEAVE_SAMPLES        = 50     # End dwell after >50 samples outside the center band
VEL_THRESHOLD        = 5      # saccade velocity (px/sample)
MIN_PIC_STABLE       = 20     # Stability window; at least 80% valid
MAX_VAR_PX           = 50     # Max gaze range in window (px)
POST_ONSET_MS        = 3000   # how far after picture onset to search for dwell
EARLY_SACCADE_MS     = 100    # Early dwell threshold (ms)

# Pre-onset exclusion disabled: without a cue, an early lateral saccade is the
# behaviour of interest, not a rule violation (own analysis decision)
PRE_ONSET_EXCLUSION  = False

# Early dwells (<=100 ms) are skipped rather than excluded, as their count looks
# inflated by the onset timing of this task (own analysis decision)
STRICT_100MS         = False


def find_trial_center(x_win):
    """Estimate gaze center from the longest center-band dwell; use a median fallback"""
    n = len(x_win)
    if n < 5:
        return float(SCREEN_CENTER_PX)
    in_roi = (~np.isnan(x_win)) & (np.abs(x_win - SCREEN_CENTER_PX) <= CROSS_ROI_HALF)
    if not in_roi.any():
        valid_x = x_win[~np.isnan(x_win)]
        if len(valid_x) == 0:
            return float(SCREEN_CENTER_PX)
        fb_center = float(np.median(valid_x))
        in_roi2 = (~np.isnan(x_win)) & (np.abs(x_win - fb_center) <= CROSS_ROI_HALF)
        if in_roi2.any():
            return float(np.nanmedian(x_win[in_roi2]))
        return fb_center
    best_start = best_end = 0
    best_len = 0
    cur_start = None
    out_count = 0
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
                    length = end - cur_start
                    if length > best_len:
                        best_len = length
                        best_start = cur_start
                        best_end = end
                    cur_start = None
                    out_count = 0
    if cur_start is not None:
        end = n - 1
        length = end - cur_start + 1
        if length > best_len:
            best_start = cur_start
            best_end = end
    if best_len == 0:
        return float(np.nanmedian(x_win[in_roi]))
    return float(np.nanmedian(x_win[best_start:best_end+1]))


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


def has_picture_dwell(x_arr, t_arr, pic_onset_ms, center_x):
    """Return left/right for an accepted lateral dwell, or early/none. 
    With STRICT_100MS=FALSE, skip early dwells and search for later dwell."""
    mask = (t_arr >= pic_onset_ms) & (t_arr < pic_onset_ms + POST_ONSET_MS)
    if mask.sum() < MIN_PIC_STABLE + 2:
        return "none"
    x_win = x_arr[mask]
    t_win = t_arr[mask]
    n = len(x_win)
    found_early = False
    for i in range(n - MIN_PIC_STABLE - 1):
        if np.isnan(x_win[i]) or np.isnan(x_win[i + 1]):
            continue
        is_early = t_win[i] - pic_onset_ms <= EARLY_SACCADE_MS
        if abs(x_win[i + 1] - x_win[i]) > VEL_THRESHOLD:
            pos_x = x_win[i + 1]
            if pos_x < center_x - CROSS_ROI_HALF:
                side = "left"
            elif pos_x > center_x + CROSS_ROI_HALF:
                side = "right"
            else:
                continue  # velocity spike but still inside center band
            seg = x_win[i + 1 : i + 1 + MIN_PIC_STABLE]
            valid = seg[~np.isnan(seg)]
            if (len(valid) >= int(MIN_PIC_STABLE * 0.8)
                    and np.max(valid) - np.min(valid) <= MAX_VAR_PX):
                if is_early:
                    if STRICT_100MS:
                        return "early"   # first stable saccade is anticipatory -> exclude
                    found_early = True
                    continue  # Skip early dwell; search for later one
                return side   # "left" or "right"
    return "early" if found_early else "none"

# movement codes: push = avoidance; pull = approach
PUSH_CODE = 1
PULL_CODE = 2

# === Load behavioral data ===
print("Dual Picture Task -- Preprocessing")
print(f"\nData directory: {DATA_DIR.resolve()}")
print(f"Results:        {RESULTS_DIR.resolve()}")
print(f"\nLoading .mat files...")


def load_mat(name):
    """Load the first non-metadata variable from a MATLAB file"""
    path = DATA_DIR / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(
            f"  ERROR: {path} not found.\n"
            f"  Expected files: reactionTime, side, movementType,\n"
            f"                  leftPictureID, rightPictureID, blockOrder"
        )
    data = sio.loadmat(str(path))
    keys = [k for k in data if not k.startswith("_")]
    return data[keys[0]]

def load_block_order():
    """Load blockOrder from CSV using one based column indices."""
    path = RESULTS_DIR / "block_order.csv" 
    if not path.exists():
        raise SystemExit(f"block_order.csv not found: {path}")

    df_bo = pd.read_csv(path)
    max_idx = int(df_bo["col_idx"].max())
    order_list = ["unknown"] * max_idx
    for _, row in df_bo.iterrows():
        order_list[int(row["col_idx"]) - 1] = row["order"]
    print(f"  blockOrder: {len(df_bo)} values loaded from block_order.csv")
    return order_list


def load_movement_type_mcos(mat_path, expected_shape):
    """Decode MCOS movement strings: 1=push, 2=pull; return None on failure."""
    import zlib as _zlib
    import struct as _struct

    # Decode UTF-16-LE push/pull strings from compressed MATLAB data
    try:
        with open(str(mat_path), "rb") as _f:
            raw = _f.read()

        push_u16 = "push".encode("utf-16-le")   # 8 bytes
        pull_u16 = "pull".encode("utf-16-le")   # 8 bytes

        
        streams = [i for i in range(len(raw) - 1)
                   if raw[i] == 0x78 and raw[i + 1] in (0x9C, 0xDA, 0x01, 0x5E)]

        events = []
        for pos in streams:
            try:
                dec = _zlib.decompress(raw[pos:])
            except Exception:
                continue
            for i in range(len(dec) - 8 + 1):
                chunk = dec[i:i + 8]
                if chunk == push_u16:
                    events.append((i, 1))
                elif chunk == pull_u16:
                    events.append((i, 2))

        if not events:
            return None

        events.sort(key=lambda x: x[0])
        # Remove duplicates from overlapping stream scans
        seen = set()
        unique = []
        for pos, val in events:
            if pos not in seen:
                seen.add(pos)
                unique.append(val)
        events = unique

        n_rows, n_cols = expected_shape
        if len(events) != n_rows * n_cols:
            print(f"  WARNING: movementType decoded {len(events)} values, "
                  f"expected {n_rows * n_cols}. Using what we have.")

        # Restore MATLAB column-major trial order
        arr = np.array(events[:n_rows * n_cols], dtype=np.int32)
        return arr.reshape(n_cols, n_rows).T   # shape (n_rows, n_cols)

    except Exception as e:
        print(f"  movementType MCOS decode failed: {e}")
        return None


reaction_time    = load_mat("reactionTime")     # expected: (n_trials, n_cols)
side             = load_mat("side")             # (n_trials, n_cols)  0=left, 1=right
left_pic_id      = load_mat("leftPictureID")    # (n_trials, n_cols)
right_pic_id     = load_mat("rightPictureID")   # (n_trials, n_cols)
block_order_raw  = load_block_order()           # list of strings or None

# Try numeric movement code, then the MCOS decoder
movement_type    = None
MOVEMENT_AVAILABLE = False
try:
    _raw = load_mat("movementType")
    _test = _raw.astype(float)
    if _raw.shape == reaction_time.shape:
        movement_type    = _raw
        MOVEMENT_AVAILABLE = True
        print(f"  movementType:   shape={_raw.shape}  (numeric, loaded via scipy)")
    else:
        print(f"  movementType:   shape={_raw.shape}  (shape mismatch -- trying MCOS decoder)")
except (TypeError, ValueError):
    pass  # expected for MCOS format

if not MOVEMENT_AVAILABLE:
    _mt = load_movement_type_mcos(
        DATA_DIR / "movementType.mat", reaction_time.shape)
    if _mt is not None:
        movement_type    = _mt
        MOVEMENT_AVAILABLE = True
        print(f"  movementType:   shape={_mt.shape}  "
              f"(decoded from MCOS UTF-16: "
              f"{(_mt==1).sum()} push, {(_mt==2).sum()} pull)")
    else:
        print(f"  movementType:   MCOS decode failed -- congruency set to 'unknown'.")

# auto-detect dimensions
N_TRIALS_RAW, N_COLS = reaction_time.shape
print(f"\n  reactionTime:   shape={reaction_time.shape}")
print(f"  side:           shape={side.shape}")
print(f"  leftPictureID:  shape={left_pic_id.shape}")
print(f"  rightPictureID: shape={right_pic_id.shape}")

# Consecutive column pairs represent two blocks per participant
N_BLOCKS_ASSUMED = 2
N_SUBJECTS_INFERRED = N_COLS // N_BLOCKS_ASSUMED
print(f"\n  Inferred: {N_SUBJECTS_INFERRED} subjects x {N_BLOCKS_ASSUMED} blocks = {N_COLS} columns")

# === Load eye-tracking data ===
_et_available  = False
_et_eye_data   = None   # cell 1×N_COLS: N_samples×6 per session
_et_time_vec   = None   # cell 1×N_COLS: N_samples time array
_et_onsets     = None   # cell 1×N_COLS: N_trials onset times (relative to session start)
_et_fix_times  = None   # cell 1×N_COLS: N_trials fixation cross onset (relative to session start)

if ET_EXCLUSION_ENABLED:
    print("\n[ET] Loading eye-tracking data for picture-dwell exclusion (Koch 2025)...")
    try:
        def _load_cell_et(name):
            d = sio.loadmat(str(DATA_DIR / f"{name}.mat"),
                            squeeze_me=False, struct_as_record=False)
            k = [x for x in d if not x.startswith("_")][0]
            return d[k]

        _et_eye_data  = _load_cell_et("eyeData")           # 1×41 cell, each N×6
        _et_time_vec  = _load_cell_et("timeVector")         # 1×41 cell, each N
        _et_onsets    = _load_cell_et("pictureOnsetTime")   # 1×41 cell, each M (relative to session start)
        _et_fix_times = _load_cell_et("fixationCrossTime")  # 1×41 cell, each M (relative to session start)

        _et_available = True
        print(f"  ET loaded OK. eyeData shape: {_et_eye_data.shape}")
        print(f"  Picture-dwell exclusion: vel threshold={VEL_THRESHOLD} px/sample, "
              f"min stable={MIN_PIC_STABLE} samples, max var={MAX_VAR_PX} px, "
              f"search window={POST_ONSET_MS} ms post onset")
    except Exception as _e:
        print(f"  WARNING: eye-tracking not loaded: {_e}")
        print(f"  Saccade exclusion skipped.")


def get_valence(pic_id):
    if int(pic_id) in POSITIVE_IDS:
        return "positive"
    elif int(pic_id) in NEGATIVE_IDS:
        return "negative"
    return "unknown"


def get_iaps_valence(pic_id):
    """Return continuous IAPS valence (0-based seq ID). NaN if not found."""
    return IAPS_LOOKUP.get(int(pic_id), np.nan)


def get_block_order(col):
    if block_order_raw and col < len(block_order_raw):
        return block_order_raw[col]
    return "unknown"


def get_movement(move_val):
    """Map numeric movement codes to push/pull, or unknown."""
    v = int(move_val) if not np.isnan(float(move_val)) else -1
    if v == PUSH_CODE:
        return "push"
    elif v == PULL_CODE:
        return "pull"
    return "unknown"


# === Build trial table and apply gaze exclusions ===
print(f"\nBuilding trial table...")

rows = []
n_et_excluded = 0              # trials excluded: no picture dwell (or only early saccade)
early_saccade_by_subj = {}     # per-subject count of trials excluded by 100ms filter
n_premature_excluded = 0       # trials excluded: gaze already lateral before picture onset
premature_by_subj = {}         # per-subject count

for col in range(N_COLS):
    subj_id  = (col // N_BLOCKS_ASSUMED) + 1    # 1-indexed
    block    = (col % N_BLOCKS_ASSUMED) + 1      # 1 or 2
    order    = get_block_order(col)

    # Load gaze arrays once per block
    _eye_mat_col   = None
    _time_vec_col  = None
    _onsets_col    = None
    _fix_times_col = None
    if _et_available:
        try:
            _eye_mat_col   = _et_eye_data[0, col]                    # N_samples × 6
            _time_vec_col  = _et_time_vec[0, col].flatten().astype(float)
            _onsets_col    = _et_onsets[0, col].flatten().astype(float)
            _fix_times_col = _et_fix_times[0, col].flatten().astype(float)
        except Exception:
            pass

    for trial_idx in range(N_TRIALS_RAW):
        rt_s = float(reaction_time[trial_idx, col])

        if np.isnan(rt_s) or rt_s <= 0:
            continue

        lp = left_pic_id[trial_idx, col]
        rp = right_pic_id[trial_idx, col]
        sd = side[trial_idx, col]
        
        if any(np.isnan(float(x)) for x in [lp, rp, sd]):
            continue

        left_pic  = int(lp)
        right_pic = int(rp)
        side_val  = int(sd)    # 0=left, 1=right (side of first fixated picture)

        if MOVEMENT_AVAILABLE:
            mv = movement_type[trial_idx, col]
            if np.isnan(float(mv)):
                continue
            move_str = get_movement(mv)
        else:
            move_str = "unknown"

        # Target = picture selected by recorded side; distractor = other picture
        if side_val == 0:
            target_pic, distractor_pic = left_pic, right_pic
            target_side = "left"
        else:
            target_pic, distractor_pic = right_pic, left_pic
            target_side = "right"

        target_valence     = get_valence(target_pic)
        distractor_valence = get_valence(distractor_pic)

        # Intensity codes valence category and picture role
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

        # Executed congruency: positive + pull or negative + push
        if move_str == "pull" and target_valence == "positive":
            executed_congruency = "congruent"
        elif move_str == "push" and target_valence == "negative":
            executed_congruency = "congruent"
        elif move_str in ("push", "pull") and target_valence in ("positive", "negative"):
            executed_congruency = "incongruent"
        else:
            executed_congruency = "unknown"

        # Compare executed congruency with block instructions
        instructed = instructed_condition(order, trial_idx)
        correct = (float(executed_congruency == instructed)
                   if executed_congruency != "unknown" else np.nan)

        congruency = instructed if CONGRUENCY_FROM_INSTRUCTION else executed_congruency


        # fixationCrossTime[N-1] precedes trial N; trial 0 uses screen center
        if (ET_EXCLUSION_ENABLED and _et_available
                and _eye_mat_col is not None and _onsets_col is not None
                and _fix_times_col is not None
                and trial_idx < len(_onsets_col)):
            try:
                onset_t = float(_onsets_col[trial_idx])
                if not np.isnan(onset_t):
                    picture_onset_ms = onset_t + _time_vec_col[0]
                    x_l = _eye_mat_col[:, 0].astype(float)
                    x_r = _eye_mat_col[:, 3].astype(float)
                    x_l = np.where((x_l < 0) | (x_l > 1920), np.nan, x_l)
                    x_r = np.where((x_r < 0) | (x_r > 1920), np.nan, x_r)
                    x_avg = np.where(
                        ~np.isnan(x_l) & ~np.isnan(x_r),
                        (x_l + x_r) / 2.0,
                        np.where(~np.isnan(x_l), x_l, x_r)
                    )
                    if trial_idx == 0 or np.isnan(_fix_times_col[trial_idx - 1]):
                        center_x = float(SCREEN_CENTER_PX)
                    else:
                        cross_ms   = float(_fix_times_col[trial_idx - 1]) + _time_vec_col[0]
                        cross_mask = (_time_vec_col >= cross_ms) & (_time_vec_col < picture_onset_ms)
                        center_x   = find_trial_center(x_avg[cross_mask])
                        # Optional pre onset exclusion (disabled by default) 
                        if PRE_ONSET_EXCLUSION and had_premature_gaze(x_avg, _time_vec_col, cross_ms, picture_onset_ms, center_x):
                            n_premature_excluded += 1
                            premature_by_subj[subj_id] = premature_by_subj.get(subj_id, 0) + 1
                            continue   # EXCLUDE
                    _dwell = has_picture_dwell(x_avg, _time_vec_col, picture_onset_ms, center_x)
                    if _dwell not in ("left", "right"):
                        n_et_excluded += 1
                        if _dwell == "early":
                            early_saccade_by_subj[subj_id] = early_saccade_by_subj.get(subj_id, 0) + 1
                        continue
            except Exception:
                pass

        rows.append({
            "subject"                 : subj_id,
            "block"                   : block,
            "trial"                   : trial_idx + 1,
            "block_order"             : order,
            "target_side"             : target_side,
            "target_pic"              : target_pic,
            "distractor_pic"          : distractor_pic,
            "target_valence"          : target_valence,
            "target_valence_e"        : -0.5 if target_valence == "positive" else (0.5 if target_valence == "negative" else np.nan),
            "distractor_valence"      : distractor_valence,
            "target_valence_iaps"     : target_iaps,
            "distractor_valence_iaps" : distractor_iaps,
            "pos_intensity_e"         : pos_intensity_e,
            "neg_intensity_e"         : neg_intensity_e,
            "target_intensity_e"      : target_intensity_e,
            "distractor_intensity_e"  : distractor_intensity_e,
            "movement"                : move_str,
            "congruency"              : congruency,            # the one the model uses
            "instructed_condition"    : instructed,            # from the block half
            "executed_congruency"     : executed_congruency,   # from the movement made
            "correct"                 : correct,               # response followed instruction
            "rt_s"                    : rt_s,
            "rt_ms"                   : rt_s * 1000.0,
        })

df = pd.DataFrame(rows)
n_raw             = N_COLS * N_TRIALS_RAW
n_after_nan_rp    = len(df) + n_premature_excluded + n_et_excluded
n_after_premature = len(df) + n_et_excluded
n_after_et        = len(df)
print(f"  Total rows:    {len(df)}")
print(f"  (NaN/zero RT skipped: {n_raw - n_after_nan_rp})")
if _et_available:
    pct_premature = 100 * n_premature_excluded / max(n_after_nan_rp, 1)
    print(f"  Pre-onset lateral gaze:  {n_premature_excluded} trials ({pct_premature:.1f}%) -- EXCLUDED")
    if n_premature_excluded > 0:
        print("    Per subject:")
        for s in sorted(premature_by_subj):
            print(f"      Subj {s}: {premature_by_subj[s]} trials")
    n_early_total = sum(early_saccade_by_subj.values())
    print(f"  No picture dwell excl.:  {n_et_excluded} trials ({100*n_et_excluded/(n_after_nan_rp or 1):.1f}%)")
    print(f"    of which anticipatory saccade (<=100ms): {n_early_total} trials")
    if n_early_total > 0:
        print("    Per subject (100ms filter):")
        for s in sorted(early_saccade_by_subj):
            print(f"      Subj {s}: {early_saccade_by_subj[s]} trials")
print(f"  Subjects:      {sorted(df.subject.unique())}")
print(f"  Movements:     {df.movement.value_counts().to_dict()}")
print(f"  Target valence:{df.target_valence.value_counts().to_dict()}")

# === Check instruction compliance and remove errors ===
n_before_err = len(df)
acc_by_subj  = df.groupby("subject")["correct"].mean() * 100
n_err        = int((df["correct"] == 0.0).sum())
overall_acc  = df["correct"].mean() * 100
if overall_acc < 60.0:
    print(f"\n  WARNING: overall instruction compliance is {overall_acc:.1f}%, close to the 50% chance level.")
    print(f"  Check trial alignment in load_movement_type_mcos()")
print(f"\n  Instruction compliance (condition from block half, "
      f"{'USED as congruency' if CONGRUENCY_FROM_INSTRUCTION else 'reported only'}):")
print(f"    trials violating the instructed mapping: {n_err} "
      f"({100*n_err/max(n_before_err,1):.1f}%)")
print("    accuracy per subject: " +
      "  ".join(f"{s}:{a:.0f}%" for s, a in acc_by_subj.round(0).items()))
_low = sorted(acc_by_subj[acc_by_subj < SENSITIVITY_ACC_PCT].index.tolist())
print(f"    below {SENSITIVITY_ACC_PCT:.0f}% (Cued AAT criterion): "
      f"{len(_low)} of {len(acc_by_subj)} subjects {_low}")
print(f"    -> reported as a sensitivity check only, NOT applied here; see the note "
      f"at the top of this file")
# Preserve accuracy before error removal for downstream sensitivity checks
df["subject_accuracy"] = df["subject"].map(acc_by_subj)
if DROP_INSTRUCTION_ERRORS:
    df = df[df["correct"] != 0.0].copy()
    print(f"    removed {n_before_err - len(df)} error trials | remaining: {len(df)}")
n_after_err = len(df)

print(f"  Congruency:    {df.congruency.value_counts().to_dict()}")
n_unknown_valence = (df["target_valence"] == "unknown").sum()
if n_unknown_valence > 0:
    print(f"  NOTE: {n_unknown_valence} trials have unknown target valence (pic ID outside 1-87)")

print(f"\n  target_intensity_e distribution:     {df['target_intensity_e'].value_counts().sort_index().to_dict()}")
print(f"  distractor_intensity_e distribution: {df['distractor_intensity_e'].value_counts().sort_index().to_dict()}")

# === Remove RT <150 ms and within-subject outliers ===
print(f"\nRemove trials with RT < 150 ms (anticipatory responses)...")
before = len(df)
df = df[df["rt_ms"] >= 150].copy()
n_after_cutoff = len(df)
print(f"  Removed: {before - len(df)} trials | Remaining: {len(df)}")

print(f"\nPer-subject outlier removal (mean +/- {N_SD} SD)...")

subj_stats = (
    df.groupby("subject")["rt_ms"]
    .agg(subj_mean="mean", subj_sd="std")
    .reset_index()
)
df = df.merge(subj_stats, on="subject")
df["outlier"] = (
    (df["rt_ms"] < df["subj_mean"] - N_SD * df["subj_sd"]) |
    (df["rt_ms"] > df["subj_mean"] + N_SD * df["subj_sd"])
)

n_out    = df["outlier"].sum()
n_after_sd = len(df) - n_out
print(f"  Flagged: {n_out} trials ({100 * n_out / len(df):.1f}%)")

# === Exclude participants with insufficient valid trials ===
print(f"\nValid trial % per subject (flag if < {MIN_VALID_PCT}%)...")

# Use planned trial counts to account for missing blocks
EXPECTED_TRIALS_PER_SUBJ = N_BLOCKS_ASSUMED * N_TRIALS_RAW

kept_per_subj = df[~df["outlier"]].groupby("subject").size()
all_subjects  = df["subject"].unique()
kept_full     = kept_per_subj.reindex(all_subjects, fill_value=0)
pct_valid     = (kept_full / EXPECTED_TRIALS_PER_SUBJ * 100).round(1)

print(f"  Expected trials per subject: {EXPECTED_TRIALS_PER_SUBJ} ({N_BLOCKS_ASSUMED} blocks x {N_TRIALS_RAW} trials)")
print("\n  Subject | Valid %")
for subj, pct in sorted(pct_valid.items()):
    flag = " <-- EXCLUDE" if pct < MIN_VALID_PCT else ""
    print(f"  {subj:6d}  | {pct:6.1f}%{flag}")


low_subj = pct_valid[pct_valid < MIN_VALID_PCT].index.tolist()
if low_subj:
    print(f"\n  EXCLUDING {len(low_subj)} subject(s) below {MIN_VALID_PCT}%: {low_subj}")
    print(f"  Reason: too few valid trials (incl. missing sessions).")
else:
    print(f"\n  All subjects >= {MIN_VALID_PCT}% valid trials.")

# === Plot RT outliers (capped y-axis) ===
kept_rt   = df.loc[~df["outlier"], "rt_ms"]
y_cap     = np.percentile(kept_rt, 99.5) * 1.15   # a bit above the 99.5th pct
n_above   = (df["rt_ms"] > y_cap).sum()


fig, ax = plt.subplots(figsize=(14, 5))
jitter = np.random.uniform(-0.3, 0.3, len(df))
colors = df["outlier"].map({False: "#4285F4", True: "#E8401C"})
ax.scatter(df["subject"] + jitter, df["rt_ms"].clip(upper=y_cap),
           c=colors, alpha=0.3, s=8, linewidths=0)
ax.set_ylim(0, y_cap)
ax.set_xlabel("Subject ID", fontsize=12)
ax.set_ylabel("RT (ms)", fontsize=12)
ax.set_title(
    f"Dual Picture Task -- Per-subject outlier removal (mean +/- {N_SD} SD)\n"
    f"y-axis capped at {y_cap:.0f} ms  |  {n_above} extreme values clipped (all flagged as outliers)",
    fontsize=11
)
ax.legend(handles=[mpatches.Patch(color="#4285F4", label="kept"),
                   mpatches.Patch(color="#E8401C", label="outlier")],
          loc="upper right")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "dualpicture_01_outlier_removal.png", dpi=150)
plt.close()
print("  Saved: dualpicture_01_outlier_removal.png")

# Keep non outlier trials from retained participants
df_clean = df[~df["outlier"] & ~df["subject"].isin(low_subj)].copy()
n_final  = len(df_clean)
if low_subj:
    print(f"\n  Excluded subjects: {low_subj}")
print(f"\n  Remaining after outlier removal + subject exclusion: {n_final} trials"
      f" ({df_clean['subject'].nunique()} subjects)")

# === Report remaining trials ===
print("   Preprocessing step                      |     N ")
print(f"   Raw (cols x trials)                     | {n_raw:>5} ")
print(f"   After NaN / zero RT removal             | {n_after_nan_rp:>5} ")
if _et_available:
    if n_premature_excluded > 0:
        print(f"   After pre-onset gaze exclusion          | {n_after_premature:>5} ")
    print(f"   After no-picture-dwell exclusion (ET)   | {n_after_et:>5} ")
if DROP_INSTRUCTION_ERRORS:
    print(f"   After instruction-error removal         | {n_after_err:>5} ")
print(f"   After RT<150ms removal                  | {n_after_cutoff:>5} ")
print(f"   After mean+/-2.5SD outlier removal      | {n_after_sd:>5} ")
if low_subj:
    print(f"   After excl. {len(low_subj)} subject(s) <70% valid    | {n_final:>5} ")
print(f"   FINAL (for LMER)                        | {n_final:>5} ")

# === Plot exclusion counts ===
funnel_labels = ["Raw", "NaN / no response"]
funnel_vals   = [n_raw, n_after_nan_rp]
if _et_available and n_premature_excluded > 0:
    funnel_labels.append("Pre-onset gaze")
    funnel_vals.append(n_after_premature)
if _et_available and n_et_excluded > 0:
    funnel_labels.append("No picture dwell")
    funnel_vals.append(n_after_et)
if DROP_INSTRUCTION_ERRORS and n_after_err < n_after_et:
    funnel_labels.append("Error trials")
    funnel_vals.append(n_after_err)
funnel_labels += ["RT < 150 ms", "±2.5 SD outliers"]
funnel_vals   += [n_after_cutoff, n_after_sd]
if low_subj:
    funnel_labels.append("Subjects < 70% valid")
    funnel_vals.append(n_final)

fig, ax = plt.subplots(figsize=(ps.W, 4.6))
bars = ax.bar(range(len(funnel_vals)), funnel_vals,
              color=ps.C_BAR, edgecolor="white", width=0.6)
for bar, val in zip(bars, funnel_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(funnel_labels)))
ax.set_xticklabels(funnel_labels, fontsize=9, rotation=30, ha="right")
ax.set_ylabel("Number of trials")

ax.set_ylim(0, max(funnel_vals) * 1.12)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
ps.save(fig, RESULTS_DIR / "dualpicture_00_preprocessing_funnel.png", verbose=False)
print("  Saved: dualpicture_00_preprocessing_funnel.png")

# === Prepare LMER variables ===
print("\nLog-transform RT + add LMER columns...")
df_clean = df_clean.copy()
df_clean["log_rt"] = np.log(df_clean["rt_ms"])

# Target side: -0.5 = left, +0.5 = right
df_clean["target_side_e"] = df_clean["target_side"].map({"left": -0.5, "right": 0.5})

# Standardize retained trial order within each participant
df_clean = df_clean.sort_values(["subject", "block", "trial"]).copy()
df_clean["trial_num"] = df_clean.groupby("subject").cumcount() + 1
df_clean["z_trial"] = df_clean.groupby("subject")["trial_num"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(f"  z_trial range: [{df_clean['z_trial'].min():.2f}, {df_clean['z_trial'].max():.2f}]")

print(f"  Mean RT (ms):  {df_clean['rt_ms'].mean():.1f}")
print(f"  Mean log(RT):  {df_clean['log_rt'].mean():.3f}")
print(f"  SD   log(RT):  {df_clean['log_rt'].std():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df_clean["rt_ms"],   bins=60, color="#4285F4", edgecolor="white", alpha=0.85)
axes[0].set_title("RT (ms) -- before transform")
axes[0].set_xlabel("RT (ms)")
axes[1].hist(df_clean["log_rt"], bins=60, color="#E8401C", edgecolor="white", alpha=0.85)
axes[1].set_title("log(RT) -- after transform")
axes[1].set_xlabel("log(RT)")
for ax in axes:
    ax.set_ylabel("Count")
plt.suptitle("Dual Picture Task -- RT distributions", fontsize=13)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "dualpicture_02_logrt_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: dualpicture_02_logrt_distribution.png")

# === Descriptive RT statistics ===
# M/SD/Mdn across participant cell means
desc_tv = subject_desc(df_clean, "target_valence")
print("\nRT (ms) by Target Valence (participant-level M/SD/Mdn):")
print(desc_tv.to_string())

_tv_cells = subject_means(df_clean, "target_valence")
pos_tv, neg_tv = _tv_cells["positive"], _tv_cells["negative"]
print(f"\nTarget valence effect: {neg_tv - pos_tv:+.1f} ms "
      "(negative - positive, per-participant means)")

# movement x valence
desc_mv = subject_desc(df_clean, ["movement", "target_valence"])
print("\nRT (ms) by Movement x Target Valence (participant-level M/SD/Mdn):")
print(desc_mv.to_string())

# congruency effect 
if "congruent" in df_clean["congruency"].values:
    _cg = subject_means(
        df_clean[df_clean["congruency"].isin(["congruent", "incongruent"])], "congruency")
    cong_rt, incong_rt = _cg["congruent"], _cg["incongruent"]
    print(f"\nCongruency effect: {incong_rt - cong_rt:+.1f} ms "
          "(incongruent - congruent, per-participant means)")

# === Plot sorted participant-mean RT ===
subj_means = df_clean.groupby("subject")["rt_ms"].mean().sort_values()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(subj_means)), subj_means.values, color="steelblue", alpha=0.8, edgecolor="white")
ax.axhline(subj_means.mean(), color="tomato", linestyle="--", linewidth=2,
           label=f"Grand mean = {subj_means.mean():.0f} ms")
ax.set_xlabel("Subject (sorted by mean RT)", fontsize=12)
ax.set_ylabel("Mean RT (ms)", fontsize=12)
ax.set_title("Dual Picture Task: Per-subject Mean RT", fontsize=13)
ax.set_xticks(range(len(subj_means)))
ax.set_xticklabels([str(s) for s in subj_means.index], rotation=45, fontsize=9)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "dualpicture_04_per_subject_rt.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: dualpicture_04_per_subject_rt.png")

# === Save preprocessed trial table ===
out_csv = RESULTS_DIR / "dualpicture_trial_table_preprocessed.csv"
df_clean.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv.name} ({len(df_clean)} trials, {df_clean['subject'].nunique()} subjects)")

print("\nDone. Next: run lmer_dualpicture.py")

