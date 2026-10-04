""" 
Dual Picture Task - Pre-onset gaze diagnostic

Plot horizontal gaze from 50 ms before fixation onset to 400 ms
after picture onset. Show the first flagged trials, then randomly
sample unflagged trials if fewer than 12 are flagged.

For visual inspection of rapid gaze shifts and persistent offsets. This script does not exclude trials.
Keep gaze-detection helpers aligned with preprocess_lmer_dualpicture.py.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path

# === Settings ===
DATA_DIR    = Path(__file__).resolve().parents[2] / "DualPictureTask"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_BLOCKS_ASSUMED = 2
SCREEN_CENTER_PX = 960    # px
CROSS_ROI_HALF   = 100    # fixation cross region: center +/- 100px
LEAVE_SAMPLES    = 50     # End dwell after >50 samples outside the center band
VEL_THRESHOLD    = 5      # saccade velocity (px/sample)
MIN_PIC_STABLE   = 20     # Stability window; at least 80% valid samples
MAX_VAR_PX       = 50     # Max gaze range in window (px)
N_DIAG           = 12

random.seed(7)

# === Gaze detection helpers ===
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


def load_cell(name):
    d = sio.loadmat(str(DATA_DIR / f"{name}.mat"), squeeze_me=False, struct_as_record=False)
    k = [x for x in d if not x.startswith("_")][0]
    return d[k]

# === Load gaze data ===
print("Loading eye-tracking data...")
et_eye_data  = load_cell("eyeData")
et_onsets    = load_cell("pictureOnsetTime")
et_fix_times = load_cell("fixationCrossTime")
et_time_vec  = load_cell("timeVector")
n_cols       = et_eye_data.shape[1]

print(f"Checking pre-onset gaze ({n_cols} columns)...")


def xavg_col(eye_c):
    """Mask off-screen values; average eyes or use the available eye."""
    xl = eye_c[:, 0].astype(float)
    xr = eye_c[:, 3].astype(float)
    xl = np.where((xl < 0) | (xl > 1920), np.nan, xl)
    xr = np.where((xr < 0) | (xr > 1920), np.nan, xr)
    return np.where(~np.isnan(xl) & ~np.isnan(xr), (xl + xr) / 2.0,
                    np.where(~np.isnan(xl), xl, xr))

# === Flag pre-onset gaze shifts ===
flagged, other = [], []
for c in range(n_cols):
    try:
        onsets_c = et_onsets[0, c].flatten().astype(float)
        fix_c    = et_fix_times[0, c].flatten().astype(float)
        tvec_c   = et_time_vec[0, c].flatten().astype(float)
        if len(tvec_c) == 0:
            continue
        xa = xavg_col(et_eye_data[0, c])
        # Fixation onset at t-1 precedes picture onset at t
        for t in range(1, len(onsets_c)):
            if np.isnan(onsets_c[t]) or np.isnan(fix_c[t - 1]):
                continue
            pic_ms   = float(onsets_c[t]) + tvec_c[0]
            cross_ms = float(fix_c[t - 1]) + tvec_c[0]
            cmask    = (tvec_c >= cross_ms) & (tvec_c < pic_ms)
            ctr      = find_trial_center(xa[cmask])
            flag     = had_premature_gaze(xa, tvec_c, cross_ms, pic_ms, ctr)
            (flagged if flag else other).append((c, t, cross_ms, pic_ms, ctr, flag))
    except Exception:
        continue

# Prioritize flagged trials; fill remaining slots randomly
sel = flagged[:N_DIAG]
if len(sel) < N_DIAG and other:
    sel += random.sample(other, min(N_DIAG - len(sel), len(other)))

# === Plot fixation and post-onset gaze ===
if sel:
    fig, axes = plt.subplots(len(sel), 1, figsize=(10, 2.0 * len(sel)), squeeze=False)
    for i, (c, t, cross_ms, pic_ms, ctr, flag) in enumerate(sel):
        tvec_c = et_time_vec[0, c].flatten().astype(float)
        xa     = xavg_col(et_eye_data[0, c])
        w      = (tvec_c >= cross_ms - 50) & (tvec_c <= pic_ms + 400)
        trel   = tvec_c[w] - pic_ms
        ax = axes[i, 0]
        ax.plot(trel, xa[w], color='#4285F4', linewidth=1.0)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(cross_ms - pic_ms, color='green', linewidth=0.8, linestyle=':')
        ax.axhline(ctr, color='gray', linewidth=0.8)
        ax.axhline(ctr + CROSS_ROI_HALF, color='#E8401C', linewidth=0.8, linestyle=':')
        ax.axhline(ctr - CROSS_ROI_HALF, color='#E8401C', linewidth=0.8, linestyle=':')
        subj = c // N_BLOCKS_ASSUMED + 1
        ax.set_title(f'Subj {subj}  Trial {t + 1}  '
                     f'{"FLAGGED pre-onset saccade" if flag else "kept"}  '
                     f'(trial center={ctr:.0f}px)', fontsize=8)
        ax.set_ylabel('GazeX (px)', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel(
        'Time from picture onset (ms)   |   green dotted = fixation cross onset',
        fontsize=8)
    fig.suptitle(
        'Dual Picture Task Diagnostic: GazeX from fixation cross onset to post picture onset\n'
        'Saccade = fast step across ROI; drift = constant offset (no step).  '
        'Gray = trial center, orange dotted = +/-100px ROI, Black: picture onset',
        fontsize=10, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(RESULTS_DIR / 'dualpicture_et_diagnostic_xposition.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dualpicture_et_diagnostic_xposition.png "
          f"({len(flagged)} trials would be flagged pre-onset; showing {len(sel)})")
else:
    print("  No trials available for diagnostic plot.")
