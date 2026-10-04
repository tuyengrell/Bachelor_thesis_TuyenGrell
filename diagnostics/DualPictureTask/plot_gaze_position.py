""" 
Dual Picture Task - Gaze position diagnostic

Plot left-eye X/Y traces for up to 12 randomly selected trials
from -200 to +400 ms around picture onset. Dotted lines mark
the horizontal fixation band (960 ±100 px; see Koch, 2025).
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
SCREEN_CENTER_PX = 960
CROSS_ROI_HALF   = 100
N_PLOT           = 12

random.seed(42)


def load_cell(name):
    d = sio.loadmat(str(DATA_DIR / f"{name}.mat"), squeeze_me=False, struct_as_record=False)
    k = [x for x in d if not x.startswith("_")][0]
    return d[k]

# === Load gaze data ===
print("Loading eye-tracking data...")
et_eye_data = load_cell("eyeData")
et_onsets   = load_cell("pictureOnsetTime")
et_time_vec = load_cell("timeVector")
n_cols      = et_eye_data.shape[1]

print(f"Generating gaze position plot ({n_cols} columns)...")

# === Sample trials with non-missing onset times ===
candidates = []
for c in range(n_cols):
    try:
        onsets_c = et_onsets[0, c].flatten().astype(float)
        t_vec_c  = et_time_vec[0, c].flatten().astype(float)
        if len(t_vec_c) == 0:
            continue
        for t in range(len(onsets_c)):
            if not np.isnan(onsets_c[t]):
                candidates.append((c, t))
    except Exception:
        continue

selected = random.sample(candidates, min(N_PLOT, len(candidates)))

# === Plot X/Y traces around picture onset ===
fig, axes = plt.subplots(len(selected), 2, figsize=(11, 2.5 * len(selected)), squeeze=False)

for i, (c, t) in enumerate(selected):
    eye_mat   = et_eye_data[0, c]
    t_vec_c   = et_time_vec[0, c].flatten().astype(float)
    onset_t   = float(et_onsets[0, c].flatten()[t])
    onset_abs = onset_t + t_vec_c[0]    # Convert session relative onset to absolute time

    mask  = (t_vec_c >= onset_abs - 200) & (t_vec_c <= onset_abs + 400)
    t_rel = t_vec_c[mask] - onset_abs
    gx    = eye_mat[mask, 0].astype(float)   # GazeX left eye
    gy    = eye_mat[mask, 1].astype(float)   # GazeY left eye
    subj  = c // N_BLOCKS_ASSUMED + 1

    ax_x = axes[i, 0]
    ax_y = axes[i, 1]

    ax_x.plot(t_rel, gx, color='#4285F4', linewidth=1.0)
    ax_x.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax_x.axhline(SCREEN_CENTER_PX + CROSS_ROI_HALF, color='#E8401C', linewidth=1.0,
                 linestyle=':', label='+ROI (1060px)')
    ax_x.axhline(SCREEN_CENTER_PX - CROSS_ROI_HALF, color='#E8401C', linewidth=1.0,
                 linestyle=':', label='-ROI (860px)')
    ax_x.set_ylabel('GazeX (px)', fontsize=8)
    ax_x.set_title(f'Subj {subj}  Trial {t + 1}', fontsize=8)
    ax_x.tick_params(labelsize=7)
    if i == 0:
        ax_x.legend(fontsize=7, loc='upper right')

    # GazeY
    ax_y.plot(t_rel, gy, color='#E8401C', linewidth=1.0)
    ax_y.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax_y.set_ylabel('GazeY (px)', fontsize=8)
    ax_y.set_title(f'Subj {subj}  Trial {t + 1}', fontsize=8)
    ax_y.tick_params(labelsize=7)

    if i == len(selected) - 1:
        ax_x.set_xlabel('Time from onset (ms)', fontsize=8)
        ax_y.set_xlabel('Time from onset (ms)', fontsize=8)

fig.suptitle(
    'Dual Picture Task: Gaze position per trial  [-200ms to +400ms]\n'
    'Left = GazeX (blue), Right = GazeY (red)  |  Orange dotted = fixation ROI boundary (860-1060 px)',
    fontsize=10, y=1.0
)

# === Save diagnostic plot ===
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS_DIR / 'dualpicture_et_gaze_position.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: dualpicture_et_gaze_position.png")
