"""
Dual Picture Task - Fixation timing check

Compare fixationCrossTime[t] and fixationCrossTime[t-1] with
pictureOnsetTime[t]. Check whether the timing supports the previous-index
window used for gaze-center estimation. No data are modified.
"""

import numpy as np
from pathlib import Path
import scipy.io as sio

DATA_DIR = Path(__file__).resolve().parents[2] / "DualPictureTask"
LONG_WINDOW_MS = 4000   # a pre-onset window longer than this looks implausible


def load_cell(name):
    d = sio.loadmat(str(DATA_DIR / f"{name}.mat"), squeeze_me=False, struct_as_record=False)
    key = [k for k in d if not k.startswith("_")][0]
    return d[key]


# === Load timing data ===
onsets = load_cell("pictureOnsetTime")
fix = load_cell("fixationCrossTime")
n_cols = onsets.shape[1]

# === Compare same index and previous-index timing ===
after_gaps = []          # fix[t] - onset[t]
used_windows = []        # onset[t] - fix[t-1]
naive_windows = []       # onset[t] - fix[t]
neg_used = 0

for c in range(n_cols):
    on = onsets[0, c].flatten().astype(float)
    fx = fix[0, c].flatten().astype(float)
    n = min(len(on), len(fx))
    for t in range(1, n):
        if np.isnan(on[t]) or np.isnan(fx[t]) or np.isnan(fx[t - 1]):
            continue
        after_gaps.append(fx[t] - on[t])
        w = on[t] - fx[t - 1]
        used_windows.append(w)
        naive_windows.append(on[t] - fx[t])
        if w <= 0:
            neg_used += 1

after_gaps = np.array(after_gaps)
used_windows = np.array(used_windows)
naive_windows = np.array(naive_windows)


def summ(a):
    return f"median={np.median(a):8.1f}  mean={np.mean(a):8.1f}  min={np.min(a):8.1f}  max={np.max(a):8.1f}"


# === Report timing differences (ms) ===
print(f"\nSessions (columns): {n_cols}")
print(f"Trial pairs checked: {len(used_windows)}\n")

print("1) Same-index offset: fix[t] - onset[t] (ms)")
print(f"   {summ(after_gaps)}")
print(f"   Fixation logged after picture onset: {100*np.mean(after_gaps > 0):.1f}% of trials")

print("2) Pipeline window: onset[t] - fix[t-1] (ms)")
print(f"   {summ(used_windows)}")
print(f"   Positive windows: {100*np.mean(used_windows > 0):.1f}%   negative: {neg_used}")
print(f"   Windows longer than {LONG_WINDOW_MS} ms: {int(np.sum(used_windows > LONG_WINDOW_MS))} "
      f"({100*np.mean(used_windows > LONG_WINDOW_MS):.1f}%)\n")

print("3) Same-index window: onset[t] - fix[t] (ms)")
print(f"   {summ(naive_windows)}")
print(f"   Negative (invalid) windows: {100*np.mean(naive_windows < 0):.1f}%")

# Timing criteria: >90% positive offsets and >95% positive pipeline windows
ok = (np.mean(after_gaps > 0) > 0.9) and (np.mean(used_windows > 0) > 0.95)
print("Timing check:", "pattern supports fix[t-1]; verify event meaning against task logs."
      if ok else "criteria not met; inspect timing and task logs.")
print("Long windows may span the previous trial. With pre-onset exclusion disabled,")
print("the pipeline uses this window for gaze-center estimation only.\n")