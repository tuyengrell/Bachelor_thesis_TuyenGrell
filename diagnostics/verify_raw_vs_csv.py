"""
Compare sampled CSV trials with raw MATLAB values for both tasks.

Reconstruct trial indices, picture roles, RTs and selected labels. Also report
full-CSV ranges and mapping inconsistencies. No files are written.

Usage: python verify_raw_vs_csv.py [trials_per_task]  (default: 12)
"""

import sys
from pathlib import Path

import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stimulus_map import SEQ_TO_IAPS, IAPS_LOOKUP, POSITIVE_IDS   # noqa: E402

N_SHOW = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def load_mat(folder, name):
    d = sio.loadmat(str(folder / f"{name}.mat"))
    return d[[k for k in d if not k.startswith("_")][0]]


def ok(flag):
    return "ok " if flag else "** MISMATCH **"


# Cued AAT
print("Cued AAT: sampled raw vs CSV trials")

CT = ROOT / "CuedAAT"
csv = pd.read_csv(CT / "results" / "cuedtask_trial_table_preprocessed.csv")
rt_m  = load_mat(CT, "reactionTime")
cue_m = load_mat(CT, "cueSide")
pic_m = load_mat(CT, "pictureSequence")
order = pd.read_csv(CT / "results" / "block_order.csv").sort_values("col_idx")["order"].tolist()

FIRST_SUB, N_BLOCKS = 48, 2
problems = 0
rows = csv.sample(min(N_SHOW, len(csv)), random_state=1)

for _, r in rows.iterrows():
    # Two columns per participant, starting at ID 48; convert to zero-based indices.
    col   = (int(r["subject"]) - FIRST_SUB) * N_BLOCKS + (int(r["block"]) - 1)
    trial = int(r["trial"]) - 1

    raw_rt   = float(rt_m[trial, col])
    raw_cue  = int(cue_m[trial, col])
    raw_left = int(pic_m[trial, 0, col])
    raw_right= int(pic_m[trial, 1, col])
    raw_ord  = order[col]

    cue_dir = "left" if raw_cue == 0 else "right"
    tgt, dis = (raw_left, raw_right) if raw_cue == 0 else (raw_right, raw_left)
    first_half = trial < 44
    cond = ("congruent" if first_half else "incongruent") if raw_ord == "A" else \
           ("incongruent" if first_half else "congruent")
    val = "positive" if tgt in POSITIVE_IDS else "negative"

    checks = {
        "block_order":  (raw_ord,            r["block_order"]),
        "condition":    (cond,               r["condition"]),
        "cue_side":     (cue_dir,            r["cue_side"]),
        "target_pic":   (tgt,                int(r["target_pic"])),
        "distractor":   (dis,                int(r["distractor_pic"])),
        "target_val":   (val,                r["target_valence"]),
        "rt_ms":        (round(raw_rt*1000, 3), round(float(r["rt_ms"]), 3)),
        "iaps_target":  (IAPS_LOOKUP[tgt],   float(r["target_valence_iaps"])),
    }
    bad = [k for k, (a, b) in checks.items() if str(a) != str(b)]
    problems += len(bad)
    print(f"\nsubj {int(r['subject'])}  block {int(r['block'])}  trial {int(r['trial'])}  "
          f"(matrix column {col})   {ok(not bad)}")
    print(f"   raw : order={raw_ord}  cue={raw_cue}({cue_dir})  L={raw_left} R={raw_right}  "
          f"rt={raw_rt:.4f}s  -> target={tgt} ({val}, IAPS {SEQ_TO_IAPS[tgt]} = {IAPS_LOOKUP[tgt]})")
    print(f"   csv : order={r['block_order']}  cue_side={r['cue_side']}  "
          f"target={int(r['target_pic'])} distractor={int(r['distractor_pic'])}  "
          f"rt={float(r['rt_ms']):.1f}ms  cond={r['condition']}")
    for k in bad:
        print(f"     -> {k}: raw={checks[k][0]!r}  csv={checks[k][1]!r}")

# Dual Picture Task
print("\nDual Picture Task: sampled raw vs CSV trials")

DP = ROOT / "DualPictureTask"
csv2  = pd.read_csv(DP / "results" / "dualpicture_trial_table_preprocessed.csv")
rt2   = load_mat(DP, "reactionTime")
side2 = load_mat(DP, "side")
lpic  = load_mat(DP, "leftPictureID")
rpic  = load_mat(DP, "rightPictureID")

rows2 = csv2.sample(min(N_SHOW, len(csv2)), random_state=1)
for _, r in rows2.iterrows():
    col   = (int(r["subject"]) - 1) * 2 + (int(r["block"]) - 1)
    trial = int(r["trial"]) - 1
    if col >= rt2.shape[1]:
        print(f"\nsubj {int(r['subject'])} block {int(r['block'])}: column {col} out of range")
        problems += 1
        continue

    raw_rt = float(rt2[trial, col])
    raw_sd = int(side2[trial, col])
    raw_l  = int(lpic[trial, col])
    raw_r  = int(rpic[trial, col])
    tgt, dis   = (raw_l, raw_r) if raw_sd == 0 else (raw_r, raw_l)
    tside      = "left" if raw_sd == 0 else "right"
    val        = "positive" if tgt in POSITIVE_IDS else "negative"

    checks = {
        "target_side": (tside,               r["target_side"]),
        "target_pic":  (tgt,                 int(r["target_pic"])),
        "distractor":  (dis,                 int(r["distractor_pic"])),
        "target_val":  (val,                 r["target_valence"]),
        "rt_ms":       (round(raw_rt*1000, 3), round(float(r["rt_ms"]), 3)),
        "iaps_target": (IAPS_LOOKUP[tgt],    float(r["target_valence_iaps"])),
    }
    bad = [k for k, (a, b) in checks.items() if str(a) != str(b)]
    problems += len(bad)
    print(f"\nsubj {int(r['subject'])}  block {int(r['block'])}  trial {int(r['trial'])}  "
          f"(matrix column {col})   {ok(not bad)}")
    print(f"   raw : side={raw_sd}({tside})  L={raw_l} R={raw_r}  rt={raw_rt:.4f}s  "
          f"-> target={tgt} ({val}, IAPS {SEQ_TO_IAPS[tgt]} = {IAPS_LOOKUP[tgt]})")
    print(f"   csv : target_side={r['target_side']}  target={int(r['target_pic'])} "
          f"distractor={int(r['distractor_pic'])}  rt={float(r['rt_ms']):.1f}ms  "
          f"movement={r['movement']}  congruency={r['congruency']}")
    for k in bad:
        print(f"     -> {k}: raw={checks[k][0]!r}  csv={checks[k][1]!r}")

# Full-CSV diagnostics; these do not contribute to the sampled-trial problem count.
print("\nFull-CSV checks")

print(f"CuedTask    RT range {csv['rt_ms'].min():.0f}-{csv['rt_ms'].max():.0f} ms  "
      f"(plausible if roughly 150-3000)")
print(f"DualPicture RT range {csv2['rt_ms'].min():.0f}-{csv2['rt_ms'].max():.0f} ms")
print(f"CuedTask    participants {csv['subject'].min()}-{csv['subject'].max()}, "
      f"blocks {sorted(csv['block'].unique())}, trials 1-{csv['trial'].max()}")
print(f"DualPicture participants {csv2['subject'].min()}-{csv2['subject'].max()}, "
      f"blocks {sorted(csv2['block'].unique())}, trials 1-{csv2['trial'].max()}")
for name, d in [("CuedAAT", csv), ("DualPicture", csv2)]:
    ids = pd.concat([d["target_pic"], d["distractor_pic"]]).unique()
    outside = sorted(set(ids) - set(SEQ_TO_IAPS))
    print(f"{name:<11} picture IDs outside 1-87: {outside if outside else 'none'}")
    mism = d[d["target_pic"].map(IAPS_LOOKUP) != d["target_valence_iaps"]]
    print(f"{name:<11} rows where target_valence_iaps != stimulus_map: {len(mism)}")

print()
print("Sampled-trial check:", "no mismatches found" if problems == 0
      else f"{problems} field mismatches or out-of-range columns; see details above")