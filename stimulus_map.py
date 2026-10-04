"""
Map experiment sequence IDs (1–87) to IAPS IDs and valence norms.

IDs were matched visually; sequence order does not follow IAPS numbering.
Norms: Lang et all (2008), Table 1 (all subjects).
IAPS 2058 was not presented. For IAPS 1610, the selected norm is 7.82.
IAPS images are not included; see the thesis Methods for usage restrictions.
"""

# Sequence ID -> IAPS picture ID
SEQ_TO_IAPS = {
    # Pleasant (1–43)
    1: 1340,  2: 1410,  3: 5210,  4: 4640,  5: 4641,  6: 5200,  7: 5202,
    8: 1620,  9: 2598, 10: 4599, 11: 4614, 12: 4628, 13: 2057, 14: 2260,
    15: 2347, 16: 2540, 17: 1610, 18: 2388, 19: 2158, 20: 2387, 21: 2360,
    22: 2391, 23: 8497, 24: 2398, 25: 2392, 26: 2332, 27: 2340, 28: 2341,
    29: 2345, 30: 2299, 31: 2300, 32: 2314, 33: 2306, 34: 2170, 35: 2208,
    36: 2274, 37: 2040, 38: 2075, 39: 2150, 40: 2160, 41: 1500, 42: 1540,
    43: 1600,
    # Unpleasant (44–87)
    44: 9184, 45: 6540, 46: 6350, 47: 9254, 48: 9414, 49: 6550, 50: 2456,
    51: 3300, 52: 9419, 53: 9941, 54: 9940, 55: 9925, 56: 9902, 57: 9905,
    58: 9908, 59: 9909, 60: 9810, 61: 9830, 62: 9832, 63: 9900, 64: 9530,
    65: 9560, 66: 9630, 67: 9800, 68: 9342, 69: 9424, 70: 9427, 71: 9520,
    72: 9220, 73: 9250, 74: 9332, 75: 9340, 76: 6838, 77: 9000, 78: 9075,
    79: 9140, 80: 3181, 81: 3230, 82: 6560, 83: 6563, 84: 2053, 85: 2301,
    86: "2345.1", 87: 2751,
}

# IAPS ID -> valence (SAM: 1 = most unpleasant, 9 = most pleasant)
IAPS_VALENCE = {
    # Pleasant
    1340: 7.13, 1410: 7.00, 1500: 7.24, 1540: 7.15, 1600: 7.37, 1610: 7.82,
    1620: 7.37, 2040: 8.17, 2057: 7.81, 2075: 7.32, 2150: 7.92, 2158: 7.31,
    2160: 7.58, 2170: 7.55, 2208: 7.35, 2260: 8.06, 2274: 7.47, 2299: 7.27,
    2300: 7.04, 2306: 7.08, 2314: 7.55, 2332: 7.64, 2340: 8.03, 2341: 7.38,
    2345: 7.41, 2347: 7.83, 2360: 7.70, 2387: 7.12, 2388: 7.44, 2391: 7.11,
    2392: 6.15, 2398: 7.48, 2540: 7.63, 2598: 7.19, 4599: 7.12, 4614: 7.15,
    4628: 7.23, 4640: 7.18, 4641: 7.20, 5200: 7.36, 5202: 7.25, 5210: 8.03,
    8497: 7.26,
    # Unpleasant
    2053: 2.47, 2301: 2.78, "2345.1": 2.26, 2456: 2.84, 2751: 2.67,
    3181: 2.30, 3230: 2.02, 3300: 2.74, 6350: 1.90, 6540: 2.19, 6550: 2.73,
    6560: 2.16, 6563: 1.77, 6838: 2.45, 9000: 2.55, 9075: 1.66, 9140: 2.19,
    9184: 2.47, 9220: 2.06, 9250: 2.57, 9254: 2.03, 9332: 2.25, 9340: 2.41,
    9342: 2.85, 9414: 2.06, 9419: 2.55, 9424: 2.87, 9427: 2.89, 9520: 2.46,
    9530: 2.93, 9560: 2.12, 9630: 2.96, 9800: 2.04, 9810: 2.09, 9830: 2.54,
    9832: 2.94, 9900: 2.46, 9902: 2.33, 9905: 2.55, 9908: 2.34, 9909: 2.78,
    9925: 2.84, 9940: 1.62, 9941: 2.91,
}

# Sequence IDs by valence
POSITIVE_IDS = set(range(1, 44))    # 43 pleasant pictures
NEGATIVE_IDS = set(range(44, 88))   # 44 unpleasant pictures

# Sequence ID -> valence
IAPS_LOOKUP = {seq: IAPS_VALENCE[pid] for seq, pid in SEQ_TO_IAPS.items()}


def _validate():
    """Check mapping coverage, unique pictures, and valence categories."""
    seqs = sorted(SEQ_TO_IAPS)
    assert seqs == list(range(1, 88)), \
        f"sequence IDs must be 1..87 with no gaps, got {len(seqs)} entries"
    pics = list(SEQ_TO_IAPS.values())
    assert len(pics) == len(set(pics)), \
        "the same IAPS picture is mapped to more than one sequence ID"
    missing = [p for p in pics if p not in IAPS_VALENCE]
    assert not missing, f"no Table 1 valence for IAPS picture(s) {missing}"
    assert 2058 not in pics, \
        "IAPS 2058 is not presented in these experiments and must not be mapped"
    assert len(POSITIVE_IDS) == 43 and len(NEGATIVE_IDS) == 44, \
        "expected 43 pleasant and 44 unpleasant sequence IDs"
    for seq in POSITIVE_IDS:
        assert IAPS_LOOKUP[seq] > 5.0, \
            f"sequence ID {seq} is pleasant but has valence {IAPS_LOOKUP[seq]}"
    for seq in NEGATIVE_IDS:
        assert IAPS_LOOKUP[seq] < 5.0, \
            f"sequence ID {seq} is unpleasant but has valence {IAPS_LOOKUP[seq]}"


_validate()
