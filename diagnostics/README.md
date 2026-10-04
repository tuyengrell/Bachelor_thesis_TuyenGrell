# Diagnostics

Checks used while building the preprocessing. They do not change any data; the plot scripts save their figures to the results folder of the task.

- `verify_raw_vs_csv.py` - compares sampled trials in the trial tables with the raw data
- `CuedAAT/refined_alignment_check.py` - aligns the 98 eye-tracking trials with the 88
  behavioural trials per block
- `CuedAAT/plot_gaze_position.py`, `DualPictureTask/plot_gaze_position.py` - gaze traces
  around picture onset
- `DualPictureTask/plot_gaze_diagnostic_xposition.py` - pre-onset gaze shifts vs calibration drift
- `DualPictureTask/verify_fixationcross_timing.py` - timing of fixation cross and picture onset
