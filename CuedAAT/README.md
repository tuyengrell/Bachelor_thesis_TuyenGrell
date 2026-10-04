# Cued AAT

- `preprocess_lmer_cuedtask.py` - gaze-based and accuracy exclusions, RT cleaning,
  intensity coding; writes the trial table and the exclusion funnel
- `lmer_cuedtask.py` - linear mixed model, intensity
  robustness checks

Expected raw files in this folder: `reactionTime`, `cueSide`, `pictureSequence`,
`reactionPerformed`, `pictureTimeIdx`, `fixationTimeIdx`, `timeVectorContinuos`,
`x/yPositionLeft/RightContinuos` (`.mat`) and `results/block_order.csv`.
