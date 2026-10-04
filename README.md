# Distractor valence intensity and approach-avoidance under spatial cueing

Analysis code for the Bachelor's thesis *"Does distractor valence intensity modulate
approach-avoidance behavior under spatial cueing?"* (Tuyen Grell, Institute of Cognitive
Science, University of Osnabrück, 2026).

## Structure

- `CuedAAT/` - preprocessing and linear mixed model, Cued AAT (H1, H2)
- `DualPictureTask/` - preprocessing and linear mixed model, Dual Picture Task
- `SinglePictureTask/` - preprocessing, Single Picture Task (descriptive comparison)
- `analyses/` - combined model (H3a-H3c) and follow-up analyses
- `figures/` - thesis figures
- `diagnostics/` - data checks used during preprocessing
- `stimulus_map.py` - sequence number mapped to IAPS picture and valence norm
- `analysis_common.py`, `plot_style.py` - shared helpers and figure style

## Run order

1. `CuedAAT/preprocess_lmer_cuedtask.py`, then `CuedAAT/lmer_cuedtask.py`
2. `DualPictureTask/preprocess_lmer_dualpicture.py`, then `DualPictureTask/lmer_dualpicture.py`
3. `SinglePictureTask/preprocess_lmer_singlepicture.py`
4. `analyses/combined_analysis.py`, then the other scripts in `analyses/`
5. scripts in `figures/`

## Requirements

Python 3.13 with numpy, pandas, scipy, statsmodels and matplotlib.

## Data

- Raw data are not included for data protection reasons
- IAPS pictures are licensed and not included (Lang et al. 2008).

