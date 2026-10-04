"""Shared RT helpers for the Cued AAT and Dual Picture Task analyses.

Always average per participant first, then across participants because the trial
counts differ too much between participants for a pooled mean.
"""

__all__ = ["subject_cells", "subject_means", "subject_desc"]


def subject_cells(df, by, val="rt_ms", subj="subject"):
    """One row per participant x condition. Base for the other helpers;
    plots use it directly too."""
    by = [by] if isinstance(by, str) else list(by)
    return df.groupby([subj] + by, observed=True)[val].mean().reset_index()


def subject_means(df, by, val="rt_ms", subj="subject"):
    """Mean of the per-participant means, by condition. Use this for any
    effect reported in the text."""
    by = [by] if isinstance(by, str) else list(by)
    cells = subject_cells(df, by, val=val, subj=subj)
    return cells.groupby(by, observed=True)[val].mean()


def subject_desc(df, by, val="rt_ms", subj="subject", decimals=2):
    """Descriptive table (M/SD/Mdn) on the participant level. SD is across
    participants, not trials."""
    by = [by] if isinstance(by, str) else list(by)
    cells = subject_cells(df, by, val=val, subj=subj)
    out = cells.groupby(by, observed=True)[val].agg(M="mean", SD="std", Mdn="median")
    out.insert(0, "n_subj", cells.groupby(by, observed=True)[val].size())  # participants per cell
    out.insert(0, "n_trials", df.groupby(by, observed=True)[val].size())  # raw trials per cell
    return out.round(decimals)
