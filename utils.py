# utils.py
# This module contains utility functions for summarizing results.

from pathlib import Path
from scipy.stats import trim_mean, bootstrap
import numpy as np

def summarize(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return {}

    iqm = trim_mean(values, 0.25)

    ci = bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=10000,
        method="BCa",
        random_state=0,
    ).confidence_interval

    return {
        "mean": float(np.mean(values)),
        "iqm": float(iqm),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "bci95": [
            float(ci.low),
            float(ci.high),
        ],
        "n": int(len(values)),
    }

def print_summary(name, stats):

    print(name)
    print(
        f"  Mean      : {stats['mean']:.4f}"
    )
    print(
        f"  IQM       : {stats['iqm']:.4f}"
    )
    print(
        f"  Median    : {stats['median']:.4f}"
    )
    print(
        f"  Std       : {stats['std']:.4f}"
    )
    print(
        f"  95% BCI   : "
        f"[{stats['bci95'][0]:.4f}, "
        f"{stats['bci95'][1]:.4f}]"
    )
    print(
        f"  Min / Max : "
        f"{stats['min']:.4f} / "
        f"{stats['max']:.4f}"
    )
    print()