"""Shape of the year, for telling an annual crop from a perennial one.

Sugarcane and a mango or citrus block are both green from June to September, which
is why a classifier reading colour on one date confuses them, and why a classifier
reading raw NDVI values confuses them too when its attention sits on the late-season
windows. What separates them is not any single value but the shape of the whole
series: cane is planted or ratooned, ramps, peaks and then collapses at harvest,
while an orchard sits high and nearly flat all year.

Measured on the Al-Moiz test feature, cane pixels have a median season amplitude of
0.505 and a median sharpest 8-day fall of -0.110. A perennial sits near 0.15 to 0.25
amplitude and rarely drops below 0.45 at all.

Every feature here is scale-free in time: it reads the series as a shape, not as a
calendar, so the same definition works on a stack of 35 windows or 37. Positional
features are returned as a fraction of the series length for the same reason.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

#: The NDVI level below which a field is not carrying a closed canopy. Cane crosses
#: it at harvest and again before the ratoon establishes; an orchard does not.
BARE_LEVEL = 0.45

#: The level above which a canopy counts as fully green.
GREEN_LEVEL = 0.60

#: Fewer valid observations than this and the shape of the year is not measurable.
MIN_VALID_OBSERVATIONS = 8


def _nan_safe(values: np.ndarray) -> np.ndarray:
    """Series as (n_dates, n_pixels) float, with non-finite entries as NaN."""
    out = np.asarray(values, dtype=np.float32)
    if out.ndim == 1:
        out = out[:, None]
    return np.where(np.isfinite(out), out, np.nan)


def _percentile(series: np.ndarray, q: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanpercentile(series, q, axis=0)


def _first_differences(series: np.ndarray) -> np.ndarray:
    return np.diff(series, axis=0)


def _position_of(index: np.ndarray, n_steps: int) -> np.ndarray:
    """Where in the series an event falls, as a fraction from 0 to 1."""
    return index.astype(np.float32) / max(n_steps - 1, 1)


def amplitude(series: np.ndarray) -> np.ndarray:
    """Seasonal swing, robust to a single bad observation at either end.

    The strongest single discriminator measured: an annual crop swings roughly half
    an NDVI unit over its cycle, a perennial a fifth of that.
    """
    return _percentile(series, 95) - _percentile(series, 5)


def ndvi_min(series: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmin(series, axis=0)


def ndvi_max(series: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmax(series, axis=0)


def ndvi_mean(series: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(series, axis=0)


def ndvi_std(series: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanstd(series, axis=0)


def interquartile(series: np.ndarray) -> np.ndarray:
    return _percentile(series, 75) - _percentile(series, 25)


def sharpest_fall(series: np.ndarray) -> np.ndarray:
    """Most negative single step. Harvest is abrupt; senescence is not."""
    with np.errstate(invalid="ignore"):
        return np.nanmin(_first_differences(series), axis=0)


def sharpest_rise(series: np.ndarray) -> np.ndarray:
    """Most positive single step, which for cane is the ratoon or the planting flush."""
    with np.errstate(invalid="ignore"):
        return np.nanmax(_first_differences(series), axis=0)


def steps_below_bare(series: np.ndarray) -> np.ndarray:
    """How much of the year the canopy is open. Zero for an established orchard."""
    return np.nansum(series < BARE_LEVEL, axis=0).astype(np.float32)


def fraction_above_green(series: np.ndarray) -> np.ndarray:
    """Share of the year spent fully green. Near 1 for a perennial."""
    valid = np.isfinite(series).sum(axis=0).astype(np.float32)
    green = np.nansum(series > GREEN_LEVEL, axis=0).astype(np.float32)
    return np.divide(green, valid, out=np.full_like(green, np.nan), where=valid > 0)


def roughness(series: np.ndarray) -> np.ndarray:
    """Spread of the step-to-step changes: how restless the trace is."""
    with np.errstate(invalid="ignore"):
        return np.nanstd(_first_differences(series), axis=0)


def trough_position(series: np.ndarray) -> np.ndarray:
    """Where the minimum falls. Cane's trough is its harvest, an orchard's is winter."""
    filled = np.where(np.isfinite(series), series, np.inf)
    return _position_of(np.argmin(filled, axis=0), series.shape[0])


def peak_position(series: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(series), series, -np.inf)
    return _position_of(np.argmax(filled, axis=0), series.shape[0])


def green_span(series: np.ndarray) -> np.ndarray:
    """Longest unbroken run above the green level, as a fraction of the series."""
    green = np.isfinite(series) & (series > GREEN_LEVEL)
    n_steps, n_pixels = green.shape
    best = np.zeros(n_pixels, dtype=np.float32)
    run = np.zeros(n_pixels, dtype=np.float32)
    for step in range(n_steps):
        run = np.where(green[step], run + 1.0, 0.0)
        best = np.maximum(best, run)
    return best / n_steps


#: Every feature, in a fixed order. Keys are what a model's sidecar records.
FEATURES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "amplitude": amplitude,
    "ndvi_min": ndvi_min,
    "ndvi_max": ndvi_max,
    "ndvi_mean": ndvi_mean,
    "ndvi_std": ndvi_std,
    "interquartile": interquartile,
    "sharpest_fall": sharpest_fall,
    "sharpest_rise": sharpest_rise,
    "steps_below_bare": steps_below_bare,
    "fraction_above_green": fraction_above_green,
    "roughness": roughness,
    "trough_position": trough_position,
    "peak_position": peak_position,
    "green_span": green_span,
}

FEATURE_NAMES: tuple[str, ...] = tuple(FEATURES)


class PhenologyError(ValueError):
    """Raised when a series cannot produce the requested features."""


def build_matrix(series: np.ndarray, feature_names: Sequence[str] = FEATURE_NAMES,
                 dtype: np.dtype = np.float32) -> np.ndarray:
    """Features for every pixel of a (n_dates, n_pixels) smoothed NDVI series.

    Returns (n_pixels, n_features) in the order given, because the models that
    consume this match features positionally.
    """
    values = _nan_safe(series)
    if values.shape[0] < 3:
        raise PhenologyError(f"need at least 3 dates, got {values.shape[0]}")
    unknown = [n for n in feature_names if n not in FEATURES]
    if unknown:
        raise PhenologyError(f"unknown features {unknown}; available: {list(FEATURES)}")

    columns = [FEATURES[name](values) for name in feature_names]
    matrix = np.stack(columns, axis=-1).astype(dtype)

    # Counting and positional features answer even when there is nothing to count:
    # an all-nodata pixel gets zero steps below bare and a trough at position 0, which
    # look like ordinary values and would be classified rather than skipped. Blank the
    # whole row wherever the series carries too little to describe.
    usable = np.isfinite(values).sum(axis=0) >= MIN_VALID_OBSERVATIONS
    matrix[~usable, :] = np.nan
    return matrix


def build_from_raster(array: np.ndarray,
                      feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """Same, from a (n_dates, height, width) stack; rows follow row-major order."""
    if array.ndim != 3:
        raise PhenologyError(f"expected (date, row, col), got shape {array.shape}")
    n_dates = array.shape[0]
    return build_matrix(array.reshape(n_dates, -1), feature_names)


def build_from_dataframe(frame, date_columns: Sequence[str],
                         feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """Same, from a table whose columns are the dates of the series."""
    missing = [c for c in date_columns if c not in frame.columns]
    if missing:
        raise PhenologyError(f"dataframe is missing date columns: {missing[:5]}")
    return build_matrix(frame[list(date_columns)].to_numpy(dtype=np.float32).T, feature_names)


def describe(feature_names: Sequence[str] = FEATURE_NAMES) -> List[str]:
    """One line per feature, for a model sidecar."""
    text = {
        "amplitude": "95th minus 5th percentile of the season",
        "ndvi_min": "lowest value in the season",
        "ndvi_max": "highest value in the season",
        "ndvi_mean": "mean over the season",
        "ndvi_std": "standard deviation over the season",
        "interquartile": "75th minus 25th percentile",
        "sharpest_fall": f"most negative single step",
        "sharpest_rise": "most positive single step",
        "steps_below_bare": f"count of observations below {BARE_LEVEL}",
        "fraction_above_green": f"share of valid observations above {GREEN_LEVEL}",
        "roughness": "standard deviation of the step-to-step changes",
        "trough_position": "where the minimum falls, 0 at the start of the series",
        "peak_position": "where the maximum falls, 0 at the start of the series",
        "green_span": f"longest unbroken run above {GREEN_LEVEL}, as a fraction",
    }
    return [f"{name}: {text[name]}" for name in feature_names if name in text]
