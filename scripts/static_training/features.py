"""Feature computation shared by static-model training and inference.

Both paths must build feature vectors the same way, or the model silently sees a
different input space than it was fitted on. That is why every feature lives here
and neither the training notebook nor the execution pipeline computes its own.

The six quantities that arrive from the imagery are the Sentinel-2 bands B2, B3,
B4, B5 and B8 as raw reflectance DN (nominally 0-10000, baseline offset already
harmonised upstream by sentinel.py) plus NDVI carried as an integer scaled by
10000. Everything else is derived here.

Derived indices are also returned on the x10000 integer scale so that a feature
vector is dimensionally uniform and can be stored as uint16 without a second
convention to remember.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

#: The band set the deployed model was built on, and the default for readers that
#: do not say otherwise.
SOURCE_COLUMNS: tuple[str, ...] = ("B2", "B3", "B4", "B5", "B8", "NDVI")

#: With the two SWIR bands added. Sentinel-2 carries them at 20 m and sentinel.py
#: resamples them to the output grid like any other band. They matter in the months
#: when several crops are green at once and visible-to-NIR alone cannot tell them
#: apart, because canopy water content still differs.
SOURCE_COLUMNS_SWIR: tuple[str, ...] = (
    "B2", "B3", "B4", "B5", "B8", "B11", "B12", "NDVI",
)

#: Index scale used for NDVI on disk and for every derived index computed here.
INDEX_SCALE = 10000.0

#: Normalised-difference indices, as (high_band, low_band) pairs.
#: index = (high - low) / (high + low), matching sentinel.py INDEX_BANDS.
_NORMALISED_DIFFERENCES: Dict[str, tuple[str, str]] = {
    "GNDVI": ("B8", "B3"),   # green-based greenness
    "NDRE": ("B8", "B5"),    # red-edge chlorophyll
    "NDWI": ("B3", "B8"),    # open water / moisture
    "NDVI_CALC": ("B8", "B4"),  # recomputed NDVI, for cross-checking the stored one
    "NDMI": ("B8", "B11"),      # canopy moisture; needs SWIR
    "NBR": ("B8", "B12"),       # structure and burn; needs SWIR
    "MNDWI": ("B3", "B11"),     # open water, SWIR-based
}

#: Simple band ratios. Ratios cancel multiplicative illumination effects, which is
#: why they survive a change of acquisition date far better than raw DN.
_RATIOS: Dict[str, tuple[str, str]] = {
    "R43": ("B4", "B3"),
    "R84": ("B8", "B4"),
    "R85": ("B8", "B5"),
    "R28": ("B2", "B8"),
    "R1112": ("B11", "B12"),  # SWIR-only contrast, independent of the NIR bands
}

#: Every feature this module can produce.
AVAILABLE_FEATURES: tuple[str, ...] = (
    *SOURCE_COLUMNS_SWIR,
    *_NORMALISED_DIFFERENCES,
    *_RATIOS,
)


class FeatureError(ValueError):
    """Raised when a requested feature cannot be built from the given source."""


def _as_float(source: Dict[str, np.ndarray], band: str) -> np.ndarray:
    try:
        return source[band].astype(np.float32)
    except KeyError as exc:  # pragma: no cover - defensive
        raise FeatureError(f"source is missing band {band!r}") from exc


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise divide, yielding NaN instead of raising where the divisor is 0."""
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def compute_feature(name: str, source: Dict[str, np.ndarray]) -> np.ndarray:
    """Build one named feature from a mapping of band name to array.

    Arrays may be any shape; the result keeps that shape. Results are float32 on
    the x10000 scale, with NaN wherever the value is undefined.
    """
    if name in SOURCE_COLUMNS_SWIR:
        return _as_float(source, name)

    if name in _NORMALISED_DIFFERENCES:
        hi_band, lo_band = _NORMALISED_DIFFERENCES[name]
        hi, lo = _as_float(source, hi_band), _as_float(source, lo_band)
        return _safe_divide(hi - lo, hi + lo) * INDEX_SCALE

    if name in _RATIOS:
        num_band, den_band = _RATIOS[name]
        num, den = _as_float(source, num_band), _as_float(source, den_band)
        return _safe_divide(num, den) * INDEX_SCALE

    raise FeatureError(
        f"unknown feature {name!r}; available: {', '.join(AVAILABLE_FEATURES)}"
    )


def build_matrix(
    source: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Stack the named features into an (n_samples, n_features) matrix.

    `source` maps band name to a 1-D array of equal length. Column order follows
    `feature_names` exactly, because XGBoost matches features positionally.
    """
    if not feature_names:
        raise FeatureError("feature_names is empty")
    columns = [compute_feature(name, source) for name in feature_names]
    lengths = {col.shape for col in columns}
    if len(lengths) != 1:
        raise FeatureError(f"features have mismatched shapes: {lengths}")
    return np.stack(columns, axis=-1).astype(dtype)


def source_from_dataframe(
    frame, columns: Optional[Iterable[str]] = None
) -> Dict[str, np.ndarray]:
    """Adapt a pandas DataFrame of band columns into the mapping this module takes.

    With no explicit `columns`, every known source band present in the frame is
    taken. That way a table carrying SWIR feeds SWIR features without the caller
    having to say so, and a table without it still works for the six-band set.
    """
    if columns is None:
        present = [c for c in SOURCE_COLUMNS_SWIR if c in frame.columns]
        if not present:
            raise FeatureError(
                f"dataframe has none of the known source bands: {SOURCE_COLUMNS_SWIR}"
            )
        return {c: frame[c].to_numpy() for c in present}
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise FeatureError(f"dataframe is missing columns: {missing}")
    return {c: frame[c].to_numpy() for c in columns}


def source_from_raster(array: np.ndarray, band_order: Sequence[str]) -> Dict[str, np.ndarray]:
    """Adapt a (n_bands, height, width) raster read into the mapping this module takes.

    Bands are flattened so the result lines up with a per-pixel feature matrix.
    """
    if array.ndim != 3:
        raise FeatureError(f"expected a 3-D (band, row, col) array, got shape {array.shape}")
    if array.shape[0] != len(band_order):
        raise FeatureError(
            f"raster has {array.shape[0]} bands but band_order names {len(band_order)}"
        )
    return {name: array[i].reshape(-1) for i, name in enumerate(band_order)}


def finite_mask(matrix: np.ndarray) -> np.ndarray:
    """Rows of a feature matrix that are usable, i.e. finite in every column."""
    return np.isfinite(matrix).all(axis=1)


def describe(feature_names: Sequence[str]) -> List[str]:
    """Human-readable definitions, for logging into a model sidecar."""
    out: List[str] = []
    for name in feature_names:
        if name in SOURCE_COLUMNS_SWIR:
            out.append(f"{name}: source band, raw DN (NDVI is x{int(INDEX_SCALE)})")
        elif name in _NORMALISED_DIFFERENCES:
            hi, lo = _NORMALISED_DIFFERENCES[name]
            out.append(f"{name}: ({hi} - {lo}) / ({hi} + {lo}), x{int(INDEX_SCALE)}")
        elif name in _RATIOS:
            num, den = _RATIOS[name]
            out.append(f"{name}: {num} / {den}, x{int(INDEX_SCALE)}")
    return out
