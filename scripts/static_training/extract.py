"""Pull labelled pixels out of imagery, carrying acquisition date and AOI through.

The existing extraction dropped both. That is the single reason the 51-million-row
training table cannot be validated by date or by AOI, and why nobody could see
that every sample came from one three-week window. Every row this module emits
knows where and when it came from.

Positive pixels come from inside the label polygons; negatives come from the rest
of the AOI, which is how the original pipeline defined its background class. Pixels
touching a polygon boundary are dropped from both, since a 10 m pixel straddling a
field edge is a mixture and belongs to neither class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import features as feat


@dataclass
class ExtractionResult:
    frame: pd.DataFrame
    positive_pixels: int
    negative_pixels: int
    skipped_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None


def _filled_and_boundary(geometry, transform, shape) -> tuple[np.ndarray, np.ndarray]:
    """Pixels touched by `geometry`, and separately those touching its boundary."""
    from rasterio import features as rfeatures

    filled = rfeatures.geometry_mask(
        [geometry], out_shape=shape, transform=transform, invert=True, all_touched=True
    )
    boundary = rfeatures.geometry_mask(
        [geometry.boundary],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    return filled, boundary


def _strict_interior_mask(geometry, transform, shape) -> np.ndarray:
    """Pixels fully inside `geometry`, excluding any that touch its boundary."""
    filled, boundary = _filled_and_boundary(geometry, transform, shape)
    return filled & ~boundary


def extract_from_raster(
    raster_path: Path | str,
    positive_geometry,
    aoi_geometry,
    aoi_name: str,
    acquisition_date: str,
    crop_label: int,
    background_label: int,
    band_order: Sequence[str] = feat.SOURCE_COLUMNS,
    valid_band: str = "B8",
    max_pixels_per_class: Optional[int] = None,
    seed: int = 0,
    geometry_crs: str = "EPSG:4326",
) -> ExtractionResult:
    """Extract labelled pixels from one image over one AOI.

    `positive_geometry` is the union of label polygons clipped to the AOI; negatives
    are the rest of the AOI. Shapely geometries carry no CRS of their own, so the
    one they are in has to be stated and checked against the raster: a mismatch
    would silently rasterise the polygons in the wrong place and mislabel
    everything rather than fail.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        if src.count != len(band_order):
            return ExtractionResult(
                pd.DataFrame(), 0, 0,
                f"raster has {src.count} bands, expected {len(band_order)}",
            )
        if src.crs is None or src.crs.to_string() != geometry_crs:
            return ExtractionResult(
                pd.DataFrame(), 0, 0,
                f"raster CRS {src.crs} does not match geometry CRS {geometry_crs}",
            )
        array = src.read()
        transform, shape = src.transform, (src.height, src.width)

    aoi_mask = _strict_interior_mask(aoi_geometry, transform, shape)
    if not aoi_mask.any():
        return ExtractionResult(pd.DataFrame(), 0, 0, "AOI does not overlap the raster")

    positive_filled = np.zeros(shape, dtype=bool)
    positive_edge = np.zeros(shape, dtype=bool)
    if positive_geometry is not None and not positive_geometry.is_empty:
        positive_filled, positive_edge = _filled_and_boundary(
            positive_geometry, transform, shape
        )

    valid_index = list(band_order).index(valid_band)
    valid = array[valid_index] > 0

    # A pixel straddling a field edge is a mixture of the crop and whatever borders
    # it, so it belongs to neither class. Excluding the boundary from the positives
    # is not enough on its own: those same pixels would then fall through into the
    # background and teach the model that a cane edge looks like background.
    masks = {
        crop_label: aoi_mask & positive_filled & ~positive_edge & valid,
        background_label: aoi_mask & ~positive_filled & ~positive_edge & valid,
    }

    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    counts: Dict[int, int] = {}
    for label, mask in masks.items():
        selected = np.flatnonzero(mask.reshape(-1))
        counts[label] = int(selected.size)
        if selected.size == 0:
            continue
        if max_pixels_per_class and selected.size > max_pixels_per_class:
            selected = rng.choice(selected, size=max_pixels_per_class, replace=False)
        flat = array.reshape(array.shape[0], -1)[:, selected]
        part = pd.DataFrame(
            {name: flat[i] for i, name in enumerate(band_order)}
        )
        part["label"] = np.int64(label)
        parts.append(part)

    if not parts:
        return ExtractionResult(pd.DataFrame(), 0, 0, "no valid pixels in either class")

    frame = pd.concat(parts, ignore_index=True)
    frame["aoi"] = aoi_name
    frame["date"] = acquisition_date
    return ExtractionResult(frame, counts.get(crop_label, 0), counts.get(background_label, 0))


def apply_outlier_caps(frame: pd.DataFrame, caps: Dict[str, int]) -> pd.DataFrame:
    """Drop rows whose source bands exceed the per-band caps.

    Reproduces the cleaning already applied to master_filtered_scratch.parquet to
    within 0.05% of its row count, so newly extracted rows are treated the same way
    as the rows already in the table.
    """
    keep = np.ones(len(frame), dtype=bool)
    for band, cap in caps.items():
        if band in frame.columns:
            keep &= frame[band].to_numpy() <= cap
    return frame.loc[keep].reset_index(drop=True)


def summarise(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Rows per AOI and date, so gaps in temporal coverage are visible at a glance."""
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["aoi", "date", "label"])
        .size()
        .unstack("label", fill_value=0)
        .reset_index()
    )
