"""Apply the never-cane orchard mask, and keep the model's hands off intercropping.

`build_orchard_exclusion_mask.py` produces two layers and they do opposite jobs.

`orchard_exclusion_mask.gpkg` is orchard ground the 2025 national scan has never
seen carrying cane. Anything the model calls cane inside it is wrong by the only
independent record available, so it is removed outright, no model, no threshold.

`orchard_blocks_with_cane.gpkg` is the other half of the same screen: orchard blocks
that do overlap 2025 cane. In Multan and the rest of the mango belt growers plant
cane between the tree rows, so those blocks hold both at once. Cane growing under
mango never goes bare and never crashes at harvest the way an open field does, which
means every phenology rule we have reads it as woody and deletes it. So those blocks
become a protected layer: inside them the residual filter is not allowed to act, and
the decision falls back to the models that look at reflectance rather than shape.

The two layers together are the whole policy. Certain orchard comes out by rule;
uncertain orchard is left to the model; orchard that might be a cane field in
disguise is left alone entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SQM_PER_ACRE = 4046.8564224


@dataclass
class MaskResult:
    pixels_before: int
    pixels_removed: int
    acres_removed: float
    blocks_touching: int

    @property
    def removed_fraction(self) -> float:
        return self.pixels_removed / max(self.pixels_before, 1)


def rasterize_layer(path: Path | str, transform, shape, bounds=None) -> np.ndarray:
    """Burn a polygon layer onto a raster grid, reading only what the grid covers."""
    import geopandas as gpd
    from rasterio import features as rfeatures

    read_bbox = tuple(bounds) if bounds is not None else None
    layer = gpd.read_file(path, bbox=read_bbox, engine="pyogrio")
    if layer.empty:
        return np.zeros(shape, dtype=bool)
    layer = layer.to_crs(4326)
    burn = rfeatures.rasterize(layer.geometry, out_shape=shape, transform=transform,
                               fill=0, default_value=1, dtype="uint8")
    return burn.astype(bool)


def apply_mask(crop_map_path: Path | str, mask_path: Path | str,
               out_path: Path | str, crop_class: int = 1,
               background_class: int = 4) -> MaskResult:
    """Remove crop that falls inside the never-cane orchard mask.

    There is no threshold here on purpose. The mask was already screened against the
    national scan, so a pixel inside it is one the production map has never called
    cane in a year we trust. Adding a model on top would only give it a way to be
    wrong twice.
    """
    import rasterio

    with rasterio.open(crop_map_path) as src:
        crop = src.read(1)
        profile = src.profile.copy()
        transform, shape, bounds = src.transform, (src.height, src.width), src.bounds

    is_crop = crop == crop_class
    before = int(is_crop.sum())
    burn = rasterize_layer(mask_path, transform, shape, bounds)
    drop = is_crop & burn

    filtered = crop.copy()
    filtered[drop] = background_class
    profile.update(compress="lzw", tiled=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)
        dst.set_band_description(1, "crop_class_orchard_mask_applied")

    pixel_sqm = abs(transform.a * transform.e)
    result = MaskResult(before, int(drop.sum()),
                        float(drop.sum() * pixel_sqm / SQM_PER_ACRE), int(burn.any()))
    log.info("orchard mask removed %d of %d crop pixels (%.2f%%, %.1f acres)",
             result.pixels_removed, before, 100 * result.removed_fraction,
             result.acres_removed)
    return result
