"""Take the orchards back out of a crop map, using the shape of the year.

The time-series model maps where cane grew and, along the way, picks up mango and
citrus blocks: both are green from June to September, and the model's attention sits
on exactly those windows. It cannot do better, because a tree was never one of its
output classes.

This runs after that model and before the static one. It reads the same smoothed
NDVI stack the pipeline already writes, so inference needs no new imagery, scores
each field on how perennial its trace looks, and drops the ones that are confidently
woody.

Two conditions have to hold before a field is dropped, and the second matters more
than it looks. The model score ranks the fields in front of it, so on its own it
always removes a top slice whether or not the area holds a single orchard: run over
an AOI with none, it took out 58 fields and inspection found cane and other crops
among them, no orchard at all. The absolute gate fixes that by asking whether the
thing being ranked is an orchard in the first place.

It is deliberately timid beyond that. The threshold in the sidecar is set to remove
as much orchard as it can while still keeping 99% of known cane fields, because a
mill would rather carry a few orchards in its acreage than lose real ones.

Decisions are made per object, not per pixel, and the object has to be the size of
an orchard. Given a delineation layer the objects are real fields. Given none, they
are patches of adjacent gate-passing pixels: grouping by mapped crop polygon instead
looks reasonable and fails completely, because a mapped cane polygon runs to
thousands of hectares while an orchard block has a median of 3.7, so the orchard
never moves the polygon's median and nothing is ever removed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import phenology as ph

log = logging.getLogger(__name__)

#: Units with fewer usable pixels than this are left alone rather than judged.
MIN_FIELD_PIXELS = 12

#: Smallest patch worth calling an orchard, in 10 m pixels. The delineated blocks
#: have a median of 3.7 ha, so 50 pixels (0.5 ha) is well under the real thing while
#: still rejecting speckle.
MIN_PATCH_PIXELS = 50

#: Necessary conditions a field must meet before its score is even consulted.
#:
#: A model score is relative: it ranks the fields in front of it, so a filter driven
#: by score alone always removes its top slice whether or not the area contains a
#: single orchard. Run over Al-Moiz Unit 1, which has none, that took out 58 fields
#: and inspection found no orchard among them, only cane and other crops.
#:
#: These are absolute instead: an orchard keeps a closed canopy all year, so it never
#: goes bare and its NDVI barely swings.
#:
#: The floor sits at 0.45 rather than 0.40 because of what the looser value let
#: through. At 0.40 six Al-Moiz fields cleared the gate and four were removed;
#: inspection found no orchard among them. All six had a minimum NDVI between 0.419
#: and 0.444, so moving the floor up by five hundredths takes every one of them out.
#: It costs less than it looks: known orchards passing go from 41.6% to 36.8%, and
#: the AOI that should lose nothing now loses nothing.
#:
#:   gate                              orchards passing   Al-Moiz fields passing
#:   ndvi_min > 0.40                              41.6%                        6
#:   ndvi_min > 0.45                              36.8%                        0
#:   ndvi_min > 0.50                              21.1%                        0
PERENNIAL_GATE = {
    "ndvi_min_above": 0.45,
    "amplitude_below": 0.45,
    "steps_below_bare_at_most": 2.0,
}


@dataclass
class FilterResult:
    filtered_path: Path
    score_path: Optional[Path]
    fields_scored: int
    fields_removed: int
    fields_passing_gate: int
    pixels_before: int
    pixels_after: int
    threshold: float

    @property
    def pixels_removed(self) -> int:
        return self.pixels_before - self.pixels_after

    @property
    def removed_fraction(self) -> float:
        return self.pixels_removed / self.pixels_before if self.pixels_before else 0.0


def load_sidecar(model_path: Path | str) -> dict:
    path = Path(model_path).with_suffix(".sidecar.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found; the detector needs its sidecar for the threshold")
    return json.loads(path.read_text())


def _field_ids(crop_mask: np.ndarray, transform, shape,
               field_polygons: Optional[Path]) -> np.ndarray:
    """An id per pixel identifying the field it belongs to, 0 outside the crop mask.

    With a delineation layer the fields are the real ones. Without, connected blobs
    of the crop class stand in: coarser, but still an object rather than a pixel.
    """
    if field_polygons is None:
        from scipy import ndimage

        labelled, count = ndimage.label(crop_mask, structure=np.ones((3, 3)))
        log.info("no field layer given; using %d connected blobs as fields", count)
        return labelled.astype(np.int32)

    import geopandas as gpd
    import rasterio
    from rasterio import features as rfeatures

    bounds = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    frame = gpd.read_file(field_polygons, bbox=(bounds[0], bounds[1], bounds[2], bounds[3]),
                          engine="pyogrio")
    log.info("field layer: %d polygons over this extent", len(frame))
    if frame.empty:
        return np.zeros(shape, dtype=np.int32)
    shapes = ((geom, i) for i, geom in enumerate(frame.geometry, start=1))
    ids = rfeatures.rasterize(shapes, out_shape=shape, transform=transform,
                              fill=0, dtype="int32", all_touched=False)
    return np.where(crop_mask, ids, 0).astype(np.int32)


def score_pixels(series_path: Path | str, field_ids_shape: tuple, crop_mask: np.ndarray,
                 model_path: Path | str, feature_names: Sequence[str],
                 gate: Dict[str, float], block_rows: int = 256):
    """Per-pixel orchard score and gate verdict over the crop mask.

    The stack is read in row blocks: phenology needs a pixel's whole time axis but
    nothing from its neighbours, so rows stream even though dates cannot.
    """
    import rasterio
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    height, width = field_ids_shape
    scores = np.full((height, width), np.nan, dtype=np.float32)
    passes = np.zeros((height, width), dtype=bool)
    gate_index = {name: list(feature_names).index(name)
                  for name in ("ndvi_min", "amplitude", "steps_below_bare")
                  if name in feature_names}

    with rasterio.open(series_path) as src:
        for start in range(0, height, block_rows):
            rows = min(block_rows, height - start)
            window = rasterio.windows.Window(0, start, width, rows)
            block = src.read(window=window).astype(np.float32)

            wanted = crop_mask[start:start + rows].reshape(-1)
            if not wanted.any():
                continue
            matrix = ph.build_from_raster(block, feature_names)
            usable = wanted & np.isfinite(matrix).all(axis=1)
            if not usable.any():
                continue

            candidate = matrix[usable]
            dmatrix = xgb.DMatrix(candidate, feature_names=list(feature_names))
            block_scores = np.full(rows * width, np.nan, dtype=np.float32)
            block_scores[usable] = booster.predict(dmatrix)
            scores[start:start + rows] = block_scores.reshape(rows, width)

            if len(gate_index) == 3:
                ok = (
                    (candidate[:, gate_index["ndvi_min"]] > gate["ndvi_min_above"])
                    & (candidate[:, gate_index["amplitude"]] < gate["amplitude_below"])
                    & (candidate[:, gate_index["steps_below_bare"]]
                       <= gate["steps_below_bare_at_most"])
                )
                block_gate = np.zeros(rows * width, dtype=bool)
                block_gate[usable] = ok
                passes[start:start + rows] = block_gate.reshape(rows, width)

    return scores, passes


def _units_from_patches(gate_pass: np.ndarray, min_pixels: int) -> np.ndarray:
    """Orchard-sized patches of gate-passing pixels, as the unit of decision.

    Without a delineation layer the obvious unit is a connected blob of the crop
    class, and that is what an earlier version used. It does not work: a mapped cane
    blob runs to thousands of hectares while an orchard block has a median of 3.7,
    so the orchard is a rounding error in the blob's median and never shows. Over
    five district chips that version cleared no field at all and removed nothing,
    even though 53% of the orchard pixels in Rahim Yar Khan clear the gate on their
    own.

    Grouping the gate-passing pixels themselves finds units at the scale of the
    thing being looked for, inside whatever larger polygon they happen to sit in.
    """
    from scipy import ndimage

    labelled, count = ndimage.label(gate_pass, structure=np.ones((3, 3)))
    if count == 0:
        return labelled.astype(np.int32)
    sizes = np.bincount(labelled.reshape(-1))
    too_small = np.flatnonzero(sizes < min_pixels)
    if too_small.size:
        drop = np.zeros(sizes.size, dtype=bool)
        drop[too_small] = True
        drop[0] = True
        labelled[drop[labelled]] = 0
    log.info("perennial patches at or above %d pixels: %d of %d",
             min_pixels, int(len(np.unique(labelled)) - 1), count)
    return labelled.astype(np.int32)


def filter_crop_map(
    crop_map_path: Path | str,
    series_path: Path | str,
    model_path: Path | str,
    out_filtered_path: Path | str,
    out_score_path: Optional[Path | str] = None,
    field_polygons: Optional[Path | str] = None,
    crop_class: int = 1,
    background_class: int = 4,
    threshold: Optional[float] = None,
    min_patch_pixels: int = MIN_PATCH_PIXELS,
    protect_polygons: Optional[Path | str] = None,
) -> FilterResult:
    """Remove confidently perennial ground from a crop classification raster.

    The unit of decision is whichever is available and finer. A delineation layer
    gives real fields, which is what the cane pipeline has over its mill AOIs.
    Without one, the unit is a patch of adjacent gate-passing pixels, sized like an
    orchard block rather than like the mapped polygon it sits inside.

    `protect_polygons` names ground the filter may not touch. It exists for the mango
    belt, where growers plant cane between the tree rows: that cane never goes bare
    and never crashes at harvest, so every phenology rule here reads it as woody and
    would delete a real field. Inside the protected layer the decision is left to the
    models that read reflectance instead of shape.
    """
    import rasterio

    sidecar = load_sidecar(model_path)
    feature_names = list(sidecar["feature_names"])
    if threshold is None:
        threshold = float(sidecar["decision_threshold"])
    gate = {**PERENNIAL_GATE, **(sidecar.get("perennial_gate") or {})}

    with rasterio.open(crop_map_path) as src:
        crop = src.read(1)
        profile = src.profile.copy()
        transform, shape = src.transform, (src.height, src.width)

    mask = crop == crop_class
    pixels_before = int(mask.sum())
    log.info("crop pixels in: %d", pixels_before)
    if pixels_before == 0:
        raise ValueError(f"{Path(crop_map_path).name} has no pixels of class {crop_class}")

    scores, gate_pass = score_pixels(series_path, shape, mask, model_path,
                                     feature_names, gate)
    gate_pass &= mask

    if protect_polygons is not None:
        from . import orchard_mask as om
        protected = om.rasterize_layer(protect_polygons, transform, shape,
                                       rasterio.open(crop_map_path).bounds)
        held = int((gate_pass & protected).sum())
        gate_pass &= ~protected
        log.info("intercropping protection held back %d gate-passing pixels", held)
    log.info("pixels clearing the perennial gate: %d (%.2f%% of the crop map)",
             int(gate_pass.sum()), 100 * gate_pass.sum() / pixels_before)

    if field_polygons is not None:
        ids = _field_ids(mask, transform, shape, Path(field_polygons))
        unit = "field"
    else:
        ids = _units_from_patches(gate_pass, min_patch_pixels)
        unit = "patch"

    labels_present = np.unique(ids)
    labels_present = labels_present[labels_present > 0]
    if labels_present.size == 0:
        log.info("nothing in this AOI looks perennial; the map comes back unchanged")

    # Aggregate to the unit: an orchard is an object, and a median over it is far
    # steadier than any single pixel.
    scored: Dict[int, float] = {}
    gated: Dict[int, bool] = {}
    flat_ids, flat_scores, flat_gate = ids.reshape(-1), scores.reshape(-1), gate_pass.reshape(-1)
    order = np.argsort(flat_ids, kind="stable")
    sorted_ids = flat_ids[order]
    edges = np.searchsorted(sorted_ids, labels_present, side="left")
    ends = np.searchsorted(sorted_ids, labels_present, side="right")
    for label, lo, hi in zip(labels_present, edges, ends):
        members = order[lo:hi]
        values = flat_scores[members]
        finite = values[np.isfinite(values)]
        if finite.size < MIN_FIELD_PIXELS:
            continue
        scored[int(label)] = float(np.median(finite))
        gated[int(label)] = bool(flat_gate[members].mean() >= 0.5)

    passing = sum(gated.values())
    log.info("%ss scored: %d, clearing the gate: %d (threshold %.2f)",
             unit, len(scored), passing, threshold)

    # Both conditions, deliberately. The score ranks; the gate decides whether the
    # thing being ranked is an orchard at all.
    remove = {label for label, score in scored.items()
              if score >= threshold and gated.get(label, False)}

    lookup = np.zeros(int(ids.max()) + 1, dtype=bool)
    for label in remove:
        lookup[label] = True
    drop = lookup[ids] & mask

    filtered = crop.copy()
    filtered[drop] = background_class
    profile.update(compress="lzw", tiled=True)
    Path(out_filtered_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_filtered_path, "w", **profile) as dst:
        dst.write(filtered, 1)
        dst.set_band_description(1, "crop_class_orchards_removed")

    score_path = None
    if out_score_path:
        # Keeping the per-pixel score makes every decision inspectable: a removed
        # patch can be put next to the imagery and checked rather than taken on trust.
        surface = np.where(mask, scores, np.nan).astype(np.float32)
        score_profile = {**profile, "dtype": "float32", "nodata": np.nan}
        with rasterio.open(out_score_path, "w", **score_profile) as dst:
            dst.write(surface, 1)
            dst.set_band_description(1, "orchard_score")
        score_path = Path(out_score_path)

    pixels_after = int((filtered == crop_class).sum())
    log.info("removed %d %ss, %d pixels (%.2f%% of the crop map)",
             len(remove), unit, pixels_before - pixels_after,
             100 * (pixels_before - pixels_after) / pixels_before)

    return FilterResult(Path(out_filtered_path), score_path, len(scored), len(remove),
                        int(passing), pixels_before, pixels_after, threshold)
