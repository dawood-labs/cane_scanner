"""Turn a 10 m crop map into clean field polygons the client can use.

The delineation was drawn on high-resolution basemap imagery and has the edges a
client wants. The crop map is 10 m Sentinel and knows only what is growing, not
where anything ends. The whole job rests on one rule:

    the delineation owns the geometry, the crop map owns only the label.

A raster edge is never allowed to become an output edge. Everything below follows
from that.

Measured over the Al-Moiz test AOI before any of this was designed:

    17,074 polygons, median 0.77 acres, 19,699 acres in total
    8,666 with invalid geometry, 11,088 overlapping pairs
    14.9% cleanly crop, 63.6% cleanly not, 21.5% mixed (5,213 acres)
    mixed polygons hold one contiguous piece of crop in 97% of cases
    858 acres of crop with no polygon over it at all

    python3 label_field_polygons.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

log = logging.getLogger("label_field_polygons")

CROPSCAN = SCRIPTS_DIR.parent
TEST = CROPSCAN / "data" / "test_data" / "almoiz_unit_1_test_feature_1"
CANE_DIR = TEST / "cane_2026"
CROP_MAP = CANE_DIR / "30_Aug_2026" / "v4" / "static_mosaic_30_Aug_2026_Cls_v4_sieved_p20.tif"
DELINEATION = (CROPSCAN / "data" / "Al_Moiz_Unit_1_field_delineation_COMPLETE"
               / "Al_Moiz_Unit_1_field_delineation_COMPLETE.shp")
OUT = CANE_DIR / "field_labelling"

CROP_CLASS = 1
UTM = 32642
SQM_PER_ACRE = 4046.8564224
PIXEL_SQM = 100.0

#: Above this a polygon is crop, below it is not, between the two it holds more than
#: one thing. 0.5 would be the naive cut; these leave a band that gets looked at
#: properly instead of being rounded away.
CROP_HIGH, CROP_LOW = 0.85, 0.15

#: Fewer pixels than this and the fraction is not a measurement.
MIN_PIXELS = 10

#: A cut has to earn its place. Splitting raises purity by 0.167 in the median case,
#: but on a polygon that is already 80% crop it can add almost nothing, and a cut
#: that adds nothing invents a boundary the ground does not have.
MIN_CUT_GAIN = 0.08

#: Crop with no polygon over it gets one, down to this size. Below it the blobs are
#: 10 m edge effects rather than fields: 5,300 of them hold 348 acres between them.
MIN_ORPHAN_ACRES = 0.15

#: Below this a piece is a topological sliver from the overlap arithmetic, not a
#: field. It has to be well under a pixel: a one-pixel floor discarded 952 acres of
#: small but real polygons on the first run.
SLIVER_SQM = 10.0

#: How far a derived polygon may be simplified, in metres. Half a pixel keeps the
#: shape honest while taking the staircase off a rasterised edge.
SIMPLIFY_M = 5.0


# --------------------------------------------------------------------- geometry

def repair(fields):
    """Make every polygon valid, single-part, and disjoint from its neighbours.

    Half the delineation fails `is_valid` and 11,088 pairs overlap. Overlaps are the
    worse of the two: a pixel claimed twice has no single answer, and the acreage a
    client adds up counts that ground twice.

    Overlaps are resolved in favour of the smaller polygon, which keeps the finest
    delineation intact and trims the coarser one around it. The alternative, letting
    the larger win, would erase exactly the detail the basemap tracing was done for.
    """
    import geopandas as gpd
    from shapely.validation import make_valid

    before = len(fields)
    fields = fields.copy()
    invalid = ~fields.geometry.is_valid
    if invalid.any():
        log.info("repairing %d invalid geometries", int(invalid.sum()))
        fields.loc[invalid, "geometry"] = fields.loc[invalid, "geometry"].apply(make_valid)

    # make_valid hands back GeometryCollections, and a collection can hold a
    # MultiPolygon inside it. One explode leaves those Multis intact, and filtering to
    # Polygon then silently discards 2,997 of them: 1,532 acres of crop that reappear
    # as orphan blobs with rasterised edges. Explode until nothing is nested.
    for _ in range(5):
        fields = fields.explode(index_parts=False)
        if not fields.geom_type.isin(["MultiPolygon", "GeometryCollection",
                                      "MultiLineString", "MultiPoint"]).any():
            break
    dropped_types = fields.geom_type[fields.geom_type != "Polygon"].value_counts().to_dict()
    fields = fields[fields.geometry.geom_type == "Polygon"]
    fields = fields[~fields.geometry.is_empty & fields.geometry.is_valid].reset_index(drop=True)
    log.info("after repair and explode: %d polygons (from %d); discarded %s",
             len(fields), before, dropped_types or "nothing")

    fields = fields.copy()
    if "fid" not in fields.columns:
        fields["fid"] = np.arange(1, len(fields) + 1)
    metric = fields.to_crs(UTM)
    # What the layer actually covers, counting overlapped ground once. The plain sum
    # of polygon areas is larger than this, and the difference is the double-count
    # that the client would otherwise be invoiced for.
    union_acres = float(metric.geometry.union_all().area / SQM_PER_ACRE)
    metric["area_m2"] = metric.area
    order = np.argsort(metric.area_m2.to_numpy())  # smallest first: they keep their shape

    index = metric.sindex
    geometries = list(metric.geometry)
    claimed: set = set()
    trimmed = 0
    for position in order:
        position = int(position)
        geom = geometries[position]
        if geom.is_empty:
            continue
        neighbours = [int(i) for i in index.query(geom, predicate="intersects")
                      if int(i) in claimed and int(i) != position]
        for other in neighbours:
            if geometries[other].is_empty:
                continue
            if geom.intersects(geometries[other]):
                new = geom.difference(geometries[other])
                if not new.equals(geom):
                    trimmed += 1
                geom = new
        geometries[position] = geom
        claimed.add(position)

    metric["geometry"] = geometries
    metric = metric.explode(index_parts=False)
    metric = metric[(metric.geom_type == "Polygon") & ~metric.geometry.is_empty]
    # Only true topological slivers go. A one-pixel floor looked reasonable and threw
    # away about a thousand acres of real, if small, fields.
    dropped = metric.area <= SLIVER_SQM
    log.info("overlap resolution trimmed %d polygons; dropped %d slivers holding %.1f acres",
             trimmed, int(dropped.sum()), float(metric.area[dropped].sum() / SQM_PER_ACRE))
    metric = metric[~dropped]
    log.info("%d disjoint parts remain, %.0f acres (input covered %.0f acres of ground)",
             len(metric), metric.area.sum() / SQM_PER_ACRE, union_acres)
    return metric.reset_index(drop=True)


# ---------------------------------------------------------------------- cutting

def _orientation(geom) -> float:
    """Angle of the polygon's long axis. Field subdivisions run parallel to it."""
    rect = geom.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return 0.0
    coords = np.array(rect.exterior.coords[:4])
    edges = np.diff(np.vstack([coords, coords[:1]]), axis=0)
    longest = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
    return float(np.arctan2(longest[1], longest[0]))


def _best_cut(xs, ys, is_crop, angles) -> Tuple[float, float, float]:
    """Where along which axis a single straight cut best separates the two classes.

    Returns the axis angle, the offset along it, and the purity achieved. Purity is
    the share of pixels on the correct side once each side takes its own majority.
    """
    best = (0.0, 0.0, 0.0)
    for theta in angles:
        projection = xs * np.cos(theta) + ys * np.sin(theta)
        order = np.argsort(projection)
        truth = is_crop[order].astype(float)
        n = truth.size
        if n < 2 * MIN_PIXELS:
            continue
        before = np.cumsum(truth)
        total = before[-1]
        left = np.arange(1, n + 1)
        right = n - left
        with np.errstate(invalid="ignore", divide="ignore"):
            left_share = before / left
            right_share = np.where(right > 0, (total - before) / np.maximum(right, 1), 0.0)
        purity = (left * np.maximum(left_share, 1 - left_share)
                  + right * np.maximum(right_share, 1 - right_share)) / n
        purity[:MIN_PIXELS] = 0.0
        purity[-MIN_PIXELS:] = 0.0
        position = int(np.nanargmax(purity))
        if purity[position] > best[2]:
            sorted_projection = projection[order]
            offset = float((sorted_projection[position] + sorted_projection[min(position + 1, n - 1)]) / 2)
            best = (theta, offset, float(purity[position]))
    return best


def _halves(geom, theta: float, offset: float):
    """The polygon either side of the line perpendicular to `theta` at `offset`."""
    from shapely.geometry import Polygon

    minx, miny, maxx, maxy = geom.bounds
    span = float(np.hypot(maxx - minx, maxy - miny)) + 10.0
    ux, uy = np.cos(theta), np.sin(theta)
    px, py = -uy, ux  # along the cut

    # Anchor the cut on the polygon itself. Taking the foot of the perpendicular from
    # the coordinate origin puts it hundreds of kilometres away in UTM, and the
    # rectangle then misses the field entirely.
    centre = geom.centroid
    along = centre.x * px + centre.y * py
    cx, cy = offset * ux + along * px, offset * uy + along * py

    def side(sign: float):
        a = (cx + px * span, cy + py * span)
        b = (cx - px * span, cy - py * span)
        c = (b[0] + sign * ux * span, b[1] + sign * uy * span)
        d = (a[0] + sign * ux * span, a[1] + sign * uy * span)
        return Polygon([a, b, c, d])

    return geom.intersection(side(+1.0)), geom.intersection(side(-1.0))


# ------------------------------------------------------------------ the pipeline

def label(fields, crop, transform, shape):
    """Label every polygon, cutting the ones a straight line can genuinely improve."""
    import geopandas as gpd
    from rasterio import features as rfeatures

    is_crop = crop == CROP_CLASS
    wgs = fields.to_crs(4326)
    ids = rfeatures.rasterize(
        ((geom, i) for i, geom in enumerate(wgs.geometry, start=1)),
        out_shape=shape, transform=transform, fill=0, dtype="int32")

    source_fid = (fields["fid"].to_numpy() if "fid" in fields.columns
                  else np.arange(1, len(fields) + 1))
    rows, geometries = [], []
    counts = {"clean": 0, "cut": 0, "majority": 0, "too_small": 0}

    positions = {}
    inside = ids > 0
    flat_ids = ids[inside]
    rr, cc = np.nonzero(inside)
    xs_w, ys_w = transform * (cc + 0.5, rr + 0.5)
    crop_flat = is_crop[inside]
    order = np.argsort(flat_ids, kind="stable")
    sorted_ids = flat_ids[order]
    starts = np.searchsorted(sorted_ids, np.arange(1, len(fields) + 1), side="left")
    ends = np.searchsorted(sorted_ids, np.arange(1, len(fields) + 1), side="right")

    transformer = None
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(4326, UTM, always_xy=True)
    except Exception:
        pass

    for index in range(len(fields)):
        lo, hi = starts[index], ends[index]
        members = order[lo:hi]
        geom = fields.geometry.iloc[index]
        pixels = members.size
        if pixels < MIN_PIXELS:
            counts["too_small"] += 1
            rows.append({"source_fid": int(source_fid[index]), "crop_fraction": np.nan,
                         "origin": "delineation", "decision": "too small to judge",
                         "pixels": pixels})
            geometries.append(geom)
            continue

        truth = crop_flat[members]
        fraction = float(truth.mean())

        if fraction >= CROP_HIGH or fraction <= CROP_LOW:
            counts["clean"] += 1
            rows.append({"source_fid": int(source_fid[index]), "crop_fraction": fraction,
                         "origin": "delineation", "decision": "clean", "pixels": pixels})
            geometries.append(geom)
            continue

        xs = np.asarray(xs_w)[members]
        ys = np.asarray(ys_w)[members]
        if transformer is not None:
            xs, ys = transformer.transform(xs, ys)
        theta = _orientation(geom)
        angle, offset, purity = _best_cut(xs, ys, truth, (theta, theta + np.pi / 2))
        baseline = max(fraction, 1 - fraction)

        if purity - baseline < MIN_CUT_GAIN:
            counts["majority"] += 1
            rows.append({"source_fid": int(source_fid[index]), "crop_fraction": fraction,
                         "origin": "delineation", "decision": "mixed, majority label",
                         "pixels": pixels})
            geometries.append(geom)
            continue

        first, second = _halves(geom, angle, offset)
        made = False
        for part in (first, second):
            if part.is_empty or part.area < 100.0:
                continue
            sel = (xs * np.cos(angle) + ys * np.sin(angle) > offset)
            side = truth[sel] if part is first else truth[~sel]
            if side.size < MIN_PIXELS:
                continue
            for piece in (part.geoms if part.geom_type == "MultiPolygon" else [part]):
                if piece.area < 100.0:
                    continue
                rows.append({"source_fid": int(source_fid[index]),
                             "crop_fraction": float(side.mean()), "origin": "split",
                             "decision": f"cut, purity {min(int(purity * 10) / 10, 0.9):.1f}+",
                             "pixels": int(side.size), "cut_purity": round(float(purity), 3)})
                geometries.append(piece)
                made = True
        if made:
            counts["cut"] += 1
        else:
            counts["majority"] += 1
            rows.append({"source_fid": int(source_fid[index]), "crop_fraction": fraction,
                         "origin": "delineation", "decision": "mixed, cut failed",
                         "pixels": pixels})
            geometries.append(geom)

    log.info("labelling: %s", counts)
    out = gpd.GeoDataFrame(rows, geometry=geometries, crs=fields.crs)
    return out, ids, is_crop


def orphan_polygons(is_crop, ids, transform, crs):
    """Polygons for crop the delineation never covered, regularised so they read as fields."""
    import geopandas as gpd
    from rasterio import features as rfeatures
    from shapely.geometry import shape as to_shape

    orphan = is_crop & (ids == 0)
    if not orphan.any():
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)

    records = []
    for geom, _ in rfeatures.shapes(orphan.astype(np.uint8), mask=orphan, transform=transform):
        records.append(to_shape(geom))
    frame = gpd.GeoDataFrame(geometry=records, crs=4326).to_crs(UTM)
    frame["acres"] = frame.area / SQM_PER_ACRE
    frame = frame[frame.acres >= MIN_ORPHAN_ACRES].reset_index(drop=True)

    # A rasterised edge is a staircase. Simplifying at half a pixel takes the steps
    # off without moving the boundary anywhere a client would notice.
    frame["geometry"] = frame.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    frame = frame[~frame.geometry.is_empty & frame.geometry.is_valid]
    log.info("orphan crop: %d polygons at or above %.2f acres, %.0f acres",
             len(frame), MIN_ORPHAN_ACRES, frame.acres.sum())
    frame["source_fid"] = -1        # no traced polygon stood here
    frame["crop_fraction"] = 1.0
    frame["origin"] = "derived from crop map"
    frame["decision"] = "no delineation polygon here"
    return frame.drop(columns=["acres"])



def tidy(frame):
    """Final pass: valid geometry, no overlaps, and traced edges beating derived ones.

    The overlap arithmetic in `repair` works on the input layer, but splitting and the
    derived orphan polygons happen afterwards and reintroduce slivers: 12,934 pairs
    sharing 40.8 acres between them, which is precision debris rather than real double
    counting, plus about 1,250 polygons that come back invalid once reprojected.

    Resolution order is the point. Traced geometry is placed first, smallest before
    largest as in `repair`, and derived polygons are placed last, so a boundary the
    crop map invented can never cut one that was drawn on the basemap. That is the
    same rule as everywhere else here: the delineation owns the geometry.
    """
    import geopandas as gpd
    from shapely.validation import make_valid

    frame = frame.copy()
    bad = ~frame.geometry.is_valid
    if bad.any():
        frame.loc[bad, "geometry"] = frame.loc[bad, "geometry"].apply(make_valid)
    for _ in range(5):
        frame = frame.explode(index_parts=False)
        if not frame.geom_type.isin(["MultiPolygon", "GeometryCollection"]).any():
            break
    frame = frame[frame.geom_type == "Polygon"]
    frame = frame[~frame.geometry.is_empty & frame.geometry.is_valid].reset_index(drop=True)

    derived = (frame.origin == "derived from crop map").to_numpy()
    areas = frame.area.to_numpy()
    order = np.lexsort((areas, derived))          # traced first, then by size

    index = frame.sindex
    geometries = list(frame.geometry)
    placed: set = set()
    for position in order:
        position = int(position)
        geom = geometries[position]
        if geom.is_empty:
            continue
        for other in index.query(geom, predicate="intersects"):
            other = int(other)
            if other == position or other not in placed or geometries[other].is_empty:
                continue
            geom = geom.difference(geometries[other])
        geometries[position] = geom
        placed.add(position)

    frame["geometry"] = geometries
    for _ in range(5):
        frame = frame.explode(index_parts=False)
        if not frame.geom_type.isin(["MultiPolygon", "GeometryCollection"]).any():
            break
    frame = frame[(frame.geom_type == "Polygon") & ~frame.geometry.is_empty]
    frame = frame[frame.geometry.is_valid]

    # A derived polygon that has been whittled below the orphan floor was never a
    # field, only the ragged edge of one the delineation already holds.
    small = (frame.origin == "derived from crop map") & (frame.area < MIN_ORPHAN_ACRES * SQM_PER_ACRE)
    frame = frame[~small & (frame.area > SLIVER_SQM)].reset_index(drop=True)
    log.info("tidy: %d polygons, %.0f acres, %d invalid",
             len(frame), frame.area.sum() / SQM_PER_ACRE,
             int((~frame.geometry.is_valid).sum()))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="crop fraction at which a polygon counts as crop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import geopandas as gpd
    import rasterio

    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(CROP_MAP) as src:
        crop = src.read(1)
        transform, shape, bounds = src.transform, (src.height, src.width), src.bounds
    log.info("crop map: %.0f acres of class %d",
             (crop == CROP_CLASS).sum() * PIXEL_SQM / SQM_PER_ACRE, CROP_CLASS)

    fields = gpd.read_file(DELINEATION, bbox=tuple(bounds), engine="pyogrio")
    fields = repair(fields)

    labelled, ids, is_crop = label(fields, crop, transform, shape)
    orphans = orphan_polygons(is_crop, ids, transform, labelled.crs)

    everything = gpd.GeoDataFrame(
        pd.concat([labelled, orphans], ignore_index=True), crs=labelled.crs)
    everything = tidy(everything)
    everything["acres"] = everything.area / SQM_PER_ACRE
    everything["is_crop"] = everything.crop_fraction >= args.threshold

    everything.to_crs(4326).to_file(OUT / "fields_labelled.gpkg", driver="GPKG")
    crop_only = everything[everything.is_crop].copy()
    crop_only.to_crs(4326).to_file(OUT / "fields_cane.gpkg", driver="GPKG")

    print("\n" + "=" * 74)
    print(f"{'origin':26s} {'polygons':>10s} {'acres':>12s} {'of which crop':>14s}")
    for origin, group in everything.groupby("origin"):
        print(f"{origin:26s} {len(group):>10,} {group.acres.sum():>12,.0f} "
              f"{group.loc[group.is_crop, 'acres'].sum():>14,.0f}")
    print(f"{'TOTAL':26s} {len(everything):>10,} {everything.acres.sum():>12,.0f} "
          f"{crop_only.acres.sum():>14,.0f}")

    print("\ndecisions:")
    for decision, group in everything.groupby("decision"):
        print(f"  {decision:34s} {len(group):>8,}  {group.acres.sum():>9,.0f} acres")

    raster_acres = (crop == CROP_CLASS).sum() * PIXEL_SQM / SQM_PER_ACRE
    print(f"\ncrop in the raster       {raster_acres:>9,.0f} acres")
    print(f"crop in the polygons     {crop_only.acres.sum():>9,.0f} acres "
          f"({100 * crop_only.acres.sum() / raster_acres:.1f}%)")
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
