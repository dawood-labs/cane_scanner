"""Find out what actually stands between a 10 m crop map and clean field polygons.

The delineation polygons were drawn on high-resolution basemap imagery and are the
geometry we want to deliver. The crop map is 10 m Sentinel and is only fit to say
what is growing, not where the edges are. Labelling one with the other sounds simple
and is not, so this counts the awkward cases before anything is designed around them.

Three were known going in: a polygon that is cleanly one class, a polygon holding
several real fields of different classes, and crop with no polygon over it. This
looks for those and for everything else it can find, because the ones nobody listed
are the ones that break a pipeline in production.

    python3 audit_field_labelling.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

log = logging.getLogger("audit_field_labelling")

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

#: Above this a polygon is called crop outright, below it non-crop; between the two
#: it holds more than one thing and has to be dealt with.
CLEAN_HIGH, CLEAN_LOW = 0.85, 0.15

#: Polygons under this carry too few pixels for the fraction to mean anything.
MIN_PIXELS = 10


def load() -> tuple:
    import geopandas as gpd
    import rasterio

    with rasterio.open(CROP_MAP) as src:
        crop = src.read(1)
        transform, shape, bounds, crs = src.transform, (src.height, src.width), src.bounds, src.crs

    fields = gpd.read_file(DELINEATION, bbox=tuple(bounds), engine="pyogrio").reset_index(drop=True)
    log.info("crop map %s, %d crop pixels (%.0f acres)", shape,
             int((crop == CROP_CLASS).sum()),
             (crop == CROP_CLASS).sum() * PIXEL_SQM / SQM_PER_ACRE)
    log.info("delineation polygons over the extent: %d, columns %s",
             len(fields), list(fields.columns))
    return crop, transform, shape, crs, fields


def geometry_health(fields, crs) -> Dict[str, int]:
    """Problems in the polygon layer itself, before any labelling is attempted."""
    import geopandas as gpd

    metric = fields.to_crs(UTM)
    acres = metric.area / SQM_PER_ACRE
    report = {
        "polygons": len(fields),
        "invalid_geometry": int((~fields.geometry.is_valid).sum()),
        "empty_geometry": int(fields.geometry.is_empty.sum()),
        "multipart": int((fields.geometry.geom_type == "MultiPolygon").sum()),
        "with_holes": int(sum(1 for g in fields.geometry
                              if g.geom_type == "Polygon" and len(g.interiors) > 0)),
        "under_quarter_acre": int((acres < 0.25).sum()),
        "over_50_acres": int((acres > 50).sum()),
    }

    # Overlaps: a pixel claimed by two polygons has no single answer.
    joined = gpd.sjoin(fields[["geometry"]], fields[["geometry"]],
                       how="inner", predicate="overlaps")
    report["overlapping_pairs"] = int(len(joined) // 2)
    report["acres_total"] = round(float(acres.sum()), 1)
    report["acres_median"] = round(float(acres.median()), 2)
    return report


def per_polygon(crop, transform, shape, fields) -> pd.DataFrame:
    from rasterio import features as rfeatures
    from scipy import ndimage

    ids = rfeatures.rasterize(
        ((geom, i) for i, geom in enumerate(fields.geometry, start=1)),
        out_shape=shape, transform=transform, fill=0, dtype="int32")
    is_crop = crop == CROP_CLASS
    inside = ids > 0

    frame = pd.DataFrame({"pid": ids[inside], "crop": is_crop[inside]})
    grouped = frame.groupby("pid").agg(pixels=("crop", "size"), crop_pixels=("crop", "sum"))
    grouped["fraction"] = grouped.crop_pixels / grouped.pixels

    # How many separate pieces of crop sit inside each polygon. One piece splits with
    # a single cut; several do not, and that is a different problem.
    pieces = {}
    labelled, _ = ndimage.label(is_crop, structure=np.ones((3, 3)))
    both = (ids > 0) & is_crop
    if both.any():
        pairs = pd.DataFrame({"pid": ids[both], "blob": labelled[both]}).drop_duplicates()
        pieces = pairs.groupby("pid").size().to_dict()
    grouped["crop_pieces"] = grouped.index.map(lambda p: pieces.get(p, 0))

    grouped = grouped[grouped.pixels >= MIN_PIXELS].copy()
    grouped["acres"] = grouped.pixels * PIXEL_SQM / SQM_PER_ACRE
    return grouped, ids, is_crop


def crop_without_polygon(is_crop, ids) -> Dict[str, float]:
    """Crop the delineation never covered, and whether it is worth a polygon."""
    from scipy import ndimage

    orphan = is_crop & (ids == 0)
    labelled, count = ndimage.label(orphan, structure=np.ones((3, 3)))
    if count == 0:
        return {"orphan_acres": 0.0, "orphan_blobs": 0}
    sizes = np.bincount(labelled.reshape(-1))[1:]
    acres = sizes * PIXEL_SQM / SQM_PER_ACRE
    return {
        "orphan_acres": round(float(acres.sum()), 1),
        "orphan_share_of_crop": round(100 * float(orphan.sum()) / float(is_crop.sum()), 1),
        "orphan_blobs": int(count),
        "blobs_over_half_acre": int((acres >= 0.5).sum()),
        "acres_in_blobs_over_half": round(float(acres[acres >= 0.5].sum()), 1),
        "largest_blob_acres": round(float(acres.max()), 1),
    }


def splittability(grouped, ids, is_crop, fields, transform) -> pd.DataFrame:
    """For the mixed polygons, can a single straight cut separate the two classes?

    Field subdivisions in this landscape run parallel to the field's own edges, so the
    test is: project the pixels onto the polygon's long axis and onto its short axis,
    and see whether either ordering separates crop from non-crop. If one does, a
    straight cut works and the output keeps clean edges. If neither does, forcing a
    cut would invent a boundary that is not there.
    """
    from shapely.geometry import box

    mixed = grouped[(grouped.fraction > CLEAN_LOW) & (grouped.fraction < CLEAN_HIGH)]
    rows: List[Dict] = []
    rows_checked = 0

    for pid in mixed.index:
        geom = fields.geometry.iloc[pid - 1]
        rect = geom.minimum_rotated_rectangle
        if rect.is_empty or rect.geom_type != "Polygon":
            continue
        coords = np.array(rect.exterior.coords[:4])
        edges = np.diff(np.vstack([coords, coords[:1]]), axis=0)
        lengths = np.hypot(edges[:, 0], edges[:, 1])
        long_edge = edges[int(np.argmax(lengths))]
        angle = np.arctan2(long_edge[1], long_edge[0])

        sel = ids == pid
        rr, cc = np.nonzero(sel)
        if rr.size < MIN_PIXELS:
            continue
        xs, ys = transform * (cc + 0.5, rr + 0.5)
        crop_here = is_crop[rr, cc]
        if crop_here.all() or not crop_here.any():
            continue
        rows_checked += 1

        best = 0.0
        for theta in (angle, angle + np.pi / 2):
            projection = np.asarray(xs) * np.cos(theta) + np.asarray(ys) * np.sin(theta)
            order = np.argsort(projection)
            truth = crop_here[order]
            # Best single cut along this axis: how purely does it divide the two.
            n = truth.size
            crop_before = np.cumsum(truth)
            crop_total = crop_before[-1]
            index = np.arange(1, n + 1)
            left_pure = crop_before / index
            right_pure = (crop_total - crop_before) / np.maximum(n - index, 1)
            purity = (index * np.maximum(left_pure, 1 - left_pure)
                      + (n - index) * np.maximum(right_pure, 1 - right_pure)) / n
            best = max(best, float(np.nanmax(purity)))
        rows.append({"pid": pid, "fraction": grouped.fraction[pid],
                     "acres": grouped.acres[pid], "best_straight_cut_purity": best,
                     "crop_pieces": grouped.crop_pieces[pid]})

    log.info("mixed polygons examined for a straight cut: %d", rows_checked)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)

    crop, transform, shape, crs, fields = load()

    print("\n" + "=" * 74)
    print("THE POLYGON LAYER ITSELF")
    health = geometry_health(fields, crs)
    for key, value in health.items():
        print(f"  {key:24s} {value:>12,}")

    grouped, ids, is_crop = per_polygon(crop, transform, shape, fields)
    print("\n" + "=" * 74)
    print(f"LABELLING, {len(grouped):,} polygons with at least {MIN_PIXELS} pixels")
    clean_crop = grouped.fraction >= CLEAN_HIGH
    clean_none = grouped.fraction <= CLEAN_LOW
    mixed = ~clean_crop & ~clean_none
    for name, sel in [("clean crop", clean_crop), ("clean non-crop", clean_none),
                      ("MIXED", mixed)]:
        print(f"  {name:16s} {int(sel.sum()):>7,}  {100*sel.mean():5.1f}%   "
              f"{grouped.acres[sel].sum():>9,.0f} acres")

    print("\n  crop fraction, distribution:")
    counts, edges = np.histogram(grouped.fraction, bins=10, range=(0, 1))
    for i in range(10):
        bar = "#" * int(52 * counts[i] / max(counts.max(), 1))
        print(f"   {edges[i]:.1f}-{edges[i+1]:.1f} {counts[i]:>7,} {bar}")

    print("\n  separate pieces of crop inside one polygon:")
    print("   ", grouped.crop_pieces.value_counts().head(6).to_dict())

    print("\n" + "=" * 74)
    print("CROP WITH NO POLYGON OVER IT")
    for key, value in crop_without_polygon(is_crop, ids).items():
        print(f"  {key:26s} {value:>12,}")

    cuts = splittability(grouped, ids, is_crop, fields, transform)
    if not cuts.empty:
        print("\n" + "=" * 74)
        print(f"CAN THE MIXED ONES BE CUT CLEANLY, {len(cuts):,} examined")
        for level in (0.95, 0.90, 0.85, 0.80):
            share = (cuts.best_straight_cut_purity >= level).mean()
            print(f"  a single straight cut reaches {level:.0%} purity: "
                  f"{100*share:5.1f}%  ({int((cuts.best_straight_cut_purity>=level).sum()):,})")
        print(f"\n  median purity of the best straight cut: "
              f"{cuts.best_straight_cut_purity.median():.3f}")
        cuts.to_csv(OUT / "mixed_polygon_cuts.csv", index=False)

    grouped.to_csv(OUT / "per_polygon.csv")
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
