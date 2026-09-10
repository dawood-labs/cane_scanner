"""Build a mask of orchard ground where sugarcane never grows.

The manual tree mask is 59 district polygons with the orchard blocks punched out as
interior rings. Those 67,004 holes are delineated orchards and are the only orchard
labels anybody has. It is tempting to subtract all of them from the crop map and be
done, and in Multan that would destroy real cane: growers there plant cane between
mango rows, so a mango block and a cane field are not exclusive categories. A blanket
orchard mask deletes the intercropped ones.

So the mask is not "orchards". It is orchard ground that has never been seen carrying
cane, measured against the 2025 national scan, which is the only independent record of
where cane actually grows. An orchard block that overlaps 2025 cane is left alone and
handed back to the model. This also makes the mask self-limiting: it can only remove
ground the production map itself has never called cane.

Two more things keep it safe. Rings are kept only between 0.5 and 100 ha, because the
tail runs to 14,667 ha and those are riverine forest belts, not orchards. And every
kept block is shrunk by one Sentinel pixel, so a mask edge can never reach across into
the field next door.

    python3 build_orchard_exclusion_mask.py --stage rings
    python3 build_orchard_exclusion_mask.py --stage screen
    python3 build_orchard_exclusion_mask.py --stage verify
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

log = logging.getLogger("build_orchard_exclusion_mask")

CROPSCAN = SCRIPTS_DIR.parent
FAO = CROPSCAN.parent / "FAO" / "cane"
TREE_MASK = (FAO / "new_tree_masked_dist_data" / "Tree_mask_wgs84_final_FAO_simplify"
             / "Tree_mask_wgs84_final_FAO_simplify.shp")
NATIONAL_CANE = (FAO / "Sugarcane_3m-10m_Pakistan-Scan_2025"
                 / "Sugarcane_3m-10m_Pakistan-Scan_2025.shp")
OUT = CROPSCAN / "data" / "orchard_exclusion_mask"

#: Rings outside this range are not orchards. Below, they are digitising noise; above,
#: they are the forest and riverine belts that dominate the area tail.
MIN_HA, MAX_HA = 0.5, 100.0

#: An orchard block that overlaps this much 2025 cane is treated as possibly
#: intercropped and is NOT masked. One per cent is about a single Sentinel pixel on a
#: half-hectare block, so it forgives registration slop without forgiving real cane.
CANE_TOLERANCE = 0.01

#: Every kept block is shrunk by one pixel so the mask edge stays inside the orchard.
SHRINK_M = 10.0

UTM = 32642
HA_PER_SQM = 1e-4
SQM_PER_ACRE = 4046.8564224


def _metric(frame):
    """A metre CRS suitable for the whole country, chosen once so areas are comparable."""
    return frame.to_crs(UTM)


def stage_rings() -> None:
    """Pull the orchard blocks out of the tree mask's interior rings."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    districts = gpd.read_file(TREE_MASK, engine="pyogrio")
    log.info("tree mask: %d district polygons", len(districts))

    records: List[Dict] = []
    for _, row in districts.iterrows():
        geoms = (row.geometry.geoms if row.geometry.geom_type == "MultiPolygon"
                 else [row.geometry])
        for part in geoms:
            for ring in part.interiors:
                records.append({"district": row.DISTRICT, "province": row.Province,
                                "geometry": Polygon(ring)})

    rings = gpd.GeoDataFrame(records, crs=districts.crs)
    log.info("interior rings: %d", len(rings))

    metric = _metric(rings)
    metric["ha"] = metric.area * HA_PER_SQM
    keep = (metric.ha >= MIN_HA) & (metric.ha <= MAX_HA)
    log.info("rings inside %.1f-%.0f ha: %d of %d (%.0f ha of %.0f)",
             MIN_HA, MAX_HA, int(keep.sum()), len(metric),
             metric.ha[keep].sum(), metric.ha.sum())

    metric = metric[keep].reset_index(drop=True)
    # A district with no rings was never digitised. Saying nothing about it is right;
    # saying "no orchards here" would be a lie the pipeline would act on.
    per_district = metric.groupby("district").ha.agg(["count", "sum"])
    log.info("districts with rings: %d of %d", len(per_district), districts.DISTRICT.nunique())

    OUT.mkdir(parents=True, exist_ok=True)
    metric.to_crs(4326).to_file(OUT / "orchard_rings.gpkg", driver="GPKG")
    per_district.to_csv(OUT / "rings_per_district.csv")
    print(per_district.sort_values("sum", ascending=False).head(20).to_string())


def stage_screen(districts: List[str] | None = None) -> None:
    """Drop every orchard block that the 2025 national scan says carries cane."""
    import geopandas as gpd

    rings = gpd.read_file(OUT / "orchard_rings.gpkg", engine="pyogrio")
    if districts:
        rings = rings[rings.district.isin(districts)]
    log.info("screening %d orchard blocks in %d districts",
             len(rings), rings.district.nunique())

    kept, dropped = [], []
    summary: List[Dict] = []
    for district, group in rings.groupby("district"):
        bounds = tuple(group.total_bounds)
        try:
            cane = gpd.read_file(NATIONAL_CANE, bbox=bounds, engine="pyogrio")
        except Exception as error:            # a district outside the scan's extent
            log.warning("%s: cane read failed (%s); keeping nothing from it", district, error)
            continue
        if cane.empty:
            # No cane mapped anywhere near: nothing to protect, but also no evidence.
            # Keep the blocks; the mask still only removes what the model calls cane.
            log.info("%-18s no 2025 cane in the bbox, %d blocks kept as-is",
                     district, len(group))
            kept.append(group)
            summary.append({"district": district, "blocks": len(group),
                            "kept": len(group), "dropped": 0, "cane_acres_nearby": 0.0})
            continue

        cane_m = _metric(cane[["geometry"]])
        cane_m["geometry"] = cane_m.geometry.buffer(0)
        group_m = _metric(group)
        group_m["block_area"] = group_m.area

        overlay = gpd.overlay(group_m.reset_index()[["index", "geometry", "block_area"]],
                             cane_m, how="intersection", keep_geom_type=True)
        if overlay.empty:
            share = pd.Series(0.0, index=group_m.index)
        else:
            overlay["hit"] = overlay.area
            share = (overlay.groupby("index").hit.sum() / group_m.block_area)
            share = share.reindex(group_m.index).fillna(0.0)

        group_m["cane_share"] = share
        clean = group_m.cane_share <= CANE_TOLERANCE
        kept.append(group_m[clean].to_crs(4326))
        dropped.append(group_m[~clean].to_crs(4326))
        log.info("%-18s %5d blocks, %5d kept, %5d dropped for cane overlap (%.1f%%)",
                 district, len(group_m), int(clean.sum()), int((~clean).sum()),
                 100 * (~clean).mean())
        summary.append({"district": district, "blocks": len(group_m),
                        "kept": int(clean.sum()), "dropped": int((~clean).sum()),
                        "dropped_ha": float(group_m.ha[~clean].sum()),
                        "median_cane_share_dropped": float(group_m.cane_share[~clean].median())
                        if (~clean).any() else 0.0})

    mask = gpd.GeoDataFrame(pd.concat(kept, ignore_index=True), crs=4326)
    metric = _metric(mask)
    metric["geometry"] = metric.geometry.buffer(-SHRINK_M)
    metric = metric[~metric.geometry.is_empty & metric.geometry.is_valid]

    # Then take the 2025 cane out of what is left. The block-level tolerance is a
    # judgement about whether a block is intercropped; this is not a judgement at all.
    # Whatever the scan called cane is cut out of the mask geometry, so no mapped cane
    # can sit inside the mask however the tolerance is set. It also means the tolerance
    # can stay loose enough to ignore a stray misregistered sliver without that
    # slackness ever costing real ground.
    carved = 0.0
    pieces = []
    for district, group in metric.groupby("district"):
        try:
            cane = gpd.read_file(NATIONAL_CANE, bbox=tuple(group.to_crs(4326).total_bounds),
                                 engine="pyogrio")
        except Exception:
            pieces.append(group)
            continue
        if cane.empty:
            pieces.append(group)
            continue
        cane_m = _metric(cane[["geometry"]])
        cane_m["geometry"] = cane_m.geometry.buffer(0)
        before_area = float(group.area.sum())
        # overlay rather than differencing against one dissolved union: dissolving a
        # district's worth of scan polygons costs far more memory than it saves.
        group = gpd.overlay(group.reset_index(drop=True), cane_m[["geometry"]],
                            how="difference", keep_geom_type=True)
        group = group[~group.geometry.is_empty & group.geometry.is_valid]
        carved += (before_area - float(group.area.sum())) * HA_PER_SQM
        pieces.append(group)
    metric = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=UTM)
    log.info("carving 2025 cane out of the mask removed a further %.0f ha", carved)

    metric = metric.explode(index_parts=False)
    metric = metric[metric.geom_type == "Polygon"]
    metric = metric[metric.area > 1000.0]        # a tenth of a hectare is not an orchard
    metric["ha"] = metric.area * HA_PER_SQM
    log.info("final mask: %d blocks, %.0f ha after shrinking by %.0f m",
             len(metric), metric.ha.sum(), SHRINK_M)

    metric.to_crs(4326).to_file(OUT / "orchard_exclusion_mask.gpkg", driver="GPKG")
    if dropped:
        gpd.GeoDataFrame(pd.concat(dropped, ignore_index=True), crs=4326).to_file(
            OUT / "orchard_blocks_with_cane.gpkg", driver="GPKG")
    frame = pd.DataFrame(summary)
    frame.to_csv(OUT / "screening_per_district.csv", index=False)

    print("\n" + "=" * 78)
    print("ORCHARD BLOCKS THAT CARRY CANE  (the intercropping the mask must not touch)")
    print(frame.sort_values("dropped", ascending=False).head(25).to_string(index=False))
    total = frame.blocks.sum()
    print(f"\n{frame.dropped.sum():,} of {total:,} blocks ({100*frame.dropped.sum()/max(total,1):.1f}%) "
          f"overlap 2025 cane and stay in the model's hands")


def stage_verify() -> None:
    """Two questions: how much real cane does the mask touch, and what does it remove."""
    import geopandas as gpd
    import rasterio
    from rasterio import features as rfeatures

    mask = gpd.read_file(OUT / "orchard_exclusion_mask.gpkg", engine="pyogrio")
    print(f"mask: {len(mask):,} blocks")

    # 1. Against the independent record. By construction this should be near zero; if
    #    it is not, the screen or the shrink is wrong and the mask would eat cane.
    rows = []
    for district, group in mask.groupby("district"):
        cane = gpd.read_file(NATIONAL_CANE, bbox=tuple(group.total_bounds), engine="pyogrio")
        if cane.empty:
            continue
        cane_m = _metric(cane[["geometry"]])
        cane_m["geometry"] = cane_m.geometry.buffer(0)
        hit = gpd.overlay(_metric(group)[["geometry"]], cane_m,
                          how="intersection", keep_geom_type=True)
        rows.append({"district": district,
                     "mask_ha": float(_metric(group).area.sum() * HA_PER_SQM),
                     "cane_ha_nearby": float(cane_m.area.sum() * HA_PER_SQM),
                     "cane_ha_inside_mask": float(hit.area.sum() * HA_PER_SQM) if not hit.empty else 0.0})
    leak = pd.DataFrame(rows)
    if not leak.empty:
        leak["leak_pct"] = 100 * leak.cane_ha_inside_mask / leak.cane_ha_nearby.replace(0, np.nan)
        print("\n" + "=" * 78)
        print("2025 CANE FALLING INSIDE THE FINAL MASK  (has to be near zero)")
        print(leak.sort_values("cane_ha_inside_mask", ascending=False).head(20).to_string(index=False))
        print(f"\ntotal: {leak.cane_ha_inside_mask.sum():,.0f} ha of "
              f"{leak.cane_ha_nearby.sum():,.0f} ha "
              f"({100*leak.cane_ha_inside_mask.sum()/max(leak.cane_ha_nearby.sum(),1):.3f}%)")
        leak.to_csv(OUT / "cane_leak_into_mask.csv", index=False)

    # 2. Against the Al-Moiz test AOI, which has no orchards in it. A correct mask
    #    removes nothing there. Anything it does remove is a false positive we can see.
    crop_map = (CROPSCAN / "data" / "test_data" / "almoiz_unit_1_test_feature_1" / "cane_2026"
                / "30_Aug_2026" / "v4" / "static_mosaic_30_Aug_2026_Cls_v4_sieved_p20.tif")
    if crop_map.exists():
        with rasterio.open(crop_map) as src:
            crop = src.read(1)
            transform, shape, bounds = src.transform, (src.height, src.width), src.bounds
        here = mask.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
        acres = (crop == 1).sum() * 100.0 / SQM_PER_ACRE
        if here.empty:
            print(f"\nAl-Moiz test AOI: no mask blocks overlap it. "
                  f"{acres:,.0f} acres of cane untouched, as expected.")
        else:
            burn = rfeatures.rasterize(here.geometry, out_shape=shape, transform=transform,
                                       fill=0, default_value=1, dtype="uint8").astype(bool)
            removed = ((crop == 1) & burn).sum() * 100.0 / SQM_PER_ACRE
            print(f"\nAl-Moiz test AOI: mask removes {removed:,.1f} of {acres:,.0f} acres "
                  f"({100*removed/max(acres,1):.2f}%)")



def stage_circularity(districts: List[str] | None = None) -> None:
    """Check whether the screen can see anything at all, district by district.

    The screen assumes the 2025 national scan is independent evidence. It may not be.
    If that scan was itself produced with this tree mask applied, then by construction
    it holds no cane inside any ring, the screen finds nothing to protect, and the mask
    quietly swallows whatever cane is really there. Multan came back at 0.6% while
    Mirpur Khas came back at 14.2%, which is the wrong way round if growers intercrop
    in the mango belt, so this has to be settled rather than assumed.

    The test: move each ring 500 m in a random direction and screen again. A displaced
    ring sits on ordinary farmland, so its overlap with the scan is what the district's
    cane density alone would produce. If real rings overlap far less than displaced
    ones, the scan is avoiding the rings and the screen is blind there. If the two are
    similar, the rings genuinely sit on ground the scan sees no cane on, and the screen
    means what it says.
    """
    import geopandas as gpd
    from shapely.affinity import translate

    rings = gpd.read_file(OUT / "orchard_rings.gpkg", engine="pyogrio")
    if districts:
        rings = rings[rings.district.isin(districts)]

    rng = np.random.default_rng(0)
    rows: List[Dict] = []
    for district, group in rings.groupby("district"):
        group_m = _metric(group)
        angle = rng.uniform(0, 2 * np.pi, len(group_m))
        shifted = group_m.copy()
        shifted["geometry"] = [
            translate(g, 500 * np.cos(a), 500 * np.sin(a))
            for g, a in zip(group_m.geometry, angle)]

        bounds = tuple(gpd.GeoSeries(list(group_m.geometry) + list(shifted.geometry),
                                     crs=UTM).to_crs(4326).total_bounds)
        try:
            cane = gpd.read_file(NATIONAL_CANE, bbox=bounds, engine="pyogrio")
        except Exception:
            continue
        if cane.empty:
            continue
        cane_m = _metric(cane[["geometry"]])
        cane_m["geometry"] = cane_m.geometry.buffer(0)

        def covered(frame):
            frame = frame.reset_index(drop=True)
            frame["own"] = frame.area
            hit = gpd.overlay(frame[["geometry", "own"]].reset_index(),
                              cane_m, how="intersection", keep_geom_type=True)
            if hit.empty:
                return 0.0
            return float(hit.area.sum() / frame.own.sum())

        real, displaced = covered(group_m), covered(shifted)
        rows.append({"district": district, "blocks": len(group_m),
                     "cane_share_in_rings": round(100 * real, 2),
                     "cane_share_500m_away": round(100 * displaced, 2),
                     "ratio": round(real / displaced, 3) if displaced > 0 else np.nan})
        log.info("%-18s rings %.2f%%, displaced %.2f%%", district,
                 100 * real, 100 * displaced)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "circularity_check.csv", index=False)
    print("\n" + "=" * 78)
    print("IS THE 2025 SCAN INDEPENDENT OF THE TREE MASK?")
    print("  ratio near 1: the rings sit on genuinely cane-free ground, screen works")
    print("  ratio near 0: the scan avoids the rings, the screen is blind there")
    print(frame.sort_values("ratio").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["rings", "screen", "circularity", "verify"])
    parser.add_argument("--districts", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.stage == "rings":
        stage_rings()
    elif args.stage == "screen":
        stage_screen(args.districts)
    elif args.stage == "circularity":
        stage_circularity(args.districts)
    else:
        stage_verify()
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
