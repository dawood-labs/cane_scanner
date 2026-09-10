"""Run the orchard filter over Al-Moiz Unit 1 and report exactly what it takes out.

This AOI is the control. An unsupervised look at it found only 0.2% of the
time-series model's cane pixels shaped like a perennial, so a correct filter should
remove very little here. A large removal is a failure of the filter, not a discovery
about the crop, and the numbers below are written to be read that way.

Everything already on disk is reused: the time-series map, its sieve, and the
smoothed NDVI stack. Nothing is re-downloaded and nothing is overwritten.

    python3 run_orchard_filter_on_test_feature.py
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from static_training import orchard_filter as of  # noqa: E402

log = logging.getLogger("run_orchard_filter")

CROPSCAN = SCRIPTS_DIR.parent
TEST = CROPSCAN / "data" / "test_data" / "almoiz_unit_1_test_feature_1"
CANE = TEST / "cane_2026"
RF_SIEVED = CANE / "almoiz_unit_1_test_feature_1_rf_classification_map_strict_sieve_multiclass_p20.tif"
# The smoothed stack is a VRT over the per-tile chunks on anything larger than a test
# feature, because materialising it is 9 GB. Either is read the same way.
_SERIES_STEM = CANE / "almoiz_unit_1_test_feature_1_smoothed_mosaic"
SERIES = next((_SERIES_STEM.with_suffix(ext) for ext in (".vrt", ".tif")
               if _SERIES_STEM.with_suffix(ext).exists()), _SERIES_STEM.with_suffix(".tif"))
FIELDS_DIR = CROPSCAN / "data" / "Al_Moiz_Unit_1_field_delineation_COMPLETE"
MODEL = CROPSCAN / "model_files" / "orchard_detector.json"
OUT_DIR = CANE / "orchard_filtered"
#: Orchard blocks the 2025 scan says carry cane. Growers in the mango belt plant
#: between the tree rows, and that cane reads as woody to every phenology rule
#: here, so the filter is not allowed to act inside them.
PROTECT = (CROPSCAN / "data" / "orchard_exclusion_mask"
           / "orchard_blocks_with_cane.gpkg")

CROP_CLASS, BACKGROUND = 1, 4
SQM_PER_ACRE = 4046.8564224


def removed_fields_layer(score_path: Path, filtered_path: Path, original_path: Path,
                         out_gpkg: Path) -> int:
    """Write the removed fields as their own layer, so they can be checked by eye."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(original_path) as src:
        before = src.read(1)
        transform, crs = src.transform, src.crs
    with rasterio.open(filtered_path) as src:
        after = src.read(1)
    with rasterio.open(score_path) as src:
        score = src.read(1)

    dropped = (before == CROP_CLASS) & (after != CROP_CLASS)
    if not dropped.any():
        log.info("nothing was removed, so no layer written")
        return 0

    records = []
    for geom, _ in shapes(dropped.astype(np.uint8), mask=dropped, transform=transform):
        records.append(shape(geom))
    frame = gpd.GeoDataFrame(geometry=records, crs=crs)
    metric = frame.to_crs(32642)
    frame["area_acres"] = metric.area / SQM_PER_ACRE
    frame["orchard_score"] = [
        float(np.nanmedian(score[dropped])) if dropped.any() else np.nan
    ] * len(frame)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(out_gpkg, driver="GPKG")
    log.info("removed patches: %d covering %.1f acres -> %s",
             len(frame), frame.area_acres.sum(), out_gpkg.name)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the sidecar threshold, for sensitivity checks")
    parser.add_argument("--no-fields", action="store_true",
                        help="ignore the delineation layer and use connected blobs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if not MODEL.exists():
        raise SystemExit(f"{MODEL} not found; train the detector first")
    for path in (RF_SIEVED, SERIES):
        if not path.exists():
            raise SystemExit(f"{path} not found")

    field_layer = None
    if not args.no_fields:
        candidates = sorted(glob.glob(str(FIELDS_DIR / "*.shp")))
        field_layer = candidates[0] if candidates else None
        log.info("field layer: %s", Path(field_layer).name if field_layer else "none")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered = OUT_DIR / "rf_classification_map_orchards_removed.tif"
    scores = OUT_DIR / "orchard_score.tif"

    result = of.filter_crop_map(
        crop_map_path=RF_SIEVED,
        series_path=SERIES,
        model_path=MODEL,
        out_filtered_path=filtered,
        out_score_path=scores,
        field_polygons=field_layer,
        crop_class=CROP_CLASS,
        background_class=BACKGROUND,
        threshold=args.threshold,
        protect_polygons=PROTECT if PROTECT.exists() else None,
    )

    acres_before = result.pixels_before * 100 / SQM_PER_ACRE
    acres_after = result.pixels_after * 100 / SQM_PER_ACRE
    n_patches = removed_fields_layer(scores, filtered, RF_SIEVED,
                                     OUT_DIR / "removed_as_orchard.gpkg")

    table = pd.DataFrame([
        {"measure": "cane pixels", "before": result.pixels_before, "after": result.pixels_after},
        {"measure": "acres", "before": round(acres_before), "after": round(acres_after)},
    ])
    print("\n" + "=" * 70)
    print(table.to_string(index=False))
    print(f"\nfields scored     {result.fields_scored:,}")
    print(f"clearing the gate {result.fields_passing_gate:,}")
    print(f"fields removed    {result.fields_removed:,}")
    print(f"removed patches   {n_patches:,}")
    print(f"threshold         {result.threshold:.2f}")
    print(f"acres removed     {acres_before - acres_after:,.0f} "
          f"({100 * result.removed_fraction:.2f}% of the map)")
    print("\nThis AOI holds no orchards: zero of its 6,537 delineated fields look")
    print("perennial. The right answer here is to remove nothing, so read the gate")
    print("count above, not the acreage: it is what stops a relative score from")
    print("taking a slice out of a map that has nothing to take.")
    print(f"\noutputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
