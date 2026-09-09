"""Test the orchard filter where orchards actually are, not where they are absent.

Al-Moiz Unit 1 answers only half the question. It holds no orchards, so all it can
show is that the filter does no harm; a filter that removes nothing anywhere would
pass it too.

This asks the other half. Over a district chip where the 2025 national cane map is
known to have swallowed orchards, it treats that map as the crop map, runs the
filter, and counts both sides:

  caught   orchard blocks the cane map calls cane, that the filter removes
  lost     cane the filter removes that no orchard block overlaps

No delineation layer exists for these districts, so the filter groups adjacent
gate-passing pixels into orchard-sized patches. Passing the cane map's own polygons
as fields was tried first and is wrong: a mapped polygon runs to thousands of
hectares against an orchard's 3.7, so the orchard never moves its median.

    python3 test_orchard_filter_where_it_matters.py
    python3 test_orchard_filter_where_it_matters.py --district "Mirpur Khas"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import build_orchard_training_set as B  # noqa: E402
from static_training import orchard_filter as of  # noqa: E402

log = logging.getLogger("test_where_it_matters")

CROPSCAN = SCRIPTS_DIR.parent
MODEL = CROPSCAN / "model_files" / "orchard_detector.json"
OUT = B.OUT / "field_test"

CROP_CLASS, BACKGROUND = 1, 4
SQM_PER_HA = 10_000.0


def build_crop_map(district: str, stack_path: Path, out_path: Path):
    """Rasterise the national cane map onto the chip grid, and keep the polygons."""
    import geopandas as gpd
    import rasterio
    from rasterio import features as rfeatures

    with rasterio.open(stack_path) as src:
        transform, shape, crs, bounds = src.transform, (src.height, src.width), src.crs, src.bounds

    cane = gpd.read_file(B.CANE_NATIONAL, bbox=tuple(bounds), engine="pyogrio")
    log.info("%s: %d cane polygons over the chip", district, len(cane))
    if cane.empty:
        raise SystemExit(f"no cane mapped over the {district} chip")

    burned = rfeatures.rasterize(
        ((geom, CROP_CLASS) for geom in cane.geometry),
        out_shape=shape, transform=transform, fill=BACKGROUND, dtype="uint8",
    )
    profile = {
        "driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
        "dtype": "uint8", "crs": crs, "transform": transform, "nodata": 255,
        "compress": "lzw", "tiled": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(burned, 1)
        dst.set_band_description(1, "national_cane_map_2025")
    return cane, out_path


def score_outcome(district: str, cane: "pd.DataFrame", crop_map: Path,
                  filtered: Path) -> Dict[str, float]:
    """How much of what was removed was orchard, and how much was not."""
    import geopandas as gpd
    import rasterio
    from rasterio import features as rfeatures

    confused = gpd.read_file(B.CONFUSED_GPKG, engine="pyogrio")
    confused = confused[confused.district == district]

    with rasterio.open(crop_map) as src:
        before = src.read(1)
        transform, shape = src.transform, (src.height, src.width)
    with rasterio.open(filtered) as src:
        after = src.read(1)

    removed = (before == CROP_CLASS) & (after != CROP_CLASS)
    kept = (before == CROP_CLASS) & (after == CROP_CLASS)

    orchard_mask = np.zeros(shape, dtype=bool)
    if not confused.empty:
        orchard_mask = rfeatures.geometry_mask(
            confused.geometry, out_shape=shape, transform=transform,
            invert=True, all_touched=True)

    cane_px = int((before == CROP_CLASS).sum())
    orchard_in_map = int((orchard_mask & (before == CROP_CLASS)).sum())
    clean_cane = int((~orchard_mask & (before == CROP_CLASS)).sum())

    caught = int((removed & orchard_mask).sum())
    lost = int((removed & ~orchard_mask).sum())

    return {
        "district": district,
        "mapped_cane_ha": round(cane_px * 100 / SQM_PER_HA, 1),
        "of_which_orchard_ha": round(orchard_in_map * 100 / SQM_PER_HA, 1),
        "orchard_caught_ha": round(caught * 100 / SQM_PER_HA, 1),
        "orchard_caught_pct": round(100 * caught / orchard_in_map, 1) if orchard_in_map else 0.0,
        "clean_cane_lost_ha": round(lost * 100 / SQM_PER_HA, 1),
        "clean_cane_lost_pct": round(100 * lost / clean_cane, 2) if clean_cane else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district", nargs="*", default=None,
                        help="districts to test; default is every chip with a stack")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("static_training.orchard_filter").setLevel(logging.INFO)

    if not MODEL.exists():
        raise SystemExit(f"{MODEL} not found; train the detector first")

    stacks = sorted(B.STACK_DIR.glob("*_smoothed.tif"))
    wanted = set(args.district) if args.district else None
    rows = []

    for stack in stacks:
        district = stack.stem.replace("_smoothed", "").replace("_", " ")
        if wanted and district not in wanted:
            continue
        work = OUT / district.replace(" ", "_")
        work.mkdir(parents=True, exist_ok=True)
        try:
            cane, crop_map = build_crop_map(district, stack, work / "cane_map_2025.tif")
            filtered = work / "cane_map_2025_orchards_removed.tif"
            result = of.filter_crop_map(
                crop_map_path=crop_map, series_path=stack, model_path=MODEL,
                out_filtered_path=filtered, out_score_path=work / "orchard_score.tif",
                field_polygons=None, crop_class=CROP_CLASS,
                background_class=BACKGROUND, threshold=args.threshold,
            )
            row = score_outcome(district, cane, crop_map, filtered)
            row["fields_scored"] = result.fields_scored
            row["cleared_gate"] = result.fields_passing_gate
            row["fields_removed"] = result.fields_removed
            rows.append(row)
            log.info("%s: caught %.1f%% of the orchard, lost %.2f%% of the clean cane",
                     district, row["orchard_caught_pct"], row["clean_cane_lost_pct"])
        except Exception as exc:
            log.error("%s failed: %s", district, exc)

    if not rows:
        raise SystemExit("nothing tested")
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "field_test.csv", index=False)
    print("\n" + "=" * 96)
    print(table.to_string(index=False))
    print("\nRead the two rightmost measures together. Catching orchards is only worth")
    print("anything if the clean cane beside them survives, and losing none of it is")
    print("worth nothing if no orchard is caught either.")
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
