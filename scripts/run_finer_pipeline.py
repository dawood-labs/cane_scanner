"""Rebuild the cane map with a finer sieve and both August dates, and say what changed.

Three deliberate changes from the deployed run, all aimed at keeping cane the current
settings throw away:

**A 0.15-acre sieve instead of 0.5.** The time-series map loses 277 acres of cane to
the 20-pixel sieve before the static model ever sees it, and a 0.15-acre field is a
real field here: the delineation layer has a median field size of 0.77 acres and 1,110
polygons under a quarter acre.

**Both August dates, unioned.** 30 August alone maps 7,255 acres and 10 August maps
8,089 within the same mask, and the difference is mostly cane cut between the two: a
field harvested on the 20th is bare on the 30th and was standing on the 10th. Taking
either date as cane recovers those fields. It also takes on each date's false
positives, which is what the comparison below is for.

**The static model re-run against the finer mask**, because the deployed static output
only covers what the 20-pixel sieve let through: its nodata is exactly the sieved map's
background, so the extra ground has never been classified at all.

    python3 run_finer_pipeline.py --stage all
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

from static_training import inference as st_inference  # noqa: E402
from static_training import sieve as st_sieve          # noqa: E402

log = logging.getLogger("run_finer_pipeline")

CROPSCAN = SCRIPTS_DIR.parent
TEST = CROPSCAN / "data" / "test_data" / "almoiz_unit_1_test_feature_1"
CANE = TEST / "cane_2026"
OUT = CANE / "finer_p6"

RF_RAW = CANE / "almoiz_unit_1_test_feature_1_rf_classification_map.tif"
MODEL = CROPSCAN / "model_files" / "fao_cane_xgb_model_v4.json"
DELINEATION = (CROPSCAN / "data" / "Al_Moiz_Unit_1_field_delineation_COMPLETE"
               / "Al_Moiz_Unit_1_field_delineation_COMPLETE.shp")

#: The two dates and the tile each was downloaded to.
DATES = [("30_Aug_2026", "static_10m_tile_0001_30_Aug_2026.tif"),
         ("10_Aug_2026", "static_10m_tile_0001_10_Aug_2026.tif")]

CROP_CLASS, BACKGROUND, NODATA = 1, 4, 255

#: 0.15 acres at 10 m. The deployed run uses 20, which is 0.49 acres.
MIN_PIXELS = 6

SQM_PER_ACRE = 4046.8564224
PIXEL_SQM = 100.0


def _acres(mask) -> float:
    return float(mask.sum()) * PIXEL_SQM / SQM_PER_ACRE


def stage_sieve() -> Path:
    """Sieve the raw time-series map at the finer size."""
    import rasterio

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"rf_sieved_p{MIN_PIXELS}.tif"
    if target.exists():
        log.info("already sieved: %s", target.name)
        return target

    produced = Path(st_sieve.apply_strict_directional_sieve(
        str(RF_RAW), target_classes=[CROP_CLASS],
        min_pixel_size=MIN_PIXELS, connectivity=4, nodata_val=NODATA))
    produced.replace(target)

    with rasterio.open(RF_RAW) as src:
        raw = src.read(1)
    with rasterio.open(target) as src:
        fine = src.read(1)
    log.info("time-series cane: %.0f acres raw, %.0f after a %d-pixel sieve",
             _acres(raw == CROP_CLASS), _acres(fine == CROP_CLASS), MIN_PIXELS)
    return target


def stage_static(mask_path: Path) -> List[Path]:
    """Classify both dates inside the finer mask."""
    import rasterio

    with rasterio.open(mask_path) as src:
        mask = src.read(1) == CROP_CLASS
    log.info("static model may see %.0f acres", _acres(mask))

    produced = []
    for folder, tile in DATES:
        image = CANE / folder / tile
        if not image.exists():
            raise FileNotFoundError(f"{image} is missing; the tile was never downloaded")
        cls_path = OUT / f"static_{folder}_Cls_v4_p{MIN_PIXELS}mask.tif"
        prob_path = OUT / f"static_{folder}_prob_v4_p{MIN_PIXELS}mask.tif"
        if cls_path.exists():
            log.info("%s already classified", folder)
            produced.append(cls_path)
            continue
        result = st_inference.classify_raster(
            image, MODEL, cls_path, prob_path, mask_array=mask,
            positive_out=CROP_CLASS, background_out=BACKGROUND, enforce_domain=False)
        log.info("%s: threshold %.2f, %s", folder, result.threshold, result.verdict)
        produced.append(cls_path)

    for path in produced:
        with rasterio.open(path) as src:
            log.info("%-40s %.0f acres of cane", path.name, _acres(src.read(1) == CROP_CLASS))
    return produced


def stage_union(classified: List[Path]) -> Path:
    """Cane where either date says cane, then sieved at the finer size.

    A union is the right shape for this question and not obviously the right answer.
    Cane cut between the two dates is bare on the later one and standing on the earlier,
    so a union recovers those fields; but a false positive on either date also survives
    into the result. The comparison stage is what decides whether that trade is worth
    taking, and it is reported per date so the source of any gain is visible.
    """
    import rasterio

    stack, profile = [], None
    for path in classified:
        with rasterio.open(path) as src:
            stack.append(src.read(1))
            profile = src.profile.copy()

    cane = np.zeros(stack[0].shape, dtype=bool)
    seen = np.zeros(stack[0].shape, dtype=bool)
    for band in stack:
        cane |= band == CROP_CLASS
        seen |= band != NODATA

    out = np.full(stack[0].shape, NODATA, dtype=np.uint8)
    out[seen] = BACKGROUND
    out[cane] = CROP_CLASS

    union_path = OUT / "static_union_Cls_v4.tif"
    profile.update(compress="lzw", tiled=True, dtype="uint8", nodata=NODATA)
    with rasterio.open(union_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "cane if either August date says cane")

    for band, (folder, _) in zip(stack, DATES):
        log.info("%-14s %.0f acres", folder, _acres(band == CROP_CLASS))
    log.info("union         %.0f acres", _acres(cane))

    sieved = Path(st_sieve.apply_strict_directional_sieve(
        str(union_path), target_classes=[CROP_CLASS],
        min_pixel_size=MIN_PIXELS, connectivity=4, nodata_val=NODATA))
    with rasterio.open(sieved) as src:
        log.info("after a %d-pixel sieve: %.0f acres", MIN_PIXELS,
                 _acres(src.read(1) == CROP_CLASS))
    return sieved


def stage_compare(sieved_union: Path) -> None:
    """What the three changes did, separately and together."""
    import rasterio

    deployed = CANE / "30_Aug_2026" / "v4" / "static_mosaic_30_Aug_2026_Cls_v4_sieved_p20.tif"
    rows: List[Dict] = []
    for name, path in [("deployed: 30 Aug, 0.5-acre sieve", deployed),
                       ("this run: both dates, 0.15-acre sieve", sieved_union)]:
        with rasterio.open(path) as src:
            band = src.read(1)
        rows.append({"map": name, "cane_acres": round(_acres(band == CROP_CLASS)),
                     "classified_acres": round(_acres(band != NODATA))})

    with rasterio.open(deployed) as src:
        old = src.read(1) == CROP_CLASS
    with rasterio.open(sieved_union) as src:
        new = src.read(1) == CROP_CLASS

    print("\n" + "=" * 74)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\ncane in both maps          {_acres(old & new):>8,.0f} acres")
    print(f"only in the deployed map   {_acres(old & ~new):>8,.0f} acres")
    print(f"only in this run           {_acres(new & ~old):>8,.0f} acres")

    # Where the new cane came from: a date the deployed run never used, or ground the
    # coarse sieve had thrown away before the static model could look at it.
    with rasterio.open(OUT / f"static_10_Aug_2026_Cls_v4_p{MIN_PIXELS}mask.tif") as src:
        aug10 = src.read(1) == CROP_CLASS
    with rasterio.open(OUT / f"static_30_Aug_2026_Cls_v4_p{MIN_PIXELS}mask.tif") as src:
        aug30 = src.read(1) == CROP_CLASS
    gained = new & ~old
    print(f"\nof the gain, seen on 10 Aug only  {_acres(gained & aug10 & ~aug30):>8,.0f} acres")
    print(f"                 seen on 30 Aug     {_acres(gained & aug30):>8,.0f} acres")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["all", "sieve", "static", "union", "compare"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("static_training.inference").setLevel(logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)

    mask_path = OUT / f"rf_sieved_p{MIN_PIXELS}.tif"
    union_sieved = OUT / f"static_union_Cls_v4_strict_sieve_multiclass_p{MIN_PIXELS}.tif"

    if args.stage in ("all", "sieve"):
        mask_path = stage_sieve()
    if args.stage in ("all", "static"):
        stage_static(mask_path)
    if args.stage in ("all", "union"):
        union_sieved = stage_union([OUT / f"static_{f}_Cls_v4_p{MIN_PIXELS}mask.tif"
                                    for f, _ in DATES])
    if args.stage in ("all", "compare"):
        stage_compare(union_sieved)

    print(f"\noutputs -> {OUT}")
    print("\nnext, the field labelling against this map:")
    print(f"  python3 label_field_polygons.py --crop-map {union_sieved} \\\n"
          f"      --out {OUT / 'field_labelling'}")


if __name__ == "__main__":
    main()
