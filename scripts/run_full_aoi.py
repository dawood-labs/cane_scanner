"""The whole cane map for the full mill AOI, four dates, five delivered layers.

Scope, so the numbers below make sense: the AOI polygon is 1,044 km2 against the 90 km2
test feature, about twelve times the ground, tiled at 0.1 degrees into roughly twenty
tiles that actually carry AOI.

Stages, each timed, each skipping work already on disk:

    timeseries   Sentinel-2 fetch, smoothing and the RandomForest over the whole AOI
    sieve        the time-series map at 0.15 acres rather than 0.5
    static       the v4 model on 10 Aug, 30 Aug and 9 Sep inside that mask
    fuse         cane where any date says cane, sieved the same way
    label        field polygons against each of the five maps
    summary      what each map and each layer holds, and where they differ

Memory is the binding constraint, not cores: twelve cores and seven gigabytes. Each
worker holds a tile's whole time series, so the pool is deliberately smaller than the
core count. Raise --jobs only after watching the resident set of a run.

    python3 run_full_aoi.py --stage timeseries --jobs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

log = logging.getLogger("run_full_aoi")

CROPSCAN = SCRIPTS_DIR.parent
AOI_DIR = CROPSCAN / "data" / "Al-Moiz-Unit-1-SM-AOI-2025"
AOI_SHP = AOI_DIR / "Al-Moiz-Unit-1-SM-AOI-2025.shp"
OUT = AOI_DIR / "cane_2026"

RF_MODEL = CROPSCAN / "model_files" / "best_rf_classifier_v4.joblib"
STATIC_MODEL = CROPSCAN / "model_files" / "fao_cane_xgb_model_v4.json"

#: The window the RandomForest was trained to read, and the inference window inside it.
NDVI_START, NDVI_END = "2025-11-24", "2026-09-09"
INFER_START, INFER_END = "2025-12-09", "2026-09-08"

#: The three cloud-free static dates. 9 September is the newest and was checked by eye.
STATIC_DATES = [("10_Aug_2026", "2026-08-10"),
                ("30_Aug_2026", "2026-08-30"),
                ("09_Sep_2026", "2026-09-09")]

CROP_CLASS, BACKGROUND, NODATA = 1, 4, 255
MIN_PIXELS = 6          #: 0.15 acres at 10 m
MIN_ACRES = 0.15        #: and the same floor on the delivered polygons

#: Tiles laboured on at once. This is a memory budget, not a core count: a tile holds
#: its whole slice of the delineation, and they vary from about 1.7 to 2.5 GB depending
#: on how many polygons fall inside. Four was set from a peak reading of 5,043 MB taken
#: on the old 7.5 GB machine and was far too cautious: measured while running, four
#: tiles hold about 3 GB between them, some 800 MB each, on twelve cores with 6 GB idle.
#: Eight uses the machine and still leaves room for the spikes the dense tiles produce.
LABEL_JOBS = 8
UTM = 32642
SQM_PER_ACRE = 4046.8564224
EXPORT_SCALE, TILE_DEG = 10, 0.1

TIMINGS = OUT / "stage_timings.json"


class Timed:
    """Record how long a stage took, so optimisation later has somewhere to start."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        log.info("=== %s: starting", self.name)
        return self

    def __exit__(self, *exc):
        elapsed = time.time() - self.start
        log.info("=== %s: %s in %.1f minutes", self.name,
                 "failed" if exc[0] else "done", elapsed / 60)
        TIMINGS.parent.mkdir(parents=True, exist_ok=True)
        record = json.loads(TIMINGS.read_text()) if TIMINGS.exists() else {}
        record[self.name] = round(elapsed, 1)
        TIMINGS.write_text(json.dumps(record, indent=1))
        return False


def stage_timeseries(jobs: int, fetch_jobs: int) -> Path:
    """Fetch, smooth and classify the whole AOI. The long pole, by a wide margin."""
    from timeseries_pipeline import execute_stac_inference_pipeline

    OUT.mkdir(parents=True, exist_ok=True)
    basename = AOI_SHP.stem
    expected = OUT / f"{basename}_rf_classification_map.tif"
    if expected.exists():
        log.info("time-series map already exists: %s", expected.name)
        return expected

    produced = execute_stac_inference_pipeline(
        input_shp_path=str(AOI_SHP),
        model_path=str(RF_MODEL),
        final_out_dir=str(OUT),
        output_basename=basename,
        inference_start_date=INFER_START,
        inference_end_date=INFER_END,
        lmbd=0.5, d=2, clip_bounds=(-1.0, 1.0),
        n_jobs=jobs,
        export_raw_mosaic=False,        # not needed downstream and it is large
        export_smoothed_mosaic=True,    # the orchard filter reads this
        export_index_mask=True,
        delete_raw_tiles=False,         # so a rerun does not fetch them again
        ndvi_start=NDVI_START, ndvi_end=NDVI_END,
        res_m=EXPORT_SCALE, tile_deg=TILE_DEG,
        # Not I/O bound, whatever it looks like: each fetch worker holds one tile's
        # whole time series, about 700 MB, and eight of them took a 7 GB box to 126 MB
        # free. Three is what fits with the processing pool alongside it.
        fetch_workers=fetch_jobs,
    )
    return Path(produced)



LAYERS = [("01_timeseries", "the time-series model alone"),
          ("02_static_10Aug", "static v4 on 10 August"),
          ("03_static_30Aug", "static v4 on 30 August"),
          ("04_static_09Sep", "static v4 on 9 September"),
          ("05_fused_union", "cane where any date says cane"),
          ("06_fused_majority", "cane where two of the three dates agree")]


def _sieve(path: Path, out: Path) -> Path:
    """Sieve at the finer size, keeping the result where we want it."""
    from static_training import sieve as st_sieve

    if out.exists():
        log.info("already sieved: %s", out.name)
        return out
    produced = Path(st_sieve.apply_strict_directional_sieve(
        str(path), target_classes=[CROP_CLASS], min_pixel_size=MIN_PIXELS,
        connectivity=4, nodata_val=NODATA))
    out.parent.mkdir(parents=True, exist_ok=True)
    produced.replace(out)
    return out


def _acres(path: Path, value: int = CROP_CLASS) -> float:
    import rasterio

    with rasterio.open(path) as src:
        return float((src.read(1) == value).sum()) * 100.0 / SQM_PER_ACRE


def clip_to_aoi(path: Path, out: Path) -> Path:
    """Cut a raster down to the AOI polygon.

    Tiles are square and an AOI is not. This one is 1,044 km2 inside a 4,988 km2 box, so
    a tiled fetch always brings back ground the client did not ask for: the grid here
    covers 1,499,922 acres against the AOI's 258,011, and 31.9% of the time-series map's
    cane sat outside it.

    The static stage never had this problem because `create_aligned_mask` intersects
    with the AOI while building its mask. The time-series stage had no such step, so it
    was both delivering ground outside the AOI and making every stage behind it work on
    that ground. Clipping here fixes both at once, since everything downstream reads
    this raster.
    """
    import geopandas as gpd
    import rasterio
    from rasterio import features as rfeatures

    if out.exists():
        log.info("already clipped: %s", out.name)
        return out

    aoi = gpd.read_file(AOI_SHP, engine="pyogrio").to_crs(4326)
    with rasterio.open(path) as src:
        band = src.read(1)
        profile = src.profile.copy()
        inside = rfeatures.rasterize(aoi.geometry, out_shape=(src.height, src.width),
                                     transform=src.transform, fill=0,
                                     default_value=1, dtype="uint8").astype(bool)
    before = float((band == CROP_CLASS).sum()) * 100 / SQM_PER_ACRE
    band[~inside] = NODATA
    after = float((band == CROP_CLASS).sum()) * 100 / SQM_PER_ACRE
    log.info("clipped to the AOI: %.0f acres of cane became %.0f, %.0f acres dropped "
             "as outside", before, after, before - after)

    profile.update(compress="lzw", tiled=True, bigtiff="YES")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(band, 1)
        dst.set_band_description(1, "crop class, clipped to the AOI")
    return out


def stage_sieve() -> Path:
    """The time-series map at 0.15 acres instead of 0.5."""
    raw = OUT / f"{AOI_SHP.stem}_rf_classification_map.tif"
    # Clip first, sieve second: sieving ground that is about to be thrown away is work
    # for nothing, and a blob straddling the AOI edge should be judged on the part that
    # is inside.
    clipped = clip_to_aoi(raw, OUT / f"{AOI_SHP.stem}_rf_classification_map_aoi.tif")
    fine = OUT / f"rf_sieved_p{MIN_PIXELS}.tif"
    _sieve(clipped, fine)
    log.info("time-series cane: %.0f acres raw, %.0f inside the AOI, %.0f after a "
             "%d-pixel sieve", _acres(raw), _acres(clipped), _acres(fine), MIN_PIXELS)
    return fine


def stage_static(mask_path: Path, fetch_jobs: int = 3) -> List[Path]:
    """Fetch and classify each of the three dates inside the finer mask."""
    from static_pipeline import execute_static_pipeline

    produced = []
    for folder, date in STATIC_DATES:
        with Timed(f"static {folder}"):
            result = execute_static_pipeline(
                base_dir=OUT, mask_path=mask_path, input_shp_path=str(AOI_SHP),
                model_file=str(STATIC_MODEL), delete_tiles=False, use_mask=True,
                mask_keep_values=[CROP_CLASS], target_class_in=CROP_CLASS,
                target_class_out=CROP_CLASS, background_out=BACKGROUND,
                static_start=date, static_end=date,
                export_scale=EXPORT_SCALE, tile_deg=TILE_DEG,
                dates=[date], n_dates=1,
                fetch_workers=fetch_jobs,
            )
            if result is None:
                raise RuntimeError(f"the static stage produced nothing for {date}")
            fine = OUT / f"static_{folder}_Cls_v4_p{MIN_PIXELS}.tif"
            _sieve(Path(result), fine)
            log.info("%s: %.0f acres of cane after the sieve", folder, _acres(fine))
            produced.append(fine)
    return produced


def stage_fuse(classified: List[Path]) -> Dict[str, Path]:
    """Union and majority, both sieved the same way.

    Two answers because they disagree about the same thing: cane cut between two dates
    is bare on the later image. A union recovers those fields and inherits every date's
    false positives; a majority refuses a single date's mistake and loses the fields
    only one date could still see. The summary reports both so the choice is made on
    numbers rather than on preference.
    """
    import numpy as np
    import rasterio

    done = {name: OUT / f"fused_{name}_Cls_v4_p{MIN_PIXELS}.tif"
            for name in ("union", "majority")}
    if all(path.exists() for path in done.values()):
        for name, path in done.items():
            log.info("%s already fused: %.0f acres", name, _acres(path))
        return done

    stack, profile = [], None
    for path in classified:
        with rasterio.open(path) as src:
            stack.append(src.read(1))
            profile = src.profile.copy()

    votes = sum((band == CROP_CLASS).astype(np.uint8) for band in stack)
    seen = np.zeros(stack[0].shape, dtype=bool)
    for band in stack:
        seen |= band != NODATA

    out: Dict[str, Path] = {}
    profile.update(compress="lzw", tiled=True, dtype="uint8", nodata=NODATA, bigtiff="YES")
    for name, rule in [("union", votes >= 1), ("majority", votes >= 2)]:
        band = np.full(stack[0].shape, NODATA, dtype=np.uint8)
        band[seen] = BACKGROUND
        band[rule & seen] = CROP_CLASS
        raw = OUT / f"fused_{name}_Cls_v4.tif"
        with rasterio.open(raw, "w", **profile) as dst:
            dst.write(band, 1)
            dst.set_band_description(1, f"cane by {name} of three August/September dates")
        fine = OUT / f"fused_{name}_Cls_v4_p{MIN_PIXELS}.tif"
        _sieve(raw, fine)
        log.info("%-9s %.0f acres before the sieve, %.0f after", name,
                 float((band == CROP_CLASS).sum()) * 100 / SQM_PER_ACRE, _acres(fine))
        out[name] = fine
    return out


def stage_label(maps: Dict[str, Path]) -> None:
    """Field polygons for each map, each in its own folder."""
    import subprocess

    for (folder, _), key in zip(LAYERS, ["timeseries", "10Aug", "30Aug", "09Sep",
                                         "union", "majority"]):
        target = OUT / "outputs" / folder
        if (target / "fields_cane.parquet").exists():
            log.info("%s already labelled", folder)
            continue
        with Timed(f"label {folder}"):
            # Tiled, not straight through: repairing 17,074 polygons peaked at 4 GB
            # and the full delineation is 144,670, which would be thirty gigabytes on
            # a seven gigabyte machine. Each tile runs in its own process, which is
            # what actually returns the memory.
            subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "label_field_polygons_tiled.py"),
                 "--crop-map", str(maps[key]), "--out", str(target),
                 "--min-acres", str(MIN_ACRES), "--jobs", str(LABEL_JOBS),
                 "--no-gpkg"],
                check=True)


def stage_gpkg() -> None:
    """Write the GeoPackages once, at the end.

    GeoPackage is what opens everywhere and is what the client gets, but at this size it
    costs about a minute a layer where Parquet costs seconds. Six layers paid that
    during the loop; now they pay it once, here, after everything else is settled.
    """
    import geopandas as gpd

    for folder, _ in LAYERS:
        target = OUT / "outputs" / folder
        for name in ("fields_labelled", "fields_cane"):
            source = target / f"{name}.parquet"
            written = target / f"{name}.gpkg"
            if not source.exists() or written.exists():
                continue
            with Timed(f"gpkg {folder}/{name}"):
                gpd.read_parquet(source).to_file(written, driver="GPKG")


def stage_summary() -> None:
    """One table per question: what each map holds, and what each layer delivers."""
    import geopandas as gpd
    import pandas as pd

    maps = {
        "01_timeseries": OUT / f"rf_sieved_p{MIN_PIXELS}.tif",
        "02_static_10Aug": OUT / f"static_10_Aug_2026_Cls_v4_p{MIN_PIXELS}.tif",
        "03_static_30Aug": OUT / f"static_30_Aug_2026_Cls_v4_p{MIN_PIXELS}.tif",
        "04_static_09Sep": OUT / f"static_09_Sep_2026_Cls_v4_p{MIN_PIXELS}.tif",
        "05_fused_union": OUT / f"fused_union_Cls_v4_p{MIN_PIXELS}.tif",
        "06_fused_majority": OUT / f"fused_majority_Cls_v4_p{MIN_PIXELS}.tif",
    }

    rows = []
    for (folder, what) in LAYERS:
        raster = maps[folder]
        row = {"layer": folder, "what it is": what,
               "raster acres": round(_acres(raster)) if raster.exists() else None}
        cane = OUT / "outputs" / folder / "fields_cane.parquet"
        if cane.exists():
            frame = gpd.read_parquet(cane).to_crs(UTM)
            row["polygons"] = len(frame)
            row["polygon acres"] = round(frame.area.sum() / SQM_PER_ACRE)
            row["captured"] = (f"{100 * (frame.area.sum() / SQM_PER_ACRE) / _acres(raster):.1f}%"
                               if raster.exists() and _acres(raster) else None)
            # The number that decides whether dissolve is needed: if the polygons
            # overlap, a client adding up the acreage is billed for ground twice.
            union = frame.geometry.union_all().area / SQM_PER_ACRE
            row["double counted"] = round(frame.area.sum() / SQM_PER_ACRE - union, 3)
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("WHAT EACH LAYER HOLDS")
    print(table.to_string(index=False))
    (OUT / "outputs").mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "outputs" / "summary.csv", index=False)

    # Where the fused maps differ from each other and from the single dates.
    import numpy as np
    import rasterio

    def read(path):
        with rasterio.open(path) as src:
            return src.read(1) == CROP_CLASS

    if all(maps[k].exists() for k in ["02_static_10Aug", "03_static_30Aug",
                                      "04_static_09Sep", "05_fused_union",
                                      "06_fused_majority"]):
        a, b, c = (read(maps["02_static_10Aug"]), read(maps["03_static_30Aug"]),
                   read(maps["04_static_09Sep"]))
        union, majority = read(maps["05_fused_union"]), read(maps["06_fused_majority"])
        ac = lambda m: float(m.sum()) * 100 / SQM_PER_ACRE
        print("\nWHERE THE DATES DISAGREE")
        print(f"  all three agree it is cane      {ac(a & b & c):>9,.0f} acres")
        print(f"  only 10 August                  {ac(a & ~b & ~c):>9,.0f} acres")
        print(f"  only 30 August                  {ac(b & ~a & ~c):>9,.0f} acres")
        print(f"  only 9 September                {ac(c & ~a & ~b):>9,.0f} acres")
        print(f"\n  union minus majority            {ac(union & ~majority):>9,.0f} acres")
        print("  that difference is what one date alone claims: recovered harvest if")
        print("  the date is right, an inherited false positive if it is not.")

    if TIMINGS.exists():
        timings = json.loads(TIMINGS.read_text())
        print("\nTIME PER STAGE (minutes)")
        for name, seconds in timings.items():
            print(f"  {name:26s} {seconds / 60:>8.1f}")
        print(f"  {'TOTAL':26s} {sum(timings.values()) / 60:>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="timeseries",
                        choices=["all", "timeseries", "sieve", "static", "fuse", "label", "gpkg",
                                 "summary"])
    parser.add_argument("--fetch-jobs", type=int, default=3,
                        help="parallel Sentinel fetches; each holds a tile's whole "
                             "time series, so this is a memory budget, not a core count")
    parser.add_argument("--jobs", type=int, default=3,
                        help="worker processes; each holds a tile's whole time series, "
                             "so this is limited by memory rather than by cores")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    stages = (["timeseries", "sieve", "static", "fuse", "label", "gpkg", "summary"]
              if args.stage == "all" else [args.stage])
    mask = OUT / f"rf_sieved_p{MIN_PIXELS}.tif"

    if "timeseries" in stages:
        with Timed("timeseries"):
            stage_timeseries(args.jobs, args.fetch_jobs)
    if "sieve" in stages:
        with Timed("sieve"):
            mask = stage_sieve()
    if "static" in stages:
        stage_static(mask, args.fetch_jobs)
    if "fuse" in stages:
        with Timed("fuse"):
            stage_fuse([OUT / f"static_{f}_Cls_v4_p{MIN_PIXELS}.tif" for f, _ in STATIC_DATES])
    if "label" in stages:
        stage_label({
            "timeseries": mask,
            "10Aug": OUT / f"static_10_Aug_2026_Cls_v4_p{MIN_PIXELS}.tif",
            "30Aug": OUT / f"static_30_Aug_2026_Cls_v4_p{MIN_PIXELS}.tif",
            "09Sep": OUT / f"static_09_Sep_2026_Cls_v4_p{MIN_PIXELS}.tif",
            "union": OUT / f"fused_union_Cls_v4_p{MIN_PIXELS}.tif",
            "majority": OUT / f"fused_majority_Cls_v4_p{MIN_PIXELS}.tif",
        })
    if "gpkg" in stages:
        stage_gpkg()
    if "summary" in stages:
        stage_summary()

    print(f"\noutputs -> {OUT / 'outputs'}")


if __name__ == "__main__":
    main()
