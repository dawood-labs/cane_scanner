"""Assemble a cane-versus-orchard training set from data that already exists.

The orchard labels were sitting in plain sight. `Tree_mask_wgs84_final_FAO_simplify`
looks like a district layer, and its 59 outer polygons are districts, but the content
is in the interior rings: 67,004 holes punched out of those districts, each one a
manually delineated orchard or woodlot block. Reversing the mask recovers them as
positive labels, so nothing has to be digitised.

Four stages, each resumable:

    python3 build_orchard_training_set.py labels     # holes -> orchard polygons
    python3 build_orchard_training_set.py confusion  # keep the ones called cane
    python3 build_orchard_training_set.py chips    # pick where both classes are dense
    python3 build_orchard_training_set.py stacks   # fetch and smooth the NDVI series
    python3 build_orchard_training_set.py extract  # labelled pixels -> parquet
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import geopandas as gpd  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

import sentinel  # noqa: E402
from geo_inference_workers import parse_stac_bands, get_penalty_matrix, process_smoothing_chunk  # noqa: E402
from static_training import config as st_config  # noqa: E402

log = logging.getLogger("build_orchard_training_set")

FAO = st_config.FAO_ROOT / "cane"
TREE_MASK = (FAO / "new_tree_masked_dist_data" / "Tree_mask_wgs84_final_FAO_simplify"
             / "Tree_mask_wgs84_final_FAO_simplify.shp")
CANE_POLYGONS = FAO / "static_model" / "cane_polygons" / "sugarmills_cropscan_clipped_after_qc.shp"

#: The 2025 national cane map, 771,376 polygons. Intersecting it with the orchard
#: blocks is what turns "here are some orchards" into "here are the orchards the
#: model actually got wrong", which is the only thing worth training on.
CANE_NATIONAL = (FAO / "Sugarcane_3m-10m_Pakistan-Scan_2025"
                 / "Sugarcane_3m-10m_Pakistan-Scan_2025.shp")

OUT = FAO / "orchard_detector"
ORCHARD_GPKG = OUT / "orchard_polygons.gpkg"
CONFUSED_GPKG = OUT / "orchards_called_cane.gpkg"
RF_DIR = OUT / "rf_stage"
CHIPS_GPKG = OUT / "chips.gpkg"
STACK_DIR = OUT / "stacks"
PARQUET_DIR = OUT / "parquets"

#: Districts to sample. Orchard density alone is not the criterion: the chip has to
#: contain cane as well, and several orchard-heavy districts have almost none.
#: Ranked by cane polygons present, these three lead the list while spanning both
#: provinces. Layyah was the first choice and was dropped: its orchards and its cane
#: sit in different parts of the district, so no single chip holds both.
#:
#: A second filter turned out to matter more than orchard count: whether the mask
#: holes in that district are actually orchards. In arid Bhakkar they run 0.03 to
#: 0.40 NDVI and peak in February, which is desert scrub; in Mirpur Khas they are
#: darker than the cane around them. Only the irrigated belt has the green, flat
#: perennials that get mistaken for standing cane, so the list is now weighted
#: towards the mango districts. train_orchard_detector.py audits this and drops any
#: district that fails, so adding one here is safe.
#:
#:   district          orchard blocks   cane polygons   holes are orchards
#:   Rahim Yar Khan             2,880          10,399   yes, 0.67 mean NDVI
#:   Multan                     2,812              74   mango belt, to be audited
#:   Khanewal                   1,751             337   mango belt, to be audited
#:   Mirpur Khas                4,499           3,680   no, darker than the cane
#:   Bhakkar                   15,407             488   no, desert scrub
DISTRICTS = ["Bhakkar", "Rahim Yar Khan", "Mirpur Khas", "Multan", "Khanewal"]

#: Orchard blocks smaller than this are noise; larger ones are riverine and forest
#: belts rather than orchards, and the untrimmed tail runs to 14,667 ha.
MIN_ORCHARD_HA, MAX_ORCHARD_HA = 0.5, 100.0

#: One chip per district. 8 km keeps each NDVI stack around 800x800 px.
CHIP_KM = 8.0
SEARCH_CELL_KM = 2.0

#: Longitude width of the strips the national cane layer is read in.
STRIP_DEG = 0.25

#: The window the deployed pipeline runs on, so the detector sees at inference
#: exactly the series it was fitted on.
NDVI_START, NDVI_END, STEP_DAYS = "2025-11-24", "2026-09-09", 8
SMOOTH_LAMBDA, SMOOTH_ORDER, CLIP_BOUNDS = 0.5, 2, (-1.0, 1.0)

ORCHARD_LABEL, CANE_LABEL = 2, 1

#: The deployed time-series model, and the window it reads.
RF_MODEL = st_config.CROPSCAN_ROOT / "model_files" / "best_rf_classifier_v4.joblib"
RF_INFERENCE_START, RF_INFERENCE_END = "2025-12-09", "2026-09-08"
UTM = 32642
SQM_PER_HA = 10_000.0


# ---------------------------------------------------------------------- labels

def stage_labels() -> None:
    """Turn the mask's interior rings into orchard polygons."""
    OUT.mkdir(parents=True, exist_ok=True)
    if ORCHARD_GPKG.exists():
        log.info("%s already exists, skipping", ORCHARD_GPKG.name)
        return

    where = " OR ".join(f"DISTRICT = '{d}'" for d in DISTRICTS)
    districts = gpd.read_file(TREE_MASK, where=where, engine="pyogrio")
    log.info("districts read: %s", list(districts.DISTRICT))

    records: List[Dict] = []
    for row in districts.itertuples():
        geom = row.geometry
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            for ring in part.interiors:
                records.append({"district": row.DISTRICT, "geometry": Polygon(ring)})

    orchards = gpd.GeoDataFrame(records, crs=districts.crs)
    orchards["area_ha"] = orchards.to_crs(UTM).area / SQM_PER_HA
    log.info("interior rings recovered: %d", len(orchards))

    keep = orchards.area_ha.between(MIN_ORCHARD_HA, MAX_ORCHARD_HA)
    log.info("after the %.1f-%.0f ha filter: %d (dropped %d small, %d oversized)",
             MIN_ORCHARD_HA, MAX_ORCHARD_HA, int(keep.sum()),
             int((orchards.area_ha < MIN_ORCHARD_HA).sum()),
             int((orchards.area_ha > MAX_ORCHARD_HA).sum()))
    orchards = orchards[keep].reset_index(drop=True)

    orchards.to_file(ORCHARD_GPKG, driver="GPKG")
    log.info("per district:\n%s", orchards.groupby("district").agg(
        n=("area_ha", "size"), median_ha=("area_ha", "median"),
        total_ha=("area_ha", "sum")).round(1).to_string())
    log.info("written -> %s", ORCHARD_GPKG)


# -------------------------------------------------------------------- rf stage

def stage_rfstage() -> None:
    """Run the time-series model over each chip and keep what it calls cane.

    This is the layer the orchard problem actually lives in, and getting it wrong
    cost a full pass of this pipeline. The national cane map is the finished
    product: the static model has already been over it, so orchards barely survive
    into it at all, 314 ha across five districts, and labels drawn from it describe
    a problem that has already been solved. Measured on the time-series output
    instead, orchards are 6 to 10% of what it calls cane in the mango belt.

    So the positives have to come from here: orchard ground that this stage, not the
    finished map, calls cane.
    """
    import joblib
    import rasterio

    RF_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(RF_MODEL)
    log.info("time-series model: %d features, classes %s",
             model.n_features_in_, model.classes_)

    for stack_path in sorted(STACK_DIR.glob("*_smoothed.tif")):
        out_path = RF_DIR / f"{stack_path.stem.replace('_smoothed', '')}_rf_cane.tif"
        if out_path.exists():
            log.info("%s already exists, skipping", out_path.name)
            continue
        with rasterio.open(stack_path) as src:
            names = list(src.descriptions)
            series = src.read()
            profile = src.profile.copy()

        dates = pd.to_datetime([n.replace("NDVI_", "").replace("_", "-") for n in names])
        window = np.where((dates >= pd.Timestamp(RF_INFERENCE_START))
                          & (dates <= pd.Timestamp(RF_INFERENCE_END)))[0]
        if len(window) != model.n_features_in_:
            log.error("%s: date slice gives %d features, model wants %d",
                      out_path.name, len(window), model.n_features_in_)
            continue

        pixels = series[window].reshape(len(window), -1).T.astype(np.float32)
        usable = ~np.all(np.isnan(pixels) | (pixels == 0), axis=1)
        predicted = np.full(pixels.shape[0], 255, dtype=np.uint8)
        predicted[usable] = model.predict(
            np.nan_to_num(pixels[usable], nan=0.0)).astype(np.uint8)

        profile.update(count=1, dtype="uint8", nodata=255, compress="lzw", tiled=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(predicted.reshape(series.shape[1:]), 1)
            dst.set_band_description(1, "timeseries_rf_class")
        log.info("%s: %d cane pixels", out_path.name, int((predicted == CANE_LABEL).sum()))


# ------------------------------------------------------------------- confusion

def stage_confusion() -> None:
    """Keep only the orchard blocks the national cane map already calls cane.

    Training on orchards and cane sampled independently teaches a contrast that was
    never hard: they sat in different places and looked different. The blocks that
    matter are the ones a cane map has already mistaken, and there are a great many
    of them: 57% of the orchard blocks in Rahim Yar Khan and 45% in Mirpur Khas,
    about 31,800 ha between the two districts.

    The cane layer is read in narrow strips. Loaded whole it needs several gigabytes,
    and the exact-overlap version of this walked into the memory watchdog.
    """
    import gc

    if CONFUSED_GPKG.exists():
        log.info("%s already exists, skipping", CONFUSED_GPKG.name)
        return
    orchards = gpd.read_file(ORCHARD_GPKG, engine="pyogrio")
    keep: List[gpd.GeoDataFrame] = []

    for district in sorted(orchards.district.unique()):
        sub = orchards[orchards.district == district].reset_index(drop=True)
        bounds = sub.total_bounds
        hits: set = set()
        edges = np.arange(bounds[0], bounds[2] + STRIP_DEG, STRIP_DEG)
        for left, right in zip(edges[:-1], edges[1:]):
            cane = gpd.read_file(CANE_NATIONAL, bbox=(left, bounds[1], right, bounds[3]),
                                 engine="pyogrio")
            if cane.empty:
                continue
            strip = sub.cx[left:right, bounds[1]:bounds[3]]
            if not strip.empty:
                joined = gpd.sjoin(strip[["geometry"]], cane[["geometry"]],
                                   how="inner", predicate="intersects")
                hits.update(joined.index.tolist())
            del cane
            gc.collect()
        if hits:
            keep.append(sub.loc[sorted(hits)])
        log.info("%-16s %5d of %5d orchard blocks are called cane (%.1f%%)",
                 district, len(hits), len(sub), 100 * len(hits) / len(sub))

    if not keep:
        raise SystemExit("no orchard block is called cane; there is nothing to fix")
    confused = gpd.GeoDataFrame(pd.concat(keep, ignore_index=True), crs=orchards.crs)
    confused.to_file(CONFUSED_GPKG, driver="GPKG")
    log.info("confused orchard blocks: %d, %.0f ha -> %s",
             len(confused), confused.area_ha.sum(), CONFUSED_GPKG)


# ----------------------------------------------------------------------- chips

def _best_window(points: gpd.GeoDataFrame, labels: np.ndarray, chip_m: float,
                 cell_m: float) -> Optional[Polygon]:
    """The chip position holding the most of whichever class is scarcer there.

    Scoring on the minimum of the two counts rather than the sum is what stops the
    search settling on a pure orchard block or a pure cane block; the detector needs
    both classes inside one stack.
    """
    if points.empty:
        return None
    xs, ys = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
    x0, y0 = xs.min(), ys.min()
    ix = ((xs - x0) // cell_m).astype(int)
    iy = ((ys - y0) // cell_m).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1

    grids = {}
    for value in (ORCHARD_LABEL, CANE_LABEL):
        grid = np.zeros((ny, nx), dtype=np.int32)
        sel = labels == value
        np.add.at(grid, (iy[sel], ix[sel]), 1)
        grids[value] = grid.cumsum(0).cumsum(1)

    span = max(1, int(round(chip_m / cell_m)))

    def window_sum(cum: np.ndarray, r: int, c: int) -> int:
        r1, c1 = min(r + span, ny) - 1, min(c + span, nx) - 1
        total = cum[r1, c1]
        if r: total -= cum[r - 1, c1]
        if c: total -= cum[r1, c - 1]
        if r and c: total += cum[r - 1, c - 1]
        return int(total)

    best, best_rc = -1, None
    for r in range(max(1, ny - span + 1)):
        for c in range(max(1, nx - span + 1)):
            score = min(window_sum(grids[ORCHARD_LABEL], r, c),
                        window_sum(grids[CANE_LABEL], r, c))
            if score > best:
                best, best_rc = score, (r, c)
    if best <= 0 or best_rc is None:
        return None

    r, c = best_rc
    left, bottom = x0 + c * cell_m, y0 + r * cell_m
    log.info("   best window holds at least %d of each class", best)
    return box(left, bottom, left + chip_m, bottom + chip_m)


def stage_chips() -> None:
    if CHIPS_GPKG.exists():
        log.info("%s already exists, skipping", CHIPS_GPKG.name)
        return
    source = CONFUSED_GPKG if CONFUSED_GPKG.exists() else ORCHARD_GPKG
    log.info("placing chips on %s", source.name)
    orchards = gpd.read_file(source, engine="pyogrio").to_crs(UTM)
    cane = gpd.read_file(CANE_POLYGONS, bbox=tuple(orchards.to_crs(4326).total_bounds),
                         engine="pyogrio").to_crs(UTM)
    log.info("cane polygons in range: %d", len(cane))

    chips = []
    for district in DISTRICTS:
        sub = orchards[orchards.district == district]
        if sub.empty:
            log.warning("%s: no orchards, skipped", district)
            continue
        bounds = sub.total_bounds
        near_cane = cane.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        log.info("%s: %d orchards, %d cane polygons in its bounds",
                 district, len(sub), len(near_cane))
        if near_cane.empty:
            log.warning("%s: no cane nearby, skipped", district)
            continue

        points = gpd.GeoDataFrame(
            geometry=list(sub.geometry.representative_point())
            + list(near_cane.geometry.representative_point()), crs=UTM)
        labels = np.array([ORCHARD_LABEL] * len(sub) + [CANE_LABEL] * len(near_cane))
        window = _best_window(points, labels, CHIP_KM * 1000, SEARCH_CELL_KM * 1000)
        if window is None:
            log.warning("%s: no window holds both classes, skipped", district)
            continue
        chips.append({"district": district, "geometry": window})

    if not chips:
        raise SystemExit("no chips found")
    frame = gpd.GeoDataFrame(chips, crs=UTM).to_crs(4326)
    frame.to_file(CHIPS_GPKG, driver="GPKG")
    for row in frame.itertuples():
        log.info("chip %-12s %s", row.district,
                 [round(v, 4) for v in row.geometry.bounds])
    log.info("written -> %s", CHIPS_GPKG)


# ---------------------------------------------------------------------- stacks

def _band_descriptions(candidates: List[Optional[Path]]) -> Optional[tuple]:
    """Band names from the first candidate that still carries them.

    The AOI-clipped mosaic drops band descriptions on the way through GDAL, so the
    red/nir positions cannot be recovered from it. The per-tile GeoTIFFs written by
    the fetcher keep them, and share the clipped file's band order.
    """
    import rasterio

    for path in candidates:
        if path and Path(path).exists():
            with rasterio.open(path) as src:
                if any(src.descriptions):
                    return src.descriptions
    return None


def _smooth_stack(raw_path: Path, out_path: Path,
                  descriptions: Optional[tuple] = None) -> Path:
    """NDVI from the red/nir stack, Whittaker-smoothed exactly as the pipeline does."""
    import rasterio

    with rasterio.open(raw_path) as src:
        names = descriptions or src.descriptions
        if not any(names):
            raise ValueError(f"{raw_path.name} carries no band descriptions")
        red_idx, nir_idx, dates = parse_stac_bands(names)
        profile = src.profile.copy()
        red = src.read([i + 1 for i in red_idx]).astype(np.float32)
        nir = src.read([i + 1 for i in nir_idx]).astype(np.float32)

    denom = nir + red
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    np.divide(nir - red, denom, out=ndvi, where=denom > 0)

    penalty = get_penalty_matrix(ndvi.shape[0], SMOOTH_LAMBDA, SMOOTH_ORDER)
    smoothed = process_smoothing_chunk(ndvi, penalty, CLIP_BOUNDS, np.nan)

    profile.update(count=smoothed.shape[0], dtype="float32", nodata=np.nan,
                   compress="lzw", tiled=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(smoothed.astype(np.float32))
        for i, date in enumerate(dates, start=1):
            dst.set_band_description(i, f"NDVI_{date.replace('-', '_')}")
    log.info("   smoothed %d dates -> %s", smoothed.shape[0], out_path.name)
    return out_path


def stage_stacks(workers: int) -> None:
    STACK_DIR.mkdir(parents=True, exist_ok=True)
    chips = gpd.read_file(CHIPS_GPKG, engine="pyogrio")

    for row in chips.itertuples():
        out_path = STACK_DIR / f"{row.district.replace(' ', '_')}_smoothed.tif"
        if out_path.exists():
            log.info("%s already built, skipping", out_path.name)
            continue
        staging = STACK_DIR / f".staging_{row.district.replace(' ', '_')}"
        log.info("=== %s", row.district)
        try:
            result = sentinel.fetch_sentinel_imagery(
                aoi=row.geometry, start=NDVI_START, end=NDVI_END,
                bands=["red", "nir"], out_dir=str(staging), step=STEP_DAYS,
                res_m=10, tile_deg=0.1, cloud_lt=97, workers=workers,
                build_vrt_mosaic=True, clip_to_aoi=True,
            )
            raw = result.get("clipped") or result.get("vrt")
            if not raw or not Path(raw).exists():
                log.error("%s: no mosaic produced", row.district)
                continue
            tiles = sorted(staging.glob("sentinel_*m_tile_*.tif"))
            names = _band_descriptions([Path(raw), result.get("vrt"),
                                        tiles[0] if tiles else None])
            _smooth_stack(Path(raw), out_path, names)
        except Exception as exc:
            log.error("%s failed: %s", row.district, exc)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------- extract

def stage_extract() -> None:
    """Label every pixel of each stack as orchard, cane, or neither."""
    import rasterio
    from rasterio import features as rfeatures

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    # Positives and negatives both come from inside what the time-series model calls
    # cane. That is the only place the confusion exists, and a classifier asked to
    # separate two things it will never be shown together learns the wrong contrast.
    log.info("orchard positives from %s, restricted to time-series cane",
             ORCHARD_GPKG.name)
    orchards = gpd.read_file(ORCHARD_GPKG, engine="pyogrio")

    for stack_path in sorted(STACK_DIR.glob("*_smoothed.tif")):
        district = stack_path.stem.replace("_smoothed", "").replace("_", " ")
        out = PARQUET_DIR / f"{stack_path.stem}.parquet"
        if out.exists():
            log.info("%s already extracted, skipping", out.name)
            continue

        with rasterio.open(stack_path) as src:
            transform, shape, bounds = src.transform, (src.height, src.width), src.bounds
            dates = list(src.descriptions)
            series = src.read().astype(np.float32)

        rf_path = RF_DIR / f"{stack_path.stem.replace('_smoothed', '')}_rf_cane.tif"
        if not rf_path.exists():
            log.warning("%s: no time-series output, run the rfstage stage", district)
            continue
        with rasterio.open(rf_path) as src:
            rf_cane = src.read(1) == CANE_LABEL

        sub = orchards[orchards.district == district]
        log.info("%s: %d orchard blocks, %d pixels called cane by the time-series model",
                 district, len(sub), int(rf_cane.sum()))
        if sub.empty or not rf_cane.any():
            log.warning("%s: nothing to contrast, skipped", district)
            continue

        def burn(frame: gpd.GeoDataFrame, offset: int) -> np.ndarray:
            """Field ids per pixel, 0 outside. Boundary pixels are left out: a 10 m
            pixel on the edge of an orchard block is half tree and half whatever
            borders it, and belongs to neither class."""
            if frame.empty:
                return np.zeros(shape, dtype=np.int32)
            ids = ((geom, offset + i) for i, geom in enumerate(frame.geometry, start=1))
            filled = rfeatures.rasterize(ids, out_shape=shape, transform=transform,
                                         fill=0, dtype="int32", all_touched=True)
            edge = rfeatures.geometry_mask(frame.geometry.boundary, out_shape=shape,
                                           transform=transform, invert=True,
                                           all_touched=True)
            filled[edge] = 0
            return filled

        # Carrying the source polygon on every row lets training aggregate to the
        # field, which is the scale an orchard actually exists at, and lets validation
        # hold out whole fields instead of splitting one across the fold boundary.
        orchard_ids = burn(sub, 0)
        # Positive: orchard ground the time-series model called cane, the error
        # itself. Negative: everything else it called cane, which is the crop.
        orchard_in_cane = (orchard_ids > 0) & rf_cane
        cane_only = rf_cane & (orchard_ids == 0)
        from scipy import ndimage

        cane_ids, _ = ndimage.label(cane_only, structure=np.ones((3, 3)))
        cane_ids = np.where(cane_only, cane_ids + 1_000_000, 0)
        masks = {ORCHARD_LABEL: orchard_in_cane, CANE_LABEL: cane_only}
        overlap = np.zeros_like(orchard_in_cane)
        valid = np.isfinite(series).all(axis=0)

        parts = []
        for label, mask in masks.items():
            keep = mask & ~overlap & valid
            if not keep.any():
                continue
            values = series[:, keep].T
            frame = pd.DataFrame(values, columns=dates)
            frame["label"] = label
            frame["field_id"] = (orchard_ids if label == ORCHARD_LABEL else cane_ids)[keep]
            parts.append(frame)
            log.info("   label %d: %d pixels across %d fields",
                     label, int(keep.sum()), int(frame.field_id.nunique()))

        if not parts:
            continue
        table = pd.concat(parts, ignore_index=True)
        table["district"] = district
        table.to_parquet(out, index=False)
        log.info("   -> %s (%d rows)", out.name, len(table))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["labels", "confusion", "chips", "stacks",
                                          "rfstage", "extract", "all"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sentinel").setLevel(logging.WARNING)

    if args.stage in ("labels", "all"):
        stage_labels()
    if args.stage in ("confusion", "all"):
        stage_confusion()
    if args.stage in ("chips", "all"):
        stage_chips()
    if args.stage in ("stacks", "all"):
        stage_stacks(args.workers)
    if args.stage in ("rfstage", "all"):
        stage_rfstage()
    if args.stage in ("extract", "all"):
        stage_extract()


if __name__ == "__main__":
    main()
