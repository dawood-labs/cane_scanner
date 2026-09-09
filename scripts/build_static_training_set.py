"""Build a multi-date, provenance-carrying training table for the static cane model.

The deployed model was fitted on a single three-week window, 24 October to 8
November, and loses a third of its recall on an August image of a standing crop.
This script produces the training data that fixes that: the same labelled fields,
seen on several acquisitions across the season, with the acquisition date and AOI
recorded on every row so the result can finally be validated by date.

Run in two stages so a network failure never costs the extraction work:

    python3 build_static_training_set.py scout     # pick cloud-free dates per AOI
    python3 build_static_training_set.py export    # fetch imagery for those dates
    python3 build_static_training_set.py extract   # labelled pixels -> parquet
    python3 build_static_training_set.py merge     # label-transfer check, then one table
    python3 build_static_training_set.py verify    # does it still match the deployed model

Every stage is resumable; work already on disk is skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

import sentinel  # noqa: E402
from static_training import config as st_config  # noqa: E402
from static_training import extract as st_extract  # noqa: E402

log = logging.getLogger("build_static_training_set")

FAO = st_config.FAO_ROOT / "cane" / "static_model"
OUT_ROOT = FAO / "v4"
IMAGES_DIR = OUT_ROOT / "images"
PARQUET_DIR = OUT_ROOT / "parquets"
SCOUT_JSON = OUT_ROOT / "scout_dates.json"
MERGED_PARQUET = OUT_ROOT / "master_multidate.parquet"

#: Sentinel-2 bands to fetch, in the order the model reads them.
FETCH_BANDS = ["blue", "green", "red", "rededge1", "nir", "ndvi"]
BAND_ORDER = ["B2", "B3", "B4", "B5", "B8", "NDVI"]

#: Scouting windows. August to end of September, which is when the mill needs its
#: map. The existing October and November rows already cover the FAO window.
WINDOWS = [
    ("2025-08-01", "2025-08-21"),
    ("2025-08-21", "2025-09-11"),
    ("2025-09-11", "2025-10-01"),
]

#: An AOI-date pair is only used if this much of the AOI is cloud-free. Training on
#: a partly clouded scene teaches the model that cloud is cane.
MIN_COVERAGE_PCT = 99.0

#: Cap per class per AOI-date, so one large district cannot dominate the table.
MAX_PIXELS_PER_CLASS = 400_000


# --------------------------------------------------------------------------- AOIs

def _v1_aois() -> List[Dict]:
    """The 40 named sugar-mill AOIs. All have complete QC'd label coverage."""
    gdf = gpd.read_file(FAO / "cane_aoi_boundary" / "sugarmills_polygons.shp", engine="pyogrio")
    gdf = gdf[gdf.Name.notna()]
    return [{"aoi": str(r.Name), "geometry": r.geometry, "source": "v1"} for r in gdf.itertuples()]


def _v3_aois() -> List[Dict]:
    """v3 AOIs that carry labels and do not duplicate a v1 AOI.

    Ids 3, 4 and 6 overlap v1 jdw1, LSM and sheikhoo by 79-95%; including them would
    put the same ground under two AOI names and quietly break leave-one-AOI-out.
    Ids 2 and 6 have no label polygons at all.
    """
    keep = {1, 5, 11, 14}
    path = FAO / "v3" / "sugarcane_model_training_data_boundary" / "sugarcane_model_training_data_boundary.shp"
    gdf = gpd.read_file(path, engine="pyogrio")
    gdf = gdf[gdf.Date.notna() & gdf.id.isin(keep)]
    return [{"aoi": f"v3_{int(r.id)}", "geometry": r.geometry, "source": "v3"} for r in gdf.itertuples()]


def load_aois() -> List[Dict]:
    aois = _v1_aois() + _v3_aois()
    log.info("AOIs: %d (%d v1, %d v3)", len(aois),
             sum(a["source"] == "v1" for a in aois), sum(a["source"] == "v3" for a in aois))
    return aois


def load_label_polygons() -> gpd.GeoDataFrame:
    """Every cane label polygon, from both v1 and v3, in one layer.

    Positives are assigned to an AOI by geometry rather than by filename, because
    the v3 label files are named after mills whose polygons do not all fall inside
    the boundary of the same name.
    """
    frames = [gpd.read_file(FAO / "cane_polygons" / "sugarmills_cropscan_clipped_after_qc.shp",
                            engine="pyogrio")[["geometry"]]]
    for shp in sorted((FAO / "v3" / "sugarcan_cropscan").glob("*/*.shp")):
        frames.append(gpd.read_file(shp, engine="pyogrio")[["geometry"]])
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    log.info("label polygons: %d", len(merged))
    return merged


# ------------------------------------------------------------------------- scout

def stage_scout(aois: List[Dict], workers: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    existing = json.loads(SCOUT_JSON.read_text()) if SCOUT_JSON.exists() else {}

    def scout_one(entry: Dict) -> tuple[str, List[Dict]]:
        picks = []
        for start, end in WINDOWS:
            try:
                sel = sentinel.select_static_dates(
                    entry["geometry"], start, end, n_dates=1, cloud_lt=95,
                    cloud_metric="aoi", selection_mode="greedy", workers=4,
                )
                picks.append({
                    "window": f"{start}..{end}",
                    "date": sel["anchor"],
                    "coverage_pct": round(float(sel["coverage_pct"]), 2),
                })
            except Exception as exc:
                picks.append({"window": f"{start}..{end}", "error": f"{type(exc).__name__}: {exc}"})
        return entry["aoi"], picks

    todo = [a for a in aois if a["aoi"] not in existing]
    log.info("scouting %d AOIs (%d already done)", len(todo), len(existing))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scout_one, a) for a in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            name, picks = fut.result()
            existing[name] = picks
            SCOUT_JSON.write_text(json.dumps(existing, indent=2))
            log.info("[%d/%d] %s", i, len(todo), name)

    usable = sum(1 for picks in existing.values()
                 for p in picks if p.get("coverage_pct", 0) >= MIN_COVERAGE_PCT)
    log.info("scout complete: %d AOI-date pairs at >= %.0f%% coverage", usable, MIN_COVERAGE_PCT)


def selected_pairs() -> List[tuple[str, str]]:
    """(aoi, date) pairs that passed the cloud-free bar, deduplicated."""
    if not SCOUT_JSON.exists():
        raise SystemExit("run the scout stage first")
    scout = json.loads(SCOUT_JSON.read_text())
    pairs = set()
    for aoi, picks in scout.items():
        for p in picks:
            if p.get("coverage_pct", 0) >= MIN_COVERAGE_PCT and p.get("date"):
                pairs.add((aoi, p["date"]))
    return sorted(pairs)


# ------------------------------------------------------------------------ export

def _image_path(aoi: str, date: str) -> Path:
    return IMAGES_DIR / f"{aoi}_{date}.tif"


def stage_export(aois: List[Dict], workers: int) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    by_name = {a["aoi"]: a for a in aois}
    pairs = [(a, d) for a, d in selected_pairs() if not _image_path(a, d).exists()]
    log.info("exporting %d AOI-date images", len(pairs))

    for i, (aoi, date) in enumerate(pairs, 1):
        staging = IMAGES_DIR / f".staging_{aoi}_{date}"
        try:
            result = sentinel.fetch_sentinel_static_imagery(
                aoi=by_name[aoi]["geometry"], start=date, end=date,
                bands=FETCH_BANDS, out_dir=str(staging), res_m=10, tile_deg=0.2,
                dates=[date], n_dates=1, mask_clouds=False, workers=workers,
                build_vrt_mosaic=True, clip_to_aoi=True, dtype="uint16",
            )
            clipped = result.get("clipped")
            if not clipped or not Path(clipped).exists():
                log.error("[%d/%d] %s %s: no clipped mosaic produced", i, len(pairs), aoi, date)
                continue
            Path(clipped).replace(_image_path(aoi, date))
            log.info("[%d/%d] %s %s exported", i, len(pairs), aoi, date)
        except Exception as exc:
            log.error("[%d/%d] %s %s failed: %s", i, len(pairs), aoi, date, exc)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


# ----------------------------------------------------------------------- extract

#: Image filenames that do not exactly match their AOI polygon's Name attribute.
_AOI_ALIASES = {"multan_dist_2": "multan_dist2"}


def _existing_images() -> List[tuple[str, str, Path]]:
    """The October and November images already on disk, so their rows gain provenance."""
    import re

    out = []
    for path in sorted((FAO / "images").glob("*.tif")):
        m = re.search(r"(\d{4})-([A-Za-z]{3})-(\d{2})", path.name)
        if not m:
            continue
        month = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
                 "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}[m.group(2)]
        date = f"{m.group(1)}-{month}-{m.group(3)}"
        aoi = path.name.replace("fao_cane_training_images_", "").replace(f"_{m.group(0)}.tif", "")
        # One image is filed as multan_dist_2 while its AOI polygon is multan_dist2.
        aoi = _AOI_ALIASES.get(aoi, aoi)
        out.append((aoi, date, path))
    for path in sorted((FAO / "v3" / "images").glob("*.tif")):
        m = re.search(r"_(\d+)_(\d{4})-([A-Za-z]{3})-(\d{2})\.tif$", path.name)
        if not m:
            continue
        if int(m.group(1)) not in {1, 5, 11, 14}:
            continue
        month = {"Oct": "10", "Nov": "11"}[m.group(3)]
        out.append((f"v3_{m.group(1)}", f"{m.group(2)}-{month}-{m.group(4)}", path))
    return out


def stage_extract(aois: List[Dict], workers: int) -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    cfg = st_config.get("cane")
    by_name = {a["aoi"]: a for a in aois}
    labels = load_label_polygons()
    label_sindex = labels.sindex

    jobs: List[tuple[str, str, Path]] = [
        (aoi, date, _image_path(aoi, date))
        for aoi, date in selected_pairs()
        if _image_path(aoi, date).exists()
    ]
    jobs += [(a, d, p) for a, d, p in _existing_images() if a in by_name]

    def run(job) -> Optional[str]:
        aoi, date, path = job
        out = PARQUET_DIR / f"{aoi}_{date}.parquet"
        if out.exists():
            return None
        geom = by_name[aoi]["geometry"]
        hits = list(label_sindex.query(geom, predicate="intersects"))
        positive = unary_union(labels.geometry.iloc[hits].values).intersection(geom) if hits else None

        result = st_extract.extract_from_raster(
            path, positive, geom, aoi, date,
            crop_label=cfg.crop_label, background_label=cfg.background_label,
            band_order=BAND_ORDER, max_pixels_per_class=MAX_PIXELS_PER_CLASS,
        )
        if not result.ok:
            return f"{aoi} {date}: skipped ({result.skipped_reason})"
        frame = st_extract.apply_outlier_caps(result.frame, cfg.outlier_caps)
        frame.to_parquet(out, index=False)
        counts = frame.label.value_counts().to_dict()
        return f"{aoi} {date}: {len(frame):,} rows {counts}"

    log.info("extracting from %d images", len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            message = fut.result()
            if message:
                log.info("[%d/%d] %s", i, len(jobs), message)


# ------------------------------------------------------------------------- merge

#: Cohen's d on NDVI between the two classes. Below this the labels no longer
#: describe what the image shows on that date, and the rows would teach the model
#: that cane and background look alike.
MIN_SEPARABILITY_D = 0.8

#: Below this many pixels in a class, separability cannot be measured for that pair.
MIN_CLASS_PIXELS = 100


def label_transfer_check(merged: pd.DataFrame, crop_label: int = 1) -> pd.DataFrame:
    """Test the assumption that a November label still holds on an earlier date.

    Reusing labels across dates only works while the crop is in the ground. If a
    date sits outside that span, cane and background stop being distinguishable and
    the separation collapses. Measuring it per AOI-date turns that assumption into
    something checked rather than believed.
    """
    import numpy as np

    rows = []
    for (aoi, date), group in merged.groupby(["aoi", "date"]):
        positive = group.loc[group.label == crop_label, "NDVI"].to_numpy(dtype=float)
        negative = group.loc[group.label != crop_label, "NDVI"].to_numpy(dtype=float)
        record = {
            "aoi": aoi, "date": date, "n": len(group),
            "n_positive": int(positive.size), "n_negative": int(negative.size),
        }
        if positive.size < MIN_CLASS_PIXELS or negative.size < MIN_CLASS_PIXELS:
            # Some AOIs are almost entirely background: multan_dist2 carries 20 cane
            # pixels, multan_dist3 carries 685. They are still useful negatives, but
            # separability cannot be measured on them, so they are reported as
            # background-only rather than silently scoring NaN.
            record.update({"cohens_d": np.nan, "status": "background-only"})
        else:
            pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2.0)
            d = float(abs(positive.mean() - negative.mean()) / pooled) if pooled > 0 else np.nan
            record.update({
                "cohens_d": round(d, 3),
                "status": "ok" if d >= MIN_SEPARABILITY_D else "weak",
            })
        rows.append(record)
    return pd.DataFrame(rows).sort_values("cohens_d", na_position="last")


def stage_merge(drop_weak: bool = True) -> None:
    """Stream the per-AOI-date parquets into one table without ever holding it whole.

    The obvious implementation, reading every file into a list and concatenating,
    needs about 6.6 GB for these 135 files: 3.3 GB for the frames and another 3.3 GB
    for the copy pd.concat builds. This machine is a WSL instance with 7.4 GB total,
    of which the editor and its helpers already hold around 1.7 GB, so that version
    triggers the kernel's OOM killer, which picks the largest process and takes the
    session down with it.

    Streaming keeps the peak at one file, about 26 MB. Writing `aoi` and `date` as
    dictionary-encoded columns cuts the result further: as Python strings those two
    account for 63% of the memory while carrying only 44 and 12 distinct values.

    Each source file becomes its own row group, so readers downstream can stream the
    result back the same way.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("nothing to merge; run extract first")

    # Pass one: separability per file, reading only the three columns it needs.
    log.info("checking label transfer across %d AOI-date pairs", len(files))
    checks = [
        label_transfer_check(pd.read_parquet(f, columns=["NDVI", "label", "aoi", "date"]))
        for f in files
    ]
    separability = pd.concat(checks, ignore_index=True).sort_values(
        "cohens_d", na_position="last"
    )
    separability.to_csv(OUT_ROOT / "label_transfer_check.csv", index=False)

    weak = separability[separability.status == "weak"]
    background_only = separability[separability.status == "background-only"]
    log.info("label-transfer check: %d of %d AOI-date pairs below d=%.1f, %d background-only",
             len(weak), len(separability), MIN_SEPARABILITY_D, len(background_only))
    if len(background_only):
        log.info("background-only pairs (kept as negatives, separability not measurable):\n%s",
                 background_only[["aoi", "date", "n_positive", "n_negative"]].to_string(index=False))
    if len(weak):
        log.info("weak pairs:\n%s", weak.head(15).to_string(index=False))

    drop = set(zip(weak.aoi, weak.date)) if drop_weak else set()

    # Pass two: write each kept file straight through as its own row group.
    schema = None
    writer = None
    kept_rows = 0
    per_month: Dict[str, Dict[int, int]] = {}
    aois: set = set()
    dates: set = set()
    try:
        for path in files:
            frame = pd.read_parquet(path)
            pair = (frame.aoi.iloc[0], frame.date.iloc[0])
            if pair in drop:
                continue
            frame["aoi"] = frame.aoi.astype("category")
            frame["date"] = frame.date.astype("category")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(MERGED_PARQUET, schema, compression="snappy")
            writer.write_table(table.cast(schema))

            kept_rows += len(frame)
            aois.add(pair[0])
            dates.add(pair[1])
            month = pair[1][:7]
            counts = frame.label.value_counts().to_dict()
            bucket = per_month.setdefault(month, {})
            for label, n in counts.items():
                bucket[label] = bucket.get(label, 0) + int(n)
            del frame, table
    finally:
        if writer is not None:
            writer.close()

    log.info("merged %d of %d files -> %s", len(files) - len(drop), len(files), MERGED_PARQUET)
    log.info("rows: %d", kept_rows)
    summary = pd.DataFrame(per_month).T.fillna(0).astype(int).sort_index()
    log.info("rows per month:\n%s", summary.to_string())
    log.info("AOIs: %d, dates: %d", len(aois), len(dates))


def stage_verify() -> None:
    """Check the re-extraction against the distribution the deployed model was fitted on.

    This pipeline re-derives the October and November rows from the same imagery the
    original used, but with its own masking and its own outlier caps. If those rows
    no longer match the deployed model's training distribution, something in the
    chain has changed and every downstream comparison is measuring that instead of
    what it claims to.

    Only the October and November rows are compared, because those are the months
    the deployed model actually saw. The August and September rows are new by design
    and are expected to differ.

    A small shift is expected even when nothing is wrong. This extraction caps each
    AOI-date at MAX_PIXELS_PER_CLASS while the original did not, so large districts
    no longer dominate and the pooled median moves toward the smaller AOIs. Read a
    result under about a quarter of an interquartile range as agreement; a larger
    one means something in the chain has genuinely changed.
    """
    import json

    sidecar_path = st_config.CROPSCAN_ROOT / "model_files" / "fao_cane_xgb_model.sidecar.json"
    if not sidecar_path.exists():
        raise SystemExit(f"{sidecar_path} not found")
    reference = json.loads(sidecar_path.read_text())["training_quantiles"]["positive"]

    # Streamed for the same reason as the merge: the whole table does not fit
    # alongside the editor on this machine.
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(MERGED_PARQUET)
    parts = []
    for index in range(handle.metadata.num_row_groups):
        group = handle.read_row_group(index, columns=BAND_ORDER + ["label", "date"]).to_pandas()
        month = str(group.date.iloc[0])[5:7]
        if month in ("10", "11"):
            parts.append(group[group.label == st_config.CANE.crop_label][BAND_ORDER])
        del group
    if not parts:
        raise SystemExit("no October or November cane rows to compare")
    cane = pd.concat(parts, ignore_index=True)
    del parts

    rows = []
    for band in BAND_ORDER:
        observed = cane[band].quantile([0.25, 0.50, 0.75])
        expected = reference.get(band)
        if not expected:
            continue
        spread = expected["q75"] - expected["q25"]
        rows.append({
            "band": band,
            "reextracted_q50": round(float(observed[0.50]), 1),
            "deployed_q50": round(expected["q50"], 1),
            "shift_iqr": round((float(observed[0.50]) - expected["q50"]) / spread, 3) if spread else None,
        })
    table = pd.DataFrame(rows)
    log.info("re-extracted Oct/Nov cane against the deployed model's training distribution:\n%s",
             table.to_string(index=False))
    worst = table.shift_iqr.abs().max()
    log.info("largest shift %.3f IQR -- %s", worst,
             "faithful" if worst < 0.25 else "the extraction has drifted, investigate before training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["scout", "export", "extract", "merge", "verify", "all"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("sentinel").setLevel(logging.WARNING)

    aois = load_aois()
    if args.stage in ("scout", "all"):
        stage_scout(aois, args.workers)
    if args.stage in ("export", "all"):
        stage_export(aois, args.workers)
    if args.stage in ("extract", "all"):
        stage_extract(aois, args.workers)
    if args.stage in ("merge", "all"):
        stage_merge()
    if args.stage in ("verify", "all"):
        stage_verify()


if __name__ == "__main__":
    main()
