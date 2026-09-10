"""Can Sentinel-1 radar separate sugarcane from orchards where NDVI cannot?

A fast, decisive probe, not an integration. The time-series NDVI RandomForest maps
mango and citrus blocks as cane. A second model reading the same NDVI cannot undo
that -- pooled leave-one-district-out AUC 0.681 -- because the orchard ground that
fools the first model is by construction the ground whose NDVI trace looked like
cane. Radar measures structure rather than greenness, so it reads a different
signal: a woody canopy scatters volume strongly and steadily all year, a grass
crop's backscatter tracks the crop cycle and collapses at harvest.

Labels, exactly as the optical work defined them:
    positive = inside an orchard polygon AND class 1 in the rf_cane raster
    negative = class 1 in rf_cane AND outside every orchard polygon
Both classes live inside what the time-series model already called cane, which is
the only place the confusion exists.

Stages:
    fetch     Sentinel-1 RTC from Microsoft Planetary Computer, one orbit direction
              and one relative orbit for every district so the geometry is constant.
              Composited in linear gamma0 power, then written as dB.
    objects   Connected patches of >=50 pixels within each class, aggregated to
              per-object radar features and NDVI phenology features.
    evaluate  Cohen's d per feature, then a small XGBoost scored as pooled
              leave-one-district-out ROC-AUC on radar, NDVI, and the two together.

Usage:
    python sar_probe.py fetch [--districts A,B]
    python sar_probe.py objects [--districts A,B]
    python sar_probe.py evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/c/Work_Work_Work/Python/Scripts/FAO/cane/orchard_detector")
SAR_DIR = ROOT / "sar"
CHIPS = ROOT / "chips.gpkg"
ORCHARDS = ROOT / "orchard_polygons.gpkg"
RF_DIR = ROOT / "rf_stage"
STACK_DIR = ROOT / "stacks"

START = "2025-11-24T00:00:00Z"
END = "2026-09-09T00:00:00Z"
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"

#: One orbit direction and one track for every chip. Ascending and descending
#: passes see a field from opposite sides and cannot be averaged; descending
#: relative orbit 5 is the single track that covers all five chips, 22 dates.
ORBIT_STATE = "descending"
RELATIVE_ORBIT = 5

#: Rows per streamed block. The chips are ~941x820, so a block of 22 dates x 2
#: polarisations costs about 40 MB -- a full district stack never exists at once.
BLOCK_ROWS = 256

#: A patch smaller than this is noise, not a field.
MIN_OBJECT_PIXELS = 50

SAR_FEATURES = ("vv_med", "vh_med", "ratio_med", "vv_std", "vh_std", "ratio_std")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_training import phenology  # noqa: E402  (same NDVI features as the optical run)

NDVI_FEATURES = tuple(phenology.FEATURE_NAMES)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def districts_on_disk() -> list[str]:
    import geopandas as gpd
    return list(gpd.read_file(CHIPS)["district"])


def _slug(name: str) -> str:
    return name.replace(" ", "_")


# ---------------------------------------------------------------- fetch ----

def _geobox_of(path: Path):
    import rasterio
    from odc.geo.geobox import GeoBox
    with rasterio.open(path) as src:
        return GeoBox((src.height, src.width), src.transform, src.crs), src.profile


def _search_items(catalog, bbox):
    search = catalog.search(collections=[COLLECTION], bbox=list(bbox),
                            datetime=f"{START}/{END}")
    items = [it for it in search.items()
             if it.properties.get("sat:orbit_state") == ORBIT_STATE
             and it.properties.get("sat:relative_orbit") == RELATIVE_ORBIT]
    return items


def _to_db(power: np.ndarray) -> np.ndarray:
    """Linear gamma0 power to dB. Values are negative; never clip at zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (10.0 * np.log10(np.where(power > 0, power, np.nan))).astype("float32")


def fetch(districts: list[str]) -> None:
    import geopandas as gpd
    import planetary_computer
    import pystac_client
    import rasterio
    from odc.stac import load as odc_load

    SAR_DIR.mkdir(parents=True, exist_ok=True)
    chips = gpd.read_file(CHIPS).set_index("district")
    catalog = pystac_client.Client.open(MPC_STAC, modifier=planetary_computer.sign_inplace)

    for district in districts:
        out_path = SAR_DIR / f"{_slug(district)}_sar.tif"
        if out_path.exists():
            log(f"{district}: {out_path.name} exists, skipping")
            continue
        geom = chips.loc[district, "geometry"]
        geobox, profile = _geobox_of(RF_DIR / f"{_slug(district)}_rf_cane.tif")
        items = _search_items(catalog, geom.bounds)
        if len(items) < 8:
            log(f"{district}: only {len(items)} items on orbit {RELATIVE_ORBIT}, skipping")
            continue
        log(f"{district}: {len(items)} S1 items, {ORBIT_STATE} orbit {RELATIVE_ORBIT}, "
            f"grid {geobox.shape}")

        ny, nx = geobox.shape
        out = np.full((len(SAR_FEATURES), ny, nx), np.nan, dtype="float32")
        n_dates = 0
        for r0 in range(0, ny, BLOCK_ROWS):
            r1 = min(r0 + BLOCK_ROWS, ny)
            sub = geobox[r0:r1, 0:nx]
            ds = odc_load(items, bands=["vv", "vh"], geobox=sub, resampling="bilinear",
                          groupby="solar_day", dtype="float32", chunks=None)
            vv = np.asarray(ds["vv"].values, dtype="float32")   # (T, rows, nx) linear power
            vh = np.asarray(ds["vh"].values, dtype="float32")
            n_dates = vv.shape[0]
            vv = np.where(vv > 0, vv, np.nan)
            vh = np.where(vh > 0, vh, np.nan)

            with np.errstate(invalid="ignore"):
                # Composite in linear power; a median of dB is not the median of power.
                vv_med = _to_db(np.nanmedian(vv, axis=0))
                vh_med = _to_db(np.nanmedian(vh, axis=0))
            vv_db = _to_db(vv)
            vh_db = _to_db(vh)
            ratio_db = vh_db - vv_db          # VH/VV in power == a difference in dB
            with np.errstate(invalid="ignore"):
                ratio_med = np.nanmedian(ratio_db, axis=0).astype("float32")
                vv_std = np.nanstd(vv_db, axis=0).astype("float32")
                vh_std = np.nanstd(vh_db, axis=0).astype("float32")
                ratio_std = np.nanstd(ratio_db, axis=0).astype("float32")

            for i, arr in enumerate((vv_med, vh_med, ratio_med, vv_std, vh_std, ratio_std)):
                out[i, r0:r1, :] = arr
            del ds, vv, vh, vv_db, vh_db, ratio_db
            log(f"  rows {r0}-{r1} done ({n_dates} dates)")

        profile.update(dtype="float32", count=len(SAR_FEATURES), nodata=np.nan,
                       compress="lzw", tiled=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)
            for i, name in enumerate(SAR_FEATURES, start=1):
                dst.set_band_description(i, name)
        finite = np.isfinite(out[0])
        log(f"{district}: wrote {out_path.name}, {100 * finite.mean():.1f}% valid, "
            f"VV median {np.nanmedian(out[0]):.2f} dB, VH median {np.nanmedian(out[1]):.2f} dB")
        (SAR_DIR / f"{_slug(district)}_sar.json").write_text(json.dumps(
            {"district": district, "n_items": len(items), "n_dates": int(n_dates),
             "orbit_state": ORBIT_STATE, "relative_orbit": RELATIVE_ORBIT,
             "start": START, "end": END,
             "dates": sorted({it.properties["datetime"][:10] for it in items})}, indent=2))
        del out


# -------------------------------------------------------------- objects ----

def _object_labels(district: str):
    """(labels, is_orchard, shape) -- one id per connected patch of >=50 pixels.

    Positives and negatives are labelled separately so a patch never straddles the
    orchard boundary.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from scipy import ndimage

    with rasterio.open(RF_DIR / f"{_slug(district)}_rf_cane.tif") as src:
        rf = src.read(1)
        transform, shape, crs = src.transform, (src.height, src.width), src.crs

    orch = gpd.read_file(ORCHARDS)
    orch = orch[orch["district"] == district]
    if orch.crs != crs:
        orch = orch.to_crs(crs)
    orch_mask = np.zeros(shape, dtype="uint8")
    if len(orch):
        orch_mask = rasterize(((g, 1) for g in orch.geometry), out_shape=shape,
                              transform=transform, fill=0, dtype="uint8")

    cane = rf == 1
    pos = cane & (orch_mask == 1)
    neg = cane & (orch_mask == 0)

    labels = np.zeros(shape, dtype="int32")
    is_orchard: list[int] = [0]          # index 0 is the "no object" slot
    next_id = 1
    for mask, flag in ((pos, 1), (neg, 0)):
        comp, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = np.bincount(comp.ravel())
        keep = np.where(sizes >= MIN_OBJECT_PIXELS)[0]
        keep = keep[keep > 0]
        remap = np.zeros(n + 1, dtype="int32")
        remap[keep] = np.arange(next_id, next_id + len(keep), dtype="int32")
        labels += remap[comp]
        is_orchard.extend([flag] * len(keep))
        next_id += len(keep)
    return labels, np.asarray(is_orchard[1:], dtype="int8"), shape


def _mean_per_object(labels: np.ndarray, band: np.ndarray, n_obj: int) -> np.ndarray:
    """nanmean of `band` inside every object id, streamed with bincount."""
    flat_lab = labels.ravel()
    flat_val = band.ravel()
    ok = (flat_lab > 0) & np.isfinite(flat_val)
    sums = np.bincount(flat_lab[ok], weights=flat_val[ok], minlength=n_obj + 1)[1:]
    counts = np.bincount(flat_lab[ok], minlength=n_obj + 1)[1:]
    return np.divide(sums, counts, out=np.full(n_obj, np.nan), where=counts > 0)


def objects(districts: list[str]) -> None:
    import pandas as pd
    import rasterio

    SAR_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for district in districts:
        sar_path = SAR_DIR / f"{_slug(district)}_sar.tif"
        if not sar_path.exists():
            log(f"{district}: no radar yet, skipping")
            continue
        labels, is_orchard, shape = _object_labels(district)
        n_obj = len(is_orchard)
        if n_obj == 0:
            log(f"{district}: no objects >= {MIN_OBJECT_PIXELS} px")
            continue
        sizes = np.bincount(labels.ravel(), minlength=n_obj + 1)[1:]

        cols = {"district": district, "is_orchard": is_orchard, "pixels": sizes}

        with rasterio.open(sar_path) as src:
            for i, name in enumerate(SAR_FEATURES, start=1):
                cols[name] = _mean_per_object(labels, src.read(i), n_obj)

        # NDVI phenology on the object-mean series: one band at a time, so a full
        # 37-date stack is never resident.
        with rasterio.open(STACK_DIR / f"{_slug(district)}_smoothed.tif") as src:
            series = np.empty((src.count, n_obj), dtype="float32")
            for b in range(1, src.count + 1):
                series[b - 1] = _mean_per_object(labels, src.read(b), n_obj)
        mat = phenology.build_matrix(series, NDVI_FEATURES)
        for j, name in enumerate(NDVI_FEATURES):
            cols[name] = mat[:, j]

        frame = pd.DataFrame(cols)
        frames.append(frame)
        log(f"{district}: {n_obj} objects "
            f"({int(is_orchard.sum())} orchard, {int((is_orchard == 0).sum())} cane), "
            f"median size {np.median(sizes):.0f} px")

    if not frames:
        log("no objects built")
        return
    out = pd.concat(frames, ignore_index=True)
    path = SAR_DIR / "objects.parquet"
    out.to_parquet(path, index=False)
    log(f"wrote {path} -- {len(out)} objects, {int(out.is_orchard.sum())} orchard")


# ------------------------------------------------------------- evaluate ----

def _cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                 / (len(pos) + len(neg) - 2))
    return float(abs(pos.mean() - neg.mean()) / sp) if sp > 0 else float("nan")


def _lodo_auc(frame, feats: list[str]):
    """Pooled leave-one-district-out AUC: out-of-fold scores concatenated, one AUC."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    y_all, p_all, per_district = [], [], {}
    for held in sorted(frame["district"].unique()):
        train = frame[frame["district"] != held]
        test = frame[frame["district"] == held]
        if test["is_orchard"].nunique() < 2 or train["is_orchard"].nunique() < 2:
            per_district[held] = float("nan")
            continue
        model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.06,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              eval_metric="logloss", n_jobs=2, verbosity=0)
        model.fit(train[feats].to_numpy("float32"), train["is_orchard"].to_numpy())
        prob = model.predict_proba(test[feats].to_numpy("float32"))[:, 1]
        y = test["is_orchard"].to_numpy()
        per_district[held] = float(roc_auc_score(y, prob))
        y_all.append(y)
        p_all.append(prob)
    if not y_all:
        return float("nan"), per_district
    return float(roc_auc_score(np.concatenate(y_all), np.concatenate(p_all))), per_district


def evaluate() -> None:
    import pandas as pd

    frame = pd.read_parquet(SAR_DIR / "objects.parquet")

    # A chip on the edge of the orbit swath has objects with no radar at all
    # (Khanewal sits at 44% coverage on descending track 5). Scoring those from an
    # all-NaN row is not a radar result, it is a gap, and pooling their scores with
    # everyone else's moves the ranking for reasons that have nothing to do with
    # backscatter. Drop them; the district still trains the model with what it has.
    covered = np.isfinite(frame[list(SAR_FEATURES)].to_numpy("float32")).all(axis=1)
    if (~covered).any():
        dropped = frame.loc[~covered, "district"].value_counts().to_dict()
        log(f"dropped {int((~covered).sum())} objects outside the radar swath: {dropped}")
        frame = frame.loc[covered].reset_index(drop=True)

    log(f"{len(frame)} objects across {frame.district.nunique()} districts: "
        f"{int(frame.is_orchard.sum())} orchard, {int((frame.is_orchard == 0).sum())} cane")
    print(frame.groupby("district")["is_orchard"].agg(["size", "sum"])
          .rename(columns={"size": "objects", "sum": "orchard"}))

    pos = frame[frame.is_orchard == 1]
    neg = frame[frame.is_orchard == 0]
    rows = []
    for name in list(SAR_FEATURES) + list(NDVI_FEATURES):
        rows.append({"feature": name,
                     "kind": "radar" if name in SAR_FEATURES else "ndvi",
                     "cohens_d": round(_cohens_d(pos[name].to_numpy(), neg[name].to_numpy()), 3),
                     "cane_median": round(float(np.nanmedian(neg[name])), 3),
                     "orchard_median": round(float(np.nanmedian(pos[name])), 3)})
    sep = pd.DataFrame(rows).sort_values("cohens_d", ascending=False)
    print("\nobject-level separability, orchard against cane:")
    print(sep.to_string(index=False))
    sep.to_csv(SAR_DIR / "separability_radar.csv", index=False)

    # Absolute gamma0 shifts between chips with incidence angle and soil moisture,
    # which is exactly what stops a feature transferring to an unseen district.
    # Subtracting each chip's own median is a scene-level normalisation that needs
    # no labels, so it is available at inference time too.
    rel_names = []
    for name in SAR_FEATURES:
        rel = f"{name}_rel"
        frame[rel] = frame[name] - frame.groupby("district")[name].transform("median")
        rel_names.append(rel)

    sets = {"radar": list(SAR_FEATURES),
            "radar_rel": rel_names,
            "ndvi": list(NDVI_FEATURES),
            "radar+ndvi": list(SAR_FEATURES) + list(NDVI_FEATURES),
            "radar_rel+ndvi": rel_names + list(NDVI_FEATURES)}
    results = []
    print("\npooled leave-one-district-out ROC-AUC:")
    for name, feats in sets.items():
        pooled, per = _lodo_auc(frame, feats)
        results.append({"feature_set": name, "n_features": len(feats),
                        "pooled_auc": round(pooled, 4),
                        **{d: round(v, 4) for d, v in per.items()}})
        print(f"  {name:<12} pooled {pooled:.4f}   " +
              "  ".join(f"{d}={v:.3f}" for d, v in per.items()))
    out = pd.DataFrame(results)
    out.to_csv(SAR_DIR / "lodo_radar.csv", index=False)
    log(f"wrote {SAR_DIR / 'lodo_radar.csv'}")

    baseline = 0.681
    print("\nVERDICT vs the 0.681 NDVI-phenology baseline "
          "(and vs this run's own NDVI number, same objects):")
    own = out.loc[out.feature_set == "ndvi", "pooled_auc"].iloc[0]
    for name in sets:
        auc = out.loc[out.feature_set == name, "pooled_auc"].iloc[0]
        print(f"  {name:<15} {auc:.3f}   vs 0.681 {auc - baseline:+.3f}   "
              f"vs this run's NDVI {auc - own:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "objects", "evaluate"])
    ap.add_argument("--districts", default="")
    args = ap.parse_args()
    picked = [d.strip() for d in args.districts.split(",") if d.strip()] or districts_on_disk()
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    if args.stage == "fetch":
        fetch(picked)
    elif args.stage == "objects":
        objects(picked)
    else:
        evaluate()


if __name__ == "__main__":
    main()
