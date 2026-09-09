"""Classify the Al-Moiz Unit 1 test feature with the rebuilt model and compare.

Reuses everything that already exists: the time-series RF map and its sieve, and
the 10 and 30 August static images already fetched. Nothing is re-downloaded and
nothing is deleted.

Outputs land beside the existing per-date results, in a `v4/` subfolder, so the old
and new maps sit next to each other for inspection in QGIS.

    python3 run_v4_on_test_feature.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from static_training import inference as st_inference  # noqa: E402

log = logging.getLogger("run_v4")

CROPSCAN = SCRIPTS_DIR.parent
TEST = CROPSCAN / "data" / "test_data" / "almoiz_unit_1_test_feature_1"
CANE = TEST / "cane_2026"
AOI_SHP = TEST / "almoiz_unit_1_test_feature_1.shp"
RF_SIEVED = CANE / "almoiz_unit_1_test_feature_1_rf_classification_map_strict_sieve_multiclass_p20.tif"
RF_RAW = CANE / "almoiz_unit_1_test_feature_1_rf_classification_map.tif"

OLD_MODEL = CROPSCAN / "model_files" / "fao_cane_xgb_model.json"
NEW_MODEL = CROPSCAN / "model_files" / "fao_cane_xgb_model_v4.json"

DATES = [("10_Aug_2026", "static_10m_tile_0001_10_Aug_2026.tif"),
         ("30_Aug_2026", "static_10m_tile_0001_30_Aug_2026.tif")]

CROP_CLASS, BACKGROUND = 1, 4
SIEVE_MIN_PIXELS = 20
MIN_AREA_ACRES = 0.5
UTM = 32642
SQM_PER_ACRE = 4046.8564224


def sieve(path: Path, out: Path, min_size: int = SIEVE_MIN_PIXELS) -> Path:
    """Plain size sieve, applied identically to old and new maps so they compare."""
    import rasterio
    from rasterio.features import sieve as rio_sieve

    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    cleaned = rio_sieve(data, size=min_size, connectivity=4)
    profile.update(compress="lzw", tiled=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(cleaned, 1)
    return out


def vectorise(path: Path, out_gpkg: Path) -> Dict:
    """Polygonise the crop class, clip to the AOI, drop slivers, report acreage."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(path) as src:
        data = src.read(1)
        transform, crs = src.transform, src.crs

    geoms = [
        shape(geom)
        for geom, value in shapes(data, mask=(data == CROP_CLASS), transform=transform)
        if value == CROP_CLASS
    ]
    if not geoms:
        return {"polygons": 0, "acres": 0.0}

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    aoi = gpd.read_file(AOI_SHP, engine="pyogrio").to_crs(crs)
    gdf = gpd.overlay(gdf, aoi[["geometry"]], how="intersection").explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    metric = gdf.to_crs(UTM)
    gdf["area_acres"] = metric.area / SQM_PER_ACRE
    gdf = gdf[gdf.area_acres >= MIN_AREA_ACRES].reset_index(drop=True)
    gdf["predicted"] = CROP_CLASS

    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")
    return {"polygons": len(gdf), "acres": float(gdf.area_acres.sum())}


def pixel_stats(path: Path, cane_mask: np.ndarray, other_mask: np.ndarray) -> Dict:
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1)
    crop = data == CROP_CLASS
    return {
        "cane_px": int(crop.sum()),
        "acres_px": float(crop.sum() * 100 / SQM_PER_ACRE),
        "recall_vs_rf": float(100 * crop[cane_mask].mean()),
        "fp_vs_rf": float(100 * crop[other_mask].mean()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("static_training.inference").setLevel(logging.INFO)

    import rasterio

    with rasterio.open(RF_SIEVED) as src:
        rf_sieved = src.read(1)
    with rasterio.open(RF_RAW) as src:
        rf_raw = src.read(1)
    mask = rf_sieved == CROP_CLASS          # what the static model is allowed to see
    cane_ref = rf_raw == CROP_CLASS         # reference for recall
    other_ref = rf_raw == BACKGROUND
    log.info("RF sieved mask: %d px; RF raw cane %d px, non-cane %d px",
             mask.sum(), cane_ref.sum(), other_ref.sum())

    rows: List[Dict] = []
    verdicts: Dict[str, str] = {}

    for folder, tile in DATES:
        date_dir = CANE / folder
        image = date_dir / tile
        out_dir = date_dir / "v4"
        out_dir.mkdir(parents=True, exist_ok=True)

        cls_path = out_dir / f"static_mosaic_{folder}_Cls_v4.tif"
        prob_path = out_dir / f"static_mosaic_{folder}_prob_v4.tif"

        log.info("=== %s", folder)
        result = st_inference.classify_raster(
            image, NEW_MODEL, cls_path, prob_path,
            mask_array=mask, positive_out=CROP_CLASS, background_out=BACKGROUND,
            enforce_domain=False,
        )
        verdicts[folder] = str(result.verdict)
        log.info("threshold %.2f, %s", result.threshold, result.verdict)

        new_sieved = sieve(cls_path, out_dir / f"static_mosaic_{folder}_Cls_v4_sieved_p20.tif")
        old_raw = date_dir / f"static_mosaic_{folder.replace('_2026', '_2026')}_Cls.tif"
        old_sieved_same = sieve(old_raw, out_dir / f"OLD_{folder}_Cls_sieved_p20_samefilter.tif")

        for label, raw, sieved in [("deployed", old_raw, old_sieved_same),
                                   ("v4", cls_path, new_sieved)]:
            stats = pixel_stats(raw, cane_ref, other_ref)
            sv = pixel_stats(sieved, cane_ref, other_ref)
            gpkg = out_dir / f"{label}_{folder}_cane.gpkg"
            vec = vectorise(sieved, gpkg)
            rows.append({
                "date": folder.replace("_2026", "").replace("_", " "),
                "model": label,
                "cane_px_raw": stats["cane_px"],
                "cane_px_sieved": sv["cane_px"],
                "recall_vs_rf_%": round(sv["recall_vs_rf"], 1),
                "fp_vs_rf_%": round(sv["fp_vs_rf"], 1),
                "polygons": vec["polygons"],
                "acres": round(vec["acres"], 0),
                "gpkg": str(gpkg),
            })
            log.info("  %-9s sieved %7d px  recall %.1f%%  fp %.1f%%  %d polys  %.0f acres",
                     label, sv["cane_px"], sv["recall_vs_rf"], sv["fp_vs_rf"],
                     vec["polygons"], vec["acres"])

    table = pd.DataFrame(rows)
    out_csv = CANE / "v4_vs_deployed_comparison.csv"
    table.drop(columns=["gpkg"]).to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("ALL OUTPUTS ARE NEW FILES. Nothing existing was overwritten or deleted.")
    print("=" * 78)
    print(table.drop(columns=["gpkg"]).to_string(index=False))

    print("\nAcreage change, sieved and clipped, 0.5-acre floor applied to both:")
    for date in table.date.unique():
        sub = table[table.date == date].set_index("model")
        old, new = sub.loc["deployed", "acres"], sub.loc["v4", "acres"]
        print(f"  {date:10s}  deployed {old:>7,.0f}  ->  v4 {new:>7,.0f} acres "
              f"({new - old:+,.0f}, {100 * (new - old) / old:+.1f}%)")

    print("\nDomain guard:")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")

    print(f"\ncomparison table -> {out_csv}")
    print(f"rasters and polygons -> {CANE}/<date>/v4/")


if __name__ == "__main__":
    main()
