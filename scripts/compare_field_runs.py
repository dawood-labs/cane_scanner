"""Compare two field-labelling runs, at the raster and at the field.

Raster acreage is the easy number and the least useful one. What a mill acts on is
fields: which ones changed hands between the two runs, and where the new acreage came
from. Both runs label the same delineation, so `source_fid` lines them up exactly and
the comparison needs no spatial join.

    python3 compare_field_runs.py
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
CROPSCAN = SCRIPTS_DIR.parent
CANE = CROPSCAN / "data" / "test_data" / "almoiz_unit_1_test_feature_1" / "cane_2026"

BASE = CANE / "field_labelling"                 # 30 August, 0.5-acre sieve
NEW = CANE / "finer_p6" / "field_labelling"     # both dates unioned, 0.15-acre sieve

BASE_RASTER = CANE / "30_Aug_2026" / "v4" / "static_mosaic_30_Aug_2026_Cls_v4_sieved_p20.tif"
NEW_RASTER = CANE / "finer_p6" / "static_union_Cls_v4_strict_sieve_multiclass_p6.tif"

A = 4046.8564224
UTM = 32642
log = logging.getLogger("compare_field_runs")


def load(folder: Path):
    import geopandas as gpd

    frame = gpd.read_parquet(folder / "fields_labelled.parquet")
    frame["ac"] = frame.to_crs(UTM).area / A
    return frame


def raster_summary() -> None:
    import rasterio

    rows = []
    for name, path in [("30 Aug, 0.5-acre sieve", BASE_RASTER),
                       ("both dates, 0.15-acre sieve", NEW_RASTER)]:
        with rasterio.open(path) as src:
            band = src.read(1)
        rows.append({"map": name,
                     "cane_acres": round(float((band == 1).sum()) * 100 / A),
                     "classified_acres": round(float((band != 255).sum()) * 100 / A)})
    print("\nTHE CLASSIFICATION")
    print(pd.DataFrame(rows).to_string(index=False))


def field_summary(base, new) -> None:
    print("\nTHE POLYGONS")
    rows = []
    for name, frame in [("30 Aug, 0.5-acre sieve", base),
                        ("both dates, 0.15-acre sieve", new)]:
        crop = frame[frame.is_crop]
        rows.append({"run": name, "polygons": len(frame), "acres": round(frame.ac.sum()),
                     "cane_polygons": len(crop), "cane_acres": round(crop.ac.sum())})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nWHERE THE CANE POLYGONS COME FROM")
    table = pd.DataFrame({
        "30 Aug": base[base.is_crop].groupby("origin").ac.sum().round(0),
        "both dates": new[new.is_crop].groupby("origin").ac.sum().round(0),
    }).fillna(0)
    table["change"] = table["both dates"] - table["30 Aug"]
    print(table.to_string())


def field_flips(base, new) -> None:
    """Which of the client's own polygons changed hands, and by how much."""
    traced = lambda f: (f[f.source_fid > 0]
                        .groupby("source_fid")
                        .agg(cane=("is_crop", "max"), ac=("ac", "sum")))
    a, b = traced(base), traced(new)
    both = a.join(b, how="outer", lsuffix="_old", rsuffix="_new")
    both[["cane_old", "cane_new"]] = both[["cane_old", "cane_new"]].fillna(False)

    gained = both[~both.cane_old & both.cane_new]
    lost = both[both.cane_old & ~both.cane_new]
    kept = both[both.cane_old & both.cane_new]

    print("\nTRACED POLYGONS THAT CHANGED HANDS")
    print(f"  cane in both runs        {len(kept):>7,} polygons  {kept.ac_new.sum():>9,.0f} acres")
    print(f"  became cane              {len(gained):>7,} polygons  {gained.ac_new.sum():>9,.0f} acres")
    print(f"  stopped being cane       {len(lost):>7,} polygons  {lost.ac_old.sum():>9,.0f} acres")

    if not gained.empty:
        print("\n  the ten biggest fields that became cane:")
        top = gained.nlargest(10, "ac_new")[["ac_new"]].round(2)
        top.columns = ["acres"]
        print(top.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    base, new = load(BASE), load(NEW)
    print("=" * 74)
    raster_summary()
    field_summary(base, new)
    field_flips(base, new)
    print("\n" + "=" * 74)
    captured = lambda f, r: 100 * f[f.is_crop].ac.sum() / r
    print(f"cane reaching a cane polygon: {captured(base, 7237):.1f}% before, "
          f"{captured(new, 8557):.1f}% after")


if __name__ == "__main__":
    main()
