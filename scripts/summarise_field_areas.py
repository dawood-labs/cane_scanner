"""Acreage on the delivered field polygons, per layer and by where the geometry came from.

The raster acreage answers "how much cane did the model see". This answers the question
a client actually asks: how much cane is in the polygons we are handing over, and how
much of that sits on boundaries they drew rather than boundaries we invented.

That second number is the one worth watching. On the Al-Moiz AOI the static and fused
layers put about 88% of their acreage on traced geometry, whole or cut, while the
time-series layer manages 74%: its map is looser, so more of its cane falls outside the
delineation and needs a polygon derived from the raster instead. That is a reason to
deliver a static or fused layer beyond the acreage itself.

    python3 summarise_field_areas.py
    python3 summarise_field_areas.py --outputs PATH
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
CROPSCAN = SCRIPTS_DIR.parent
DEFAULT_OUTPUTS = (CROPSCAN / "data" / "Al-Moiz-Unit-1-SM-AOI-2025" / "cane_2026"
                   / "outputs")

SQM_PER_ACRE = 4046.8564224
UTM = 32642


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS,
                        help="the folder holding one directory per delivered layer")
    args = parser.parse_args()

    import geopandas as gpd

    layers = sorted(d.name for d in args.outputs.iterdir()
                    if d.is_dir() and (d / "fields_cane.parquet").exists())
    if not layers:
        raise SystemExit(f"no layers with fields_cane.parquet under {args.outputs}")

    rows, origins = [], {}
    for name in layers:
        frame = gpd.read_parquet(args.outputs / name / "fields_cane.parquet").to_crs(UTM)
        frame["ac"] = frame.area / SQM_PER_ACRE
        rows.append({"layer": name, "polygons": len(frame),
                     "acres": round(frame.ac.sum()),
                     "median_ac": round(frame.ac.median(), 2),
                     "p95_ac": round(frame.ac.quantile(0.95), 2),
                     "largest_ac": round(frame.ac.max(), 1),
                     "under_1ac": int((frame.ac < 1).sum())})
        origins[name] = frame.groupby("origin").ac.agg(["size", "sum"]).round(0)

    print("CANE ON THE DELIVERED FIELD POLYGONS")
    print(pd.DataFrame(rows).to_string(index=False))

    acres = pd.DataFrame({n: origins[n]["sum"] for n in layers}).fillna(0).astype(int)
    print("\nWHERE THE GEOMETRY CAME FROM, acres")
    print(acres.to_string())
    print("\nsame, as a share of each layer")
    print((100 * acres / acres.sum()).round(1).to_string())

    # Traced geometry is what the delineation drew, whole or cut. The rest is ground the
    # delineation had no boundary for, and its edges come from the 10 m raster.
    traced = [i for i in acres.index if i in ("delineation", "split")]
    share = 100 * acres.loc[traced].sum() / acres.sum()
    print("\nshare of each layer's acreage sitting on traced boundaries")
    print(share.round(1).to_string())

    print("\nPOLYGON COUNT BY ORIGIN")
    print(pd.DataFrame({n: origins[n]["size"] for n in layers}).fillna(0)
          .astype(int).to_string())

    acres.to_csv(args.outputs / "area_by_origin.csv")
    pd.DataFrame(rows).to_csv(args.outputs / "area_by_layer.csv", index=False)
    print(f"\nwritten -> {args.outputs / 'area_by_layer.csv'}")


if __name__ == "__main__":
    main()
