"""Run the field labelling over a large AOI, one tile at a time.

The labelling holds the whole delineation in memory: repairing 17,074 polygons over the
test bbox peaked at 4 GB, and the full mill layer is 144,670. Straight through, that is
thirty gigabytes on a seven gigabyte machine.

So it runs per tile in a separate process, which is what actually returns the memory
between tiles rather than merely dropping references. Each tile reads a margin beyond
its own edges so a polygon lying across the boundary is still measured against all of
its pixels, and each polygon is then kept by exactly one tile, the one containing its
representative point. The margin has to exceed the widest polygon in the layer or a
straddling field would be scored on only part of itself; the largest here is about 530 m
across, so a kilometre is enough with room to spare.

    python3 label_field_polygons_tiled.py --crop-map MAP --out DIR --tile-km 12
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
log = logging.getLogger("label_tiled")

#: Wider than the widest polygon in the delineation, so a field straddling a tile edge
#: is still measured against every pixel it covers.
MARGIN_M = 1000.0

SQM_PER_ACRE = 4046.8564224
UTM = 32642


def tiles_for(bounds, tile_km: float) -> List[tuple]:
    """A grid over the raster in metres, returned as lat/lon boxes."""
    from pyproj import Transformer

    to_m = Transformer.from_crs(4326, UTM, always_xy=True)
    to_deg = Transformer.from_crs(UTM, 4326, always_xy=True)
    x0, y0 = to_m.transform(bounds.left, bounds.bottom)
    x1, y1 = to_m.transform(bounds.right, bounds.top)

    step = tile_km * 1000.0
    out = []
    for x in np.arange(x0, x1, step):
        for y in np.arange(y0, y1, step):
            core = (x, y, min(x + step, x1), min(y + step, y1))
            grown = (core[0] - MARGIN_M, core[1] - MARGIN_M,
                     core[2] + MARGIN_M, core[3] + MARGIN_M)
            lo = to_deg.transform(grown[0], grown[1])
            hi = to_deg.transform(grown[2], grown[3])
            out.append((core, (lo[0], lo[1], hi[0], hi[1])))
    return out


def label_one(index, core, wgs, args, scratch, cache_dir, total):
    """Clip the map to one tile, label it in its own process, keep what this tile owns.

    A separate process per tile is what returns the memory; running several at once is
    what uses the machine. The labeller inside is single-threaded and spends its time in
    shapely, so the cores are otherwise idle: sequentially this was 72 seconds a tile,
    fifty minutes a layer, five hours for six layers.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds, Window
    from shapely.geometry import box

    tile_dir = scratch / f"tile_{index:03d}"
    clipped = scratch / f"tile_{index:03d}.tif"
    piece = scratch / f"piece_{index:03d}.parquet"

    # A finished tile is kept, so a run stopped part way does not redo it. The repair
    # cache already survives such a stop; the labelling itself did not, and losing
    # thirty finished tiles to a watchdog firing on the thirty-first is the expensive
    # half. Keyed by tile index under this layer's own directory, so two layers cannot
    # read each other's answers.
    if piece.exists():
        log.info("tile %d/%d already done", index, total)
        return piece

    with rasterio.open(args.crop_map) as src:
        window = from_bounds(*wgs, transform=src.transform)
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            return None
        data = src.read(1, window=window)
        if not (data == 1).any():
            log.info("tile %d/%d holds no crop, skipped", index, total)
            return None
        meta = src.profile.copy()
        meta.update(height=int(window.height), width=int(window.width),
                    transform=rasterio.windows.transform(window, src.transform))
    with rasterio.open(clipped, "w", **meta) as dst:
        dst.write(data, 1)

    log.info("tile %d/%d: labelling", index, total)
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "label_field_polygons.py"),
         "--crop-map", str(clipped), "--out", str(tile_dir),
         "--min-acres", str(args.min_acres), "--threshold", str(args.threshold),
         *(["--delineation", str(args.delineation)] if args.delineation else []),
         "--cache", str(cache_dir / f"tile_{index:03d}.parquet"),
         "--no-gpkg"],
        check=True, stdout=subprocess.DEVNULL)
    clipped.unlink(missing_ok=True)

    produced = tile_dir / "fields_labelled.parquet"
    if not produced.exists():
        return None
    frame = gpd.read_parquet(produced).to_crs(UTM)
    # One tile owns each polygon: the one holding its representative point. Without
    # this every polygon inside a margin would be delivered twice.
    frame = frame[frame.representative_point().within(box(*core))]
    if frame.empty:
        return None
    frame.to_crs(4326).to_parquet(piece)
    return piece


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-km", type=float, default=12.0,
                        help="tile side in kilometres; the test bbox was about 10 km "
                             "and took 85 seconds and 1.5 GB")
    parser.add_argument("--jobs", type=int, default=3,
                        help="tiles laboured on at once. Each is a separate process "
                             "holding one tile's delineation, so this is a memory "
                             "budget; three fits alongside the watchdog's ceiling.")
    parser.add_argument("--no-gpkg", action="store_true",
                        help="write GeoParquet only. GeoPackage takes a minute per "
                             "layer at this size where Parquet takes seconds, so a run "
                             "producing six layers writes them all at the end instead.")
    parser.add_argument("--delineation", type=Path, default=None)
    parser.add_argument("--name", default="fields",
                        help="stem for the delivered files and for the layer inside "
                             "each GeoPackage. QGIS names a layer after the layer in "
                             "the file, not the folder it sits in, so six runs writing "
                             "fields_cane.gpkg open as six layers all called "
                             "fields_cane. Pass the layer's own name instead.")
    parser.add_argument("--min-acres", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where each tile's repaired delineation is kept. Every tile "
                             "has its own, because they are keyed on the window read; a "
                             "single shared cache would be invalidated by each tile in "
                             "turn and never hit. Kept across maps, so only the first of "
                             "the six runs pays for the repair.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.windows import from_bounds
    from shapely.geometry import box

    args.out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.crop_map) as src:
        bounds, profile = src.bounds, src.profile.copy()
    grid = tiles_for(bounds, args.tile_km)
    log.info("%d tiles of %.0f km over the map", len(grid), args.tile_km)

    from label_field_polygons import DELINEATION as _DEFAULT
    delineation = args.delineation or _DEFAULT
    # Each delineation layer gets its own cache: the key includes the source path, so
    # sharing one folder between AOIs would rebuild on every switch.
    cache_dir = args.cache_dir or delineation.parent / "repair_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    scratch = args.out / "_tiles"
    scratch.mkdir(parents=True, exist_ok=True)
    pieces: List[Path] = []

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(label_one, index, core, wgs, args, scratch,
                               cache_dir, len(grid))
                   for index, (core, wgs) in enumerate(grid, start=1)]
        for future in as_completed(futures):
            piece = future.result()
            if piece is not None:
                pieces.append(piece)

    if not pieces:
        raise RuntimeError("no tile produced any polygons")

    log.info("merging %d tiles", len(pieces))
    merged = gpd.GeoDataFrame(
        pd.concat([gpd.read_parquet(p) for p in pieces], ignore_index=True), crs=4326)
    merged["acres"] = merged.to_crs(UTM).area / SQM_PER_ACRE
    merged = merged[merged.acres >= args.min_acres].reset_index(drop=True)

    crop_only = merged[merged.is_crop].copy()
    merged.to_parquet(args.out / f"{args.name}_labelled.parquet")
    crop_only.to_parquet(args.out / f"{args.name}_cane.parquet")
    if not args.no_gpkg:
        merged.to_file(args.out / f"{args.name}_labelled.gpkg", driver="GPKG",
                       layer=f"{args.name}_labelled")
        crop_only.to_file(args.out / f"{args.name}_cane.gpkg", driver="GPKG",
                          layer=f"{args.name}_cane")

    # The per-tile pieces have served their purpose once the layer is merged and
    # written. The repair caches are not touched: those are shared between layers.
    shutil.rmtree(scratch, ignore_errors=True)

    summed = crop_only.acres.sum()
    union = crop_only.to_crs(UTM).geometry.union_all().area / SQM_PER_ACRE
    print(f"\n{len(merged):,} polygons, {merged.acres.sum():,.0f} acres")
    print(f"{len(crop_only):,} cane polygons, {summed:,.0f} acres")
    print(f"double counted: {summed - union:.3f} acres")
    print(f"\noutputs -> {args.out}")


if __name__ == "__main__":
    main()
