"""The time-series stage, as a module instead of a notebook cell.

Lifted out of `Model_Execution_Pipeline_v3.0.ipynb` cell 5 unchanged, so a script and
the notebook cannot drift apart. Tiles the AOI at `tile_deg`, fetches Sentinel-2 per
tile, smooths, runs the RandomForest, and mosaics the result. Raw tiles are kept when
`delete_raw_tiles=False`, and a tile already on disk is not fetched again.
"""

import os
import sys
import shutil
import logging
import multiprocessing
import ctypes
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.merge import merge
from osgeo import gdal
from tqdm import tqdm  # Reverted to standard tqdm for strict stdout control
import joblib
import gc

from geo_inference_workers import worker_process_local_tile
from sentinel import fetch_sentinel_imagery
# ==============================================================================
# FORCE LOGGING VISIBILITY IN JUPYTER
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger("PipelineOrchestrator")

# Prevent multiprocessing deadlocks
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['GDAL_NUM_THREADS'] = '1'

# Enable GDAL exceptions but suppress harmless PROJ/C-level warnings
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

def release_os_memory() -> None:
    """Forces Python and the OS (glibc) to release unreferenced memory."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError:
        pass

def burn_aoi_into_tiles(tiles, aoi_path, block: int = 2048) -> int:
    """Zero every band outside the AOI, so the inference never works on that ground.

    The fetch already drops tiles that miss the AOI entirely (`sentinel._build_tiles`),
    so what is left is the overhang: the part of a 0.1-degree cell that sticks out past
    the polygon. On the Al-Moiz mill that is 2,513 km2 processed for a 1,044 km2 AOI.

    Zeroing is the right way to say "nothing here", not an invention: the raw tiles carry
    no declared nodata and encode missing as 0 in every band, and the worker re-derives
    validity from `nir + red > 0`. So a zeroed pixel arrives at the worker as NaN NDVI in
    every timestep, which drops it out of both expensive steps for free -- the Whittaker
    solve skips pixels missing in all bands, and `predict` is only called on what
    survives. No change to the worker, no change to the profile.

    What it does not save is the block read, the NDVI divide and the write. The gain is
    the linear algebra and the model, which is most of the stage but not all of it.

    Windowed, because a tile is 37 bands and holding one whole is pointless when the
    mask is built per window anyway. A clipped tile is marked with a sidecar so a resumed
    run does not clip what it already clipped -- clipping twice is harmless but reading
    3 GB to discover that is not.
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, transform as window_transform

    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"{aoi_path} holds no geometry to clip to")

    done = 0
    for path in tiles:
        marker = Path(path).with_suffix(".aoiclip")
        if marker.exists():
            continue

        with rasterio.open(path, "r+") as src:
            shapes = aoi.to_crs(src.crs).geometry.values
            kept = dropped = 0
            for y in range(0, src.height, block):
                for x in range(0, src.width, block):
                    win = Window(x, y, min(block, src.width - x), min(block, src.height - y))
                    inside = geometry_mask(
                        shapes, out_shape=(int(win.height), int(win.width)),
                        transform=window_transform(win, src.transform),
                        invert=True, all_touched=True)
                    if inside.all():
                        kept += inside.size
                        continue
                    data = src.read(window=win)
                    data[:, ~inside] = 0
                    src.write(data, window=win)
                    kept += int(inside.sum())
                    dropped += int((~inside).sum())

        marker.write_text(f"cleared {dropped} pixels outside the AOI, kept {kept}\n")
        logger.info(f"  {Path(path).name}: {100 * dropped / max(kept + dropped, 1):.0f}% "
                    f"of the tile lay outside the AOI")
        done += 1
    return done


def stack_as_vrt(input_paths: List[Path], output_path: Path) -> Path:
    """Point a VRT at the per-tile chunks instead of merging them into one raster.

    `mosaic_continuous_rasters` reads every input into one array and then holds that
    same array while GTiff compresses it, so it peaks twice. For the smoothed NDVI stack
    over a mill AOI that array is 7,791 x 7,791 x 36 bands of float32, about 8.7 GB, and
    on a 7 GB machine it took the whole session down with it.

    A VRT is a few kilobytes of XML naming the chunks and where they sit. GDAL and
    rasterio open it exactly like a GeoTIFF and read it windowed, which is how the only
    consumer, `orchard_filter.score_pixels`, reads it anyway. The chunks are written
    block by block upstream, so the chain becomes streaming end to end and nothing ever
    holds the whole AOI.

    The chunks are the VRT's pixels, not a temporary: whatever cleans up afterwards has
    to leave them alone.
    """
    from osgeo import gdal

    output_path = Path(output_path).with_suffix(".vrt")
    logger.info(f"Building a VRT over {len(input_paths)} chunks: {output_path.name}")
    vrt = gdal.BuildVRT(str(output_path), [str(p) for p in input_paths],
                        options=gdal.BuildVRTOptions(resampleAlg="nearest"))
    if vrt is None:
        raise RuntimeError(f"gdal.BuildVRT produced nothing for {output_path}")
    # BuildVRT does not carry band descriptions across, and losing them has broken this
    # pipeline before: a mosaic without them made the static stage report no indexes to
    # read. The dates are the only thing naming which timestep a band is.
    with rasterio.open(input_paths[0]) as first:
        names = first.descriptions
    for index, name in enumerate(names, start=1):
        if name:
            vrt.GetRasterBand(index).SetDescription(name)
    vrt.FlushCache()
    del vrt

    with rasterio.open(output_path) as check:
        logger.info(f"VRT: {check.width} x {check.height}, {check.count} bands, "
                    f"{check.dtypes[0]}, first band {check.descriptions[0]}")
    return output_path


def mosaic_continuous_rasters(input_paths: List[Path], output_path: Path) -> None:
    logger.info(f"\nMosaicing {len(input_paths)} continuous rasters using rasterio merge...")
    src_files = [rasterio.open(fp) for fp in input_paths]
    try:
        template_src = src_files[0]
        nodata_val = template_src.nodata
        if nodata_val is None:
            is_float = np.issubdtype(template_src.dtypes[0], np.floating)
            nodata_val = np.nan if is_float else 0
            
        mosaic, out_trans = merge(src_files, res=template_src.res, nodata=nodata_val, method='first')
        
        out_meta = template_src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2],
            "transform": out_trans, "nodata": nodata_val, "compress": "lzw",
            "tiled": True, "blockxsize": 256, "blockysize": 256,
            "bigtiff": "YES"
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
            dest.update_tags(**template_src.tags())
            for i, desc in enumerate(template_src.descriptions, start=1):
                if desc: dest.set_band_description(i, desc)
        logger.info(f"Successfully generated mosaic: {output_path}")
    finally:
        for src in src_files: src.close()
        release_os_memory()

def mosaic_categorical_rasters(input_paths: List[Path], output_path: Path, band_name: str = "Classification") -> None:
    logger.info(f"\nMosaicing {len(input_paths)} categorical chunks ({band_name}) using rasterio merge...")
    src_files = [rasterio.open(fp) for fp in input_paths]
    try:
        template_src = src_files[0]
        # `or 255` would turn a legitimate nodata of 0 into 255.
        nodata_val = 255 if template_src.nodata is None else template_src.nodata
            
        mosaic, out_trans = merge(src_files, res=template_src.res, nodata=nodata_val, method='first')
        
        out_meta = template_src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2],
            "transform": out_trans, "nodata": nodata_val, "compress": "lzw",
            "tiled": True, "blockxsize": 256, "blockysize": 256,
            "bigtiff": "YES"
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
            dest.update_tags(**template_src.tags())
            dest.set_band_description(1, band_name)
                    
        logger.info(f"Successfully generated final categorical map: {output_path}")
    finally:
        for src in src_files: src.close()
        release_os_memory()

def generate_global_index_mask(reference_path: Path, out_path: Path, nodata_val: int = -1) -> None:
    """Already memory-safe (O(1) block-windowed reading)."""
    with rasterio.open(reference_path) as src:
        meta = src.profile.copy()
        W = src.width
        ref_nodata = src.nodata if src.nodata is not None else 255
        meta.update(dtype=rasterio.int32, count=1, nodata=nodata_val, compress="lzw", tiled=True, blockxsize=256, blockysize=256, bigtiff="YES")

        with rasterio.open(out_path, "w", **meta) as dst:
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)
                valid_mask = (data != ref_nodata)
                
                rows, cols = np.indices((window.height, window.width))
                global_rows = rows + window.row_off
                global_cols = cols + window.col_off
                global_1d_indices = (global_rows * W + global_cols).astype(np.int32)
                
                index_block = np.full(data.shape, nodata_val, dtype=np.int32)
                index_block[valid_mask] = global_1d_indices[valid_mask]
                dst.write(index_block, 1, window=window)

# ==============================================================================
# PIPELINE ORCHESTRATOR 
# ==============================================================================
def execute_stac_inference_pipeline(
    input_shp_path: str, model_path: str, final_out_dir: str, output_basename: str, 
    inference_start_date: str, inference_end_date: str, lmbd: float = 0.5, d: int = 2,
    clip_bounds: Tuple[float, float] = (-1.0, 1.0), n_jobs: int = -1,
    export_raw_mosaic: bool = False, export_smoothed_mosaic: bool = False,
    export_index_mask: bool = False, delete_raw_tiles: bool = True,
    ndvi_start: str = "2025-11-24", ndvi_end: str = "2026-09-09",
    step_days: int = 8, res_m: int = 10, tile_deg: float = 0.1,
    fetch_workers: int = 4, clip_tiles_to_aoi: bool = True,
):
    # The notebook took these from cell-level globals. Named here instead so a script
    # cannot pick up a different window than the notebook did without saying so.
    NDVI_START, NDVI_END = ndvi_start, ndvi_end
    STEP_DAYS, EXPORT_SCALE, TILE_DEG = step_days, res_m, tile_deg
    out_dir = Path(final_out_dir)
    raw_tiles_dir = out_dir / "raw_stac_tiles"
    chunks_dir = out_dir / "processing_chunks"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_tiles_dir.mkdir(exist_ok=True)
    chunks_dir.mkdir(exist_ok=True)
    
    final_class_path = out_dir / f"{output_basename}_rf_classification_map.tif"
    index_map_path = out_dir / f"{output_basename}_index_mask.tif"
    
    if not final_class_path.exists():
        logger.info("--- PHASE 1: STAC IMAGERY ACQUISITION (I/O BOUND) ---")
        logger.info("NOTE: Downloading time-series chunks. This may take several minutes. Please wait for Phase 2...")
        
        fetch_sentinel_imagery(
            aoi=input_shp_path,
            start=NDVI_START,
            end=NDVI_END,
            bands=["red", "nir"],
            out_dir=str(raw_tiles_dir),
            step=STEP_DAYS,
            res_m=EXPORT_SCALE,
            tile_deg=TILE_DEG,
            cloud_lt=97,
            workers=fetch_workers,
            build_vrt_mosaic=False, 
            clip_to_aoi=False
        )
        
        raw_tifs = list(raw_tiles_dir.glob("sentinel_*m_tile_*.tif"))
        if not raw_tifs:
            raise FileNotFoundError("STAC acquisition failed to output tiles.")

        if clip_tiles_to_aoi:
            logger.info("--- CLIPPING TILES TO THE AOI ---")
            clipped = burn_aoi_into_tiles(raw_tifs, input_shp_path)
            logger.info(f"clipped {clipped} tile(s); {len(raw_tifs) - clipped} were "
                        f"already done")
            
        logger.info(f"--- PHASE 2: DISTRIBUTED INFERENCE ({len(raw_tifs)} Grids) ---")
        total_cores = multiprocessing.cpu_count()
        usable_cores = max(1, int(total_cores * 0.75)) if n_jobs == -1 else n_jobs
        
        rf_model = joblib.load(model_path)
        rf_model.n_jobs = 1  
        
        processed_results = []

        # A tile whose chunks are already on disk is not inferred again. Losing a run
        # part way through used to mean redoing every tile, and the parts that survive
        # a kill are exactly the parts worth keeping: the chunks are written before the
        # process moves on, so whatever is there is finished.
        def _existing(tif_path):
            stem = Path(tif_path).stem
            done = {"pred": chunks_dir / f"{stem}_predicted.tif",
                    "raw": chunks_dir / f"{stem}_raw.tif" if export_raw_mosaic else None,
                    "smoothed": chunks_dir / f"{stem}_smoothed.tif" if export_smoothed_mosaic else None}
            if not done["pred"].exists():
                return None
            for key in ("raw", "smoothed"):
                if done[key] is not None and not done[key].exists():
                    return None
            return {k: (v if v is None else str(v)) for k, v in done.items()}

        pending = []
        for tif_path in raw_tifs:
            found = _existing(tif_path)
            if found is not None:
                processed_results.append(found)
            else:
                pending.append(tif_path)
        if processed_results:
            logger.info(f"{len(processed_results)} tiles already inferred; "
                        f"{len(pending)} left to do")

        with ProcessPoolExecutor(max_workers=usable_cores, max_tasks_per_child=1) as executor:
            futures = {
                executor.submit(
                    worker_process_local_tile, tif_path, chunks_dir, rf_model,
                    inference_start_date, inference_end_date, lmbd, d, clip_bounds, 255,
                    export_raw_mosaic, export_smoothed_mosaic
                ): tif_path
                for tif_path in pending
            }
            
            # Using strict sys.stdout and dynamic_ncols to prevent multi-line rendering
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Grids", file=sys.stdout, dynamic_ncols=True, leave=True):
                tif_path = futures[future]
                try:
                    result = future.result(timeout=600)
                    processed_results.append(result)
                except Exception as e:
                    logger.error(f"[ERROR ENCOUNTERED] Error processing {tif_path.name}: {e}")
        
        del rf_model
        release_os_memory()

        logger.info("--- PHASE 3: MOSAICING (Code 1 rasterio.merge logic) ---")
        try:
            pred_paths = [res["pred"] for res in processed_results if res["pred"] is not None]
            if pred_paths: 
                mosaic_categorical_rasters(pred_paths, final_class_path, "RF_Classification")
            
            if export_raw_mosaic:
                raw_paths = [res["raw"] for res in processed_results if res["raw"] is not None]
                if raw_paths:
                    stack_as_vrt(raw_paths, out_dir / f"{output_basename}_raw_mosaic.vrt")
                
            if export_smoothed_mosaic:
                smoothed_paths = [res["smoothed"] for res in processed_results if res["smoothed"] is not None]
                if smoothed_paths:
                    stack_as_vrt(smoothed_paths, out_dir / f"{output_basename}_smoothed_mosaic.vrt")
            
            if delete_raw_tiles:
                if final_class_path.exists():
                    logger.info("--- CLEANING UP RAW STAC TILES ---")
                    shutil.rmtree(raw_tiles_dir, ignore_errors=True)
                else:
                    logger.warning("--- SKIPPING RAW TILE CLEANUP: Final map was not produced. Tiles retained for debugging. ---")

        finally:
            # The smoothed and raw chunks are what the VRTs point at, so they are not
            # temporary any more. Only the predicted chunks can go, and only once the
            # categorical mosaic that absorbed them exists.
            if final_class_path.exists():
                removed = 0
                for chunk in chunks_dir.glob("*_predicted.tif"):
                    chunk.unlink(missing_ok=True)
                    removed += 1
                logger.info(f"--- CLEANED UP {removed} PREDICTED CHUNKS; "
                            f"KEEPING THE STACKS THE VRTs READ ---")
            else:
                logger.warning("--- KEEPING ALL CHUNKS: the final map was not produced ---")
            release_os_memory()
            
    else:
        logger.info(f"[CHECKPOINT FOUND] Classification map already exists at: {final_class_path}")

    if export_index_mask:
        if final_class_path.exists() and not index_map_path.exists():
            generate_global_index_mask(final_class_path, index_map_path)
            release_os_memory()

    if not final_class_path.exists():
        logger.error(f"PIPELINE FAILED: The final output map was NOT produced.")
        return None
            
    return final_class_path
