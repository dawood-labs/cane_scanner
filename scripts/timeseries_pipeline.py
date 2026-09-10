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
        nodata_val = template_src.nodata or 255
            
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
    fetch_workers: int = 4,
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
            
        logger.info(f"--- PHASE 2: DISTRIBUTED INFERENCE ({len(raw_tifs)} Grids) ---")
        total_cores = multiprocessing.cpu_count()
        usable_cores = max(1, int(total_cores * 0.75)) if n_jobs == -1 else n_jobs
        
        rf_model = joblib.load(model_path)
        rf_model.n_jobs = 1  
        
        processed_results = []
        
        with ProcessPoolExecutor(max_workers=usable_cores, max_tasks_per_child=1) as executor:
            futures = {
                executor.submit(
                    worker_process_local_tile, tif_path, chunks_dir, rf_model,
                    inference_start_date, inference_end_date, lmbd, d, clip_bounds, 255,
                    export_raw_mosaic, export_smoothed_mosaic
                ): tif_path
                for tif_path in raw_tifs
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
                    mosaic_continuous_rasters(raw_paths, out_dir / f"{output_basename}_raw_mosaic.tif")
                
            if export_smoothed_mosaic:
                smoothed_paths = [res["smoothed"] for res in processed_results if res["smoothed"] is not None]
                if smoothed_paths: 
                    mosaic_continuous_rasters(smoothed_paths, out_dir / f"{output_basename}_smoothed_mosaic.tif")
            
            if delete_raw_tiles:
                if final_class_path.exists():
                    logger.info("--- CLEANING UP RAW STAC TILES ---")
                    shutil.rmtree(raw_tiles_dir, ignore_errors=True)
                else:
                    logger.warning("--- SKIPPING RAW TILE CLEANUP: Final map was not produced. Tiles retained for debugging. ---")

        finally:
            logger.info("--- CLEANING UP TEMPORARY CHUNKS ---")
            shutil.rmtree(chunks_dir, ignore_errors=True)
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
