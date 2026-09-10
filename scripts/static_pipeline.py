"""The static stage, as a module instead of a notebook cell.

Lifted out of `Model_Execution_Pipeline_v3.0.ipynb` unchanged, so a script and the
notebook cannot drift apart. Fetches a single-date mosaic over the AOI, aligns the
time-series mask to it, and classifies. `classify_large_image` is the notebook's own
path; scripts here call `static_training.inference.classify_raster` instead, which adds
the domain guard and the model's own threshold, and uses `create_aligned_mask` from
this module to line the mask up first.
"""

import os
import sys
import gc
import ctypes
import logging
import warnings
import shutil
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from osgeo import gdal
import xgboost as xgb
from tqdm import tqdm
from typing import Any, Optional, Sequence
from pathlib import Path
from datetime import datetime
from sentinel import fetch_sentinel_static_imagery
from static_training import inference as st_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def release_os_memory() -> None:
    """Forces Python, GDAL, and the OS (glibc) to release unreferenced memory."""
    gdal.SetCacheMax(0) 
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError:
        pass

def configure_model_hardware(model: Any) -> Any:
    logger.info("Detecting hardware acceleration capabilities...")
    dummy_data = np.zeros((1, 6), dtype=np.float32)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            model.set_params(device="cuda")
            model.predict(dummy_data)
            
            fell_back = any("Device is changed from GPU to CPU" in str(warn.message) or 
                            "No visible GPU is found" in str(warn.message) for warn in w)
            
            if not fell_back:
                logger.info("Hardware bound: GPU (CUDA via device='cuda')")
                return model
        except Exception:
            pass

    model.set_params(device="cpu", n_jobs=-1)
    logger.info("Hardware bound: CPU (Utilizing all available cores for fast processing)")
    return model

def _assert_grid_parity(path_a: str, path_b: str) -> None:
    """Hard gate: two rasters must share CRS, transform, and dimensions."""
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        assert a.crs == b.crs, f"CRS mismatch: {a.crs} vs {b.crs}"
        assert (a.width, a.height) == (b.width, b.height), (
            f"Dimension mismatch: {a.width}x{a.height} vs {b.width}x{b.height}"
        )
        assert np.allclose(tuple(a.transform), tuple(b.transform), atol=1e-9), (
            f"Transform mismatch:\n{a.transform}\nvs\n{b.transform}"
        )

def create_aligned_mask(
    target_raster_path: str,
    raw_mask_path: str,
    shp_path: str,
    output_mask_path: str,
    keep_values: Sequence[int] = (1,),
    chunk_size: int = 2048,
    max_coverage_fraction: float = 0.90
) -> None:
    """
    Builds a 1:1 aligned binary mask clipped to the AOI and specific class labels.
    """
    logger.info(f"Generating aligned mask | keep_values={tuple(keep_values)} | clip={shp_path}")

    gdf = gpd.read_file(shp_path)
    assert not gdf.empty, "Pipeline Error: Input shapefile is empty."
    assert gdf.is_valid.all(), "Pipeline Error: Invalid geometries in clip shapefile."

    with rasterio.open(target_raster_path) as trg:
        trg_profile = trg.profile.copy()
        trg_crs = trg.crs
        trg_transform = trg.transform
        trg_width, trg_height = trg.width, trg.height

    if gdf.crs != trg_crs:
        gdf = gdf.to_crs(trg_crs)
    geometries = gdf.geometry.values

    # THE FIX: Explicitly enforce GTiff driver and add tiling for optimized downstream reading
    trg_profile.update(
        driver="GTiff",
        count=1, 
        dtype=rasterio.uint8, 
        nodata=0, 
        compress="lzw", 
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="YES"
    )
    
    keep_arr = np.asarray(keep_values)
    kept_px = 0
    total_px = trg_width * trg_height

    with rasterio.open(raw_mask_path) as src:
        vrt_options = {
            "resampling": Resampling.nearest,
            "crs": trg_crs,
            "transform": trg_transform,
            "height": trg_height,
            "width": trg_width,
            "nodata": src.nodata if src.nodata is not None else 0,
        }

        with WarpedVRT(src, **vrt_options) as vrt, rasterio.open(output_mask_path, "w", **trg_profile) as dst:
            y_offsets = range(0, trg_height, chunk_size)
            x_offsets = range(0, trg_width, chunk_size)
            n_blocks = len(list(y_offsets)) * len(list(x_offsets))

            with tqdm(total=n_blocks, desc="Clipping Mask", unit="block", file=sys.stdout, dynamic_ncols=True) as pbar:
                for y in range(0, trg_height, chunk_size):
                    for x in range(0, trg_width, chunk_size):
                        win_h = min(chunk_size, trg_height - y)
                        win_w = min(chunk_size, trg_width - x)
                        win = Window(x, y, win_w, win_h)

                        mask_chunk = vrt.read(1, window=win)
                        win_transform = rasterio.windows.transform(win, trg_transform)

                        geom_mask = geometry_mask(
                            geometries,
                            out_shape=(win_h, win_w),
                            transform=win_transform,
                            invert=True,
                            all_touched=False,
                        )

                        class_mask = np.isin(mask_chunk, keep_arr)
                        final_mask = (class_mask & geom_mask).astype(rasterio.uint8)
                        
                        kept_px += int(final_mask.sum())
                        dst.write(final_mask, 1, window=win)
                        pbar.update(1)

    coverage = kept_px / total_px
    logger.info(f"Mask coverage: {kept_px:,} px ({coverage:.2%} of target grid)")
    
    # Allow 0 coverage for areas where the target crop genuinely isn't present
    if coverage == 0:
        logger.warning(f"Degenerate mask: zero pixels kept. Target classes {keep_values} were not found in this AOI.")
        
    assert coverage < max_coverage_fraction, (
        f"Degenerate mask: {coverage:.2%} coverage exceeds {max_coverage_fraction:.0%}. "
        "Binarization is likely admitting background classes."
    )

    _assert_grid_parity(target_raster_path, output_mask_path)
    logger.info("Grid parity verified: mask is 1:1 with input image.")

def classify_large_image(
    input_path: str, 
    output_path: str, 
    mask_raster_path: Optional[str],
    input_shp_path: Optional[str],
    model: Any, 
    chunk_size: int = 2048,
    mask_keep_values: Sequence[int] = (1,),
    use_mask: bool = True,
    target_class_in: int = 1,
    target_class_out: int = 1,
    background_out: int = 0
) -> None:
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aligned_mask_path = None

    if use_mask:
        assert mask_raster_path and input_shp_path, (
            "use_mask=True requires both mask_raster_path and input_shp_path."
        )
        aligned_mask_path = str(Path(output_path).parent / f"{Path(output_path).stem}_temp_mask.tif")
        create_aligned_mask(
            target_raster_path=input_path,
            raw_mask_path=mask_raster_path,
            shp_path=input_shp_path,
            output_mask_path=aligned_mask_path,
            keep_values=mask_keep_values,
            chunk_size=chunk_size,
        )

    with rasterio.open(input_path) as src:
        mask_src = rasterio.open(aligned_mask_path) if use_mask else None
        try:
            profile = src.profile.copy()
            width, height = src.width, src.height
            bands = src.count
            nodata_val = src.nodata if src.nodata is not None else 0

            profile.update(
                driver="GTiff", dtype=rasterio.uint8, count=1, nodata=background_out,
                compress="lzw", tiled=True, blockxsize=256, blockysize=256, bigtiff="YES"
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                y_offsets = list(range(0, height, chunk_size))
                x_offsets = list(range(0, width, chunk_size))
                total_blocks = len(y_offsets) * len(x_offsets)

                with tqdm(total=total_blocks, desc="Classifying Raster Blocks", unit="block", file=sys.stdout, dynamic_ncols=True) as pbar:
                    for y in y_offsets:
                        for x in x_offsets:
                            win_h = min(chunk_size, height - y)
                            win_w = min(chunk_size, width - x)
                            win = Window(x, y, win_w, win_h)
                            
                            label_chunk_1d = np.full(win_h * win_w, background_out, dtype=np.uint8)

                            if use_mask:
                                # Aligned mask writes 1 for kept valid pixels
                                geo_mask = mask_src.read(1, window=win) == 1
                            else:
                                geo_mask = np.ones((win_h, win_w), dtype=bool)
                            
                            if not geo_mask.any():
                                dst.write(label_chunk_1d.reshape((win_h, win_w)), 1, window=win)
                                pbar.update(1)
                                if use_mask:
                                    del geo_mask
                                continue

                            chunk = src.read(window=win)
                            if isinstance(nodata_val, float) and np.isnan(nodata_val):
                                valid_mask = np.any(~np.isnan(chunk), axis=0)
                            else:
                                valid_mask = np.any(chunk != nodata_val, axis=0)

                            combined_mask_1d = (valid_mask & geo_mask).ravel()

                            if combined_mask_1d.any():
                                reshaped = chunk.reshape(bands, -1).T
                                valid_pixels = reshaped[combined_mask_1d].astype(np.float32)

                                preds = model.predict(valid_pixels)
                                mapped_preds = np.where(preds == target_class_in, target_class_out, background_out).astype(np.uint8) 
                                label_chunk_1d[combined_mask_1d] = mapped_preds
                                
                                del reshaped, valid_pixels, preds, mapped_preds

                            dst.write(label_chunk_1d.reshape((win_h, win_w)), 1, window=win)
                            pbar.update(1)
                            
                            del chunk, label_chunk_1d, combined_mask_1d
                            if use_mask:
                                del geo_mask
        finally:
            if mask_src is not None:
                mask_src.close()
                # Clean up the temporary aligned mask to save disk space
                Path(aligned_mask_path).unlink(missing_ok=True)

def execute_static_pipeline(
    base_dir: Path, 
    mask_path: Optional[Path], 
    input_shp_path: str,
    model_file: str, 
    delete_tiles: bool,
    use_mask: bool,
    mask_keep_values: Sequence[int],
    target_class_in: int,
    target_class_out: int,
    background_out: int,
    static_start: str,
    static_end: str,
    export_scale: int,
    tile_deg: float,
    dates: Optional[Sequence[str]] = None,
    n_dates: int = 2,
    fetch_workers: int = 4,
) -> Path:
    # The notebook read the wanted dates from a cell global. Named here so a script
    # cannot silently classify a different image than the one it asked for.
    static_image_date = list(dates) if dates else None
    
    base_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = base_dir / "staging_temp"
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initiating STAC acquisition for optimal static composite into staging directory...")
    
    static_result = fetch_sentinel_static_imagery(
        aoi=input_shp_path,
        start=static_start,
        end=static_end,
        bands=["blue", "green", "red", "rededge1", "nir", "ndvi"],
        out_dir=str(staging_dir),
        res_m=export_scale,
        tile_deg=tile_deg,
        n_dates=n_dates,
        dates=static_image_date,
        mask_clouds=False,
        workers=fetch_workers,
        build_vrt_mosaic=True,
        clip_to_aoi=False
    )
    
    raw_dates = static_result.get("dates", [])
    if not raw_dates:
        logger.warning("No 'dates' key found. Defaulting to 'Unknown_Date'.")
        date_suffix = "Unknown_Date"
    else:
        formatted_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%d_%b_%Y") for d in raw_dates]
        date_suffix = "_and_".join(formatted_dates)
        
    final_out_dir = base_dir / date_suffix
    final_output_path = final_out_dir / f"static_mosaic_{date_suffix}_Cls.tif"

    if final_output_path.exists():
        logger.info(f"[CHECKPOINT FOUND] Classification map for {date_suffix} already exists at: {final_output_path}")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return final_output_path

    final_out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Renaming chunks and mapping to target directory: {final_out_dir.name}")
    new_chunk_paths = []
    
    for chunk in staging_dir.glob("*.tif"):
        new_chunk_name = f"{chunk.stem}_{date_suffix}{chunk.suffix}"
        new_chunk_path = final_out_dir / new_chunk_name
        shutil.move(str(chunk), str(new_chunk_path))
        new_chunk_paths.append(str(new_chunk_path))
        
    for meta_file in staging_dir.glob("*.json"):
        new_meta_name = f"{meta_file.stem}_{date_suffix}{meta_file.suffix}"
        shutil.move(str(meta_file), str(final_out_dir / new_meta_name))

    input_image_path = final_out_dir / f"static_mosaic_{date_suffix}.vrt"
    if new_chunk_paths:
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
        gdal.BuildVRT(str(input_image_path), new_chunk_paths, options=vrt_options)

    shutil.rmtree(staging_dir, ignore_errors=True)
    
    # The mask is built exactly as before; only the classification step changes.
    aligned_mask_path = None
    if use_mask:
        aligned_mask_path = str(final_out_dir / f"{final_output_path.stem}_temp_mask.tif")
        create_aligned_mask(
            target_raster_path=str(input_image_path),
            raw_mask_path=str(mask_path),
            shp_path=input_shp_path,
            output_mask_path=aligned_mask_path,
            keep_values=mask_keep_values,
            chunk_size=2048,
        )

    # classify_raster reads its decision threshold from the model sidecar instead of
    # assuming 0.5, builds features from the same module training used, and scores the
    # image against the distribution the model was fitted on before classifying it.
    # It also writes a probability raster, so the threshold can be revisited later
    # without running inference over the AOI a second time.
    logger.info(f"Initiating classification pipeline for: {input_image_path}")
    probability_path = final_out_dir / f"static_mosaic_{date_suffix}_prob.tif"

    result = st_inference.classify_raster(
        raster_path=str(input_image_path),
        model_path=model_file,
        out_label_path=str(final_output_path),
        out_probability_path=str(probability_path),
        mask_path=aligned_mask_path,
        positive_out=target_class_out,
        background_out=background_out,
        enforce_domain=False,   # set True once the guard is trusted to stop a run
    )

    logger.info(f"Domain guard: {result.verdict}")
    logger.info(
        f"Threshold {result.threshold:.2f} taken from the sidecar; "
        f"{100 * result.positive_fraction:.1f}% of masked pixels classified as crop"
    )
    logger.info(f"Probability raster: {probability_path}")
    
    if final_output_path.exists():
        logger.info(f"Classification complete. GeoTIFF generated at: {final_output_path}")

    if delete_tiles:
        if final_output_path.exists():
            logger.info(f"--- CLEANING UP RAW STATIC STAC TILES AND VRT IN {final_out_dir.name} ---")
            for p in final_out_dir.rglob("*"):
                if p.is_file() and p.suffix in ['.tif', '.vrt', '.json']:
                    # Protect the final output from deletion
                    if p.absolute() != final_output_path.absolute():
                        try:
                            p.unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to delete {p.name}: {e}")
        else:
            logger.warning("--- SKIPPING RAW TILE CLEANUP: Final map was not produced. Tiles retained for debugging. ---")

    # --- FINAL SAFETY CHECK ---
    if not final_output_path.exists():
        logger.error("❌ PIPELINE FAILED: The final output map was NOT produced.")
        return None
                        
    return final_output_path