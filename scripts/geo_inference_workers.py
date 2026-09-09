import os
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

def parse_stac_bands(descriptions: Tuple[str, ...]) -> Tuple[List[int], List[int], List[str]]:
    """Parses red and nir bands generated dynamically by sentinel.py STAC fetcher."""
    red_idx, nir_idx, dates = [], [], []
    for i, d in enumerate(descriptions):
        if not d: continue
        if d.startswith('red_'):
            red_idx.append(i)
            parts = d.split('_')
            dates.append(f"{parts[1]}-{parts[2]}-{parts[3]}")
        elif d.startswith('nir_'):
            nir_idx.append(i)
    assert len(red_idx) == len(nir_idx), "Mismatch between Red and NIR band counts."
    return red_idx, nir_idx, dates

def get_penalty_matrix(n: int, lmbd: float, d: int) -> np.ndarray:
    E = np.eye(n)
    D = np.diff(E, n=d, axis=0)
    return lmbd * (D.T @ D)

def whittaker_pixel(y: np.ndarray, penalty_mat: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    if not mask.any(): return y
    W = np.diag(mask.astype(float))
    A = W + penalty_mat
    rhs = np.zeros_like(y)
    rhs[mask] = y[mask]
    try: 
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError: 
        return np.linalg.lstsq(A, rhs, rcond=None)[0]

def process_smoothing_chunk(
    data_chunk: np.ndarray, penalty_mat: np.ndarray,
    clip_bounds: Optional[Tuple[float, float]], nodata_val: Optional[float]
) -> np.ndarray:
    B, H, W_w = data_chunk.shape
    pixels = data_chunk.transpose(1, 2, 0).reshape(-1, B).astype(np.float32)

    missing_mask = np.isnan(pixels) if nodata_val is None or np.isnan(nodata_val) else (pixels == nodata_val)
    missing_counts = missing_mask.sum(axis=1)

    full_valid_idx = np.where(missing_counts == 0)[0]
    partial_valid_idx = np.where((missing_counts > 0) & (missing_counts < B))[0]

    if len(full_valid_idx) > 0:
        A_full = np.eye(B) + penalty_mat
        try: 
            pixels[full_valid_idx] = np.linalg.solve(A_full, pixels[full_valid_idx].T).T
        except np.linalg.LinAlgError: 
            pixels[full_valid_idx] = np.linalg.lstsq(A_full, pixels[full_valid_idx].T, rcond=None)[0].T

    if len(partial_valid_idx) > 0:
        pixels[partial_valid_idx] = np.apply_along_axis(whittaker_pixel, 1, pixels[partial_valid_idx], penalty_mat=penalty_mat)

    if clip_bounds:
        processed_idx = np.concatenate([full_valid_idx, partial_valid_idx])
        if len(processed_idx) > 0: 
            pixels[processed_idx] = np.clip(pixels[processed_idx], *clip_bounds)

    return pixels.reshape(H, W_w, B).transpose(2, 0, 1)

def worker_process_local_tile(
    raw_local_path: Path, out_dir: Path, rf_model: Any, 
    inference_start_date: str, inference_end_date: str, lmbd: float, d: int, 
    clip_bounds: Tuple[float, float], output_nodata: int = 255,
    export_raw: bool = False, export_smoothed: bool = False
) -> Dict[str, Any]:
    """Processes a single STAC tile stored locally: Reads, Smooths, Predicts, and saves."""

    # Sandboxing GDAL environment per-process prevents C-level cache/thread corruption
    with rasterio.Env(GDAL_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'):
        grid_filename = raw_local_path.name
        pred_local_path = out_dir / grid_filename.replace('.tif', '_predicted.tif')
        pred_tmp_path = pred_local_path.with_suffix('.tmp.tif')

        smoothed_local_path = out_dir / grid_filename.replace('.tif', '_smoothed.tif') if export_smoothed else None
        smoothed_tmp_path = smoothed_local_path.with_suffix('.tmp.tif') if export_smoothed else None

        # Robust Resume Logic: Physically reads a 1x1 block to verify the LZW stream isn't truncated
        if pred_local_path.exists():
            try:
                with rasterio.open(pred_local_path) as chk:
                    _ = chk.read(1, window=Window(0, 0, 1, 1)) 
                return {"pred": pred_local_path, "raw": raw_local_path if export_raw else None, "smoothed": smoothed_local_path}
            except Exception:
                pred_local_path.unlink(missing_ok=True)

        try:
            with rasterio.open(raw_local_path) as src:
                red_idx, nir_idx, raster_dates = parse_stac_bands(src.descriptions)
                dt_index = pd.to_datetime(raster_dates, errors='coerce')

                target_mask = (dt_index >= pd.to_datetime(inference_start_date)) & (dt_index <= pd.to_datetime(inference_end_date))
                target_band_indices = np.where(target_mask)[0]

                n_timesteps = len(raster_dates)
                penalty_mat = get_penalty_matrix(n_timesteps, lmbd, d)

                # Explicitly define clean output profiles to prevent inheriting conflicting STAC metadata
                out_profile = {
                    "driver": "GTiff",
                    "height": src.height,
                    "width": src.width,
                    "transform": src.transform,
                    "crs": src.crs,
                    "dtype": rasterio.uint8,
                    "count": 1,
                    "nodata": output_nodata,
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "predictor": 2,
                    "bigtiff": "YES"
                }

                smoothed_dst = None
                if export_smoothed:
                    smooth_meta = out_profile.copy()
                    smooth_meta.update({
                        "dtype": "float32", 
                        "count": n_timesteps
                    })
                    smoothed_dst = rasterio.open(smoothed_tmp_path, 'w', **smooth_meta)
                    for i, date_str in enumerate(raster_dates, 1):
                        smoothed_dst.set_band_description(i, f"NDVI_{date_str.replace('-', '_')}")

                try:
                    with rasterio.open(pred_tmp_path, 'w', **out_profile) as dst:
                        for _, window in src.block_windows(1):
                            raw_chunk = src.read(window=window)

                            red = raw_chunk[red_idx, :, :].astype(np.float32)
                            nir = raw_chunk[nir_idx, :, :].astype(np.float32)

                            denom = nir + red
                            valid_px_mask = denom > 0
                            ndvi_chunk = np.full_like(red, np.nan)
                            np.divide(nir - red, denom, out=ndvi_chunk, where=valid_px_mask)

                            smoothed_chunk = process_smoothing_chunk(ndvi_chunk, penalty_mat, clip_bounds, nodata_val=np.nan)

                            if smoothed_dst:
                                smoothed_dst.write(smoothed_chunk, window=window)

                            sliced_data = smoothed_chunk[target_band_indices, :, :]
                            B, H, W = sliced_data.shape
                            pixels = sliced_data.transpose(1, 2, 0).reshape(-1, B)

                            valid_mask = ~np.all(np.isnan(pixels) | (pixels == 0), axis=1)
                            pred_block = np.full(pixels.shape[0], output_nodata, dtype=np.uint8)

                            if valid_mask.any():
                                valid_pixels = np.nan_to_num(pixels[valid_mask], nan=0.0) 
                                pred_block[valid_mask] = rf_model.predict(valid_pixels).astype(np.uint8)

                            dst.write(pred_block.reshape(H, W), 1, window=window)

                    pred_tmp_path.rename(pred_local_path)

                finally:
                    if smoothed_dst:
                        smoothed_dst.close()
                        if smoothed_tmp_path and smoothed_tmp_path.exists():
                            smoothed_tmp_path.rename(smoothed_local_path)

        finally:
            if pred_tmp_path.exists(): pred_tmp_path.unlink()
            if smoothed_tmp_path and smoothed_tmp_path.exists(): smoothed_tmp_path.unlink()
            gc.collect()

        return {
            "pred": pred_local_path, 
            "raw": raw_local_path if export_raw else None, 
            "smoothed": smoothed_local_path if export_smoothed else None
        }
