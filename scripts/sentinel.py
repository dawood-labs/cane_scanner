import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from typing import Tuple, Union, Optional


def align_to_sentinel_grid(
    source_data: np.ndarray,
    source_transform: rasterio.Affine,
    source_crs: rasterio.CRS,
    source_nodata: Union[float, int],
    target_resolution: int = 10,
    resampling_method: Resampling = Resampling.average
) -> Tuple[np.ndarray, rasterio.Affine, int, int]:
    """
    Resample raster data to align with the Sentinel-2 grid.
    
    This function takes a raster dataset and resamples it to align with the Sentinel-2
    grid system at the specified resolution (default 10m). The output grid will have
    coordinates that are multiples of the target resolution, ensuring proper alignment
    with other Sentinel-2 derived products.
    
    Args:
        source_data: Source raster data array
        source_transform: Affine transform of the source raster
        source_crs: Coordinate reference system of the source raster
        source_nodata: Value representing no data in the source raster
        target_resolution: Target resolution in meters (default: 10m for Sentinel-2)
        resampling_method: Resampling method to use (default: average)
        
    Returns:
        Tuple containing:
        - resampled_data: The resampled raster data aligned to the Sentinel grid
        - dst_transform: The affine transform of the resampled raster
        - dst_width: Width of the resampled raster in pixels
        - dst_height: Height of the resampled raster in pixels
    
    Example:
        >>> import rasterio
        >>> from rasterio.warp import Resampling
        >>> with rasterio.open('input.tif') as src:
        ...     data = src.read(1)
        ...     transform = src.transform
        ...     crs = src.crs
        ...     nodata = src.nodata
        ...
        >>> resampled_data, new_transform, width, height = align_to_sentinel_grid(
        ...     data, transform, crs, nodata, target_resolution=10
        ... )
        >>> # Write to a new file
        >>> profile = src.profile.copy()
        >>> profile.update({
        ...     'height': height,
        ...     'width': width,
        ...     'transform': new_transform
        ... })
        >>> with rasterio.open('output.tif', 'w', **profile) as dst:
        ...     dst.write(resampled_data, 1)
    """
    # Get current bounds of the raster
    bounds = rasterio.transform.array_bounds(
        source_data.shape[0], source_data.shape[1], source_transform
    )
    minx, miny, maxx, maxy = bounds
    
    # Adjust bounds to align with Sentinel grid (multiples of target_resolution)
    # By using floor/ceil, we ensure the new grid completely contains the original data
    aligned_minx = np.floor(minx / target_resolution) * target_resolution
    aligned_miny = np.floor(miny / target_resolution) * target_resolution
    aligned_maxx = np.ceil(maxx / target_resolution) * target_resolution
    aligned_maxy = np.ceil(maxy / target_resolution) * target_resolution
    
    # Calculate dimensions of the aligned grid
    dst_width = int((aligned_maxx - aligned_minx) / target_resolution)
    dst_height = int((aligned_maxy - aligned_miny) / target_resolution)
    
    # Create transform for aligned grid
    # The origin is the upper left corner (aligned_minx, aligned_maxy)
    dst_transform = rasterio.transform.from_origin(
        aligned_minx, aligned_maxy, target_resolution, target_resolution
    )
    
    # Create destination array initialized with nodata values
    dst_array = np.full(
        (dst_height, dst_width), source_nodata, dtype=source_data.dtype
    )
    
    # Reproject to aligned grid
    reproject(
        source=source_data,
        destination=dst_array,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=dst_transform,
        dst_crs=source_crs,  # Keep the same CRS
        resampling=resampling_method,
        src_nodata=source_nodata,
        dst_nodata=source_nodata
    )
    
    return dst_array, dst_transform, dst_width, dst_height


def align_raster_file(
    input_path: str,
    output_path: str,
    target_resolution: int = 10,
    resampling_method: Resampling = Resampling.average,
    band: int = 1
) -> None:
    """
    Align a raster file to the Sentinel grid and save to a new file.
    
    This is a convenience wrapper around align_to_sentinel_grid that operates
    directly on files rather than arrays.
    
    Args:
        input_path: Path to the input raster file
        output_path: Path where the aligned raster should be saved
        target_resolution: Target resolution in meters (default: 10m)
        resampling_method: Resampling method to use (default: average)
        band: Band number to process (default: 1)
        
    Returns:
        None
    """
    with rasterio.open(input_path) as src:
        # Read data
        data = src.read(band)
        
        # Get metadata
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else 0
        
        # Align to Sentinel grid
        aligned_data, aligned_transform, width, height = align_to_sentinel_grid(
            data, transform, crs, nodata, 
            target_resolution=target_resolution,
            resampling_method=resampling_method
        )
        
        # Update profile for output
        profile = src.profile.copy()
        profile.update({
            'height': height,
            'width': width,
            'transform': aligned_transform,
            'compress': 'lzw'  # Add compression
        })
        
        # Write output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(aligned_data, band)


# Example usage
if __name__ == "__main__":
    # Example of aligning a file
    # align_raster_file(
    #     "input.tif",
    #     "output_aligned.tif",
    #     target_resolution=10,
    #     resampling_method=Resampling.average
    # )
    
    # Or use the lower-level function directly
    # with rasterio.open("input.tif") as src:
    #     data = src.read(1)
    #     aligned, transform, width, height = align_to_sentinel_grid(
    #         data, src.transform, src.crs, src.nodata
    #     )
    pass


# =============================================================================
# Open-source Sentinel-2 imagery acquisition (GEE-free, environment-agnostic)
# =============================================================================
#
# A generalized, framework-independent replacement for the YieldPro Airflow
# `s2_baseline.py` builder. It pulls Sentinel-2 L2A imagery straight from free
# public STAC catalogs (Microsoft Planetary Computer primary, Element84 Earth
# Search fallback) -- NO Google Earth Engine, NO Airflow, NO Celery/S3.
#
# Anyone in the company can call these from anywhere -- a JupyterHub notebook, a
# script, a different DAG, a service -- just by connecting the standard-libraries
# repo:
#
#     from farmdar.sentinel import fetch_sentinel_composite, fetch_sentinel_imagery
#
#     # Quick single-composite (best for interactive / JupyterHub use):
#     out = fetch_sentinel_composite(
#         aoi="field.geojson",           # path, shapely geom, GeoJSON dict, or bbox
#         start="2025-04-01", end="2025-04-30",
#         bands=["blue", "green", "red", "nir"],   # default; friendly names or B2/B08/...
#     )
#     rgb = out["bands"]                 # {"blue": ndarray, "green": ..., ...}
#     transform, crs = out["transform"], out["crs"]
#
#     # Full tiled baseline builder (large AOIs, seamless VRT mosaic, resume-safe):
#     summary = fetch_sentinel_imagery(
#         aoi="province.shp", out_dir="/tmp/s2",
#         start="2025-01-24", end="2025-07-31",
#         bands=["blue", "green", "red", "nir"], step=8, res_m=10,
#     )
#
# Heavy geospatial deps (odc-stac, odc-geo, pystac-client, planetary-computer,
# geopandas) are imported lazily inside the functions so that `import
# farmdar.sentinel` keeps working in environments that only need the grid-align
# helpers above.
#
# GEE parity note (carried over from s2_baseline.py): same ESA L2A source, same
# SCL mask (SCL<8 & SCL!=3), median reducer, and the -1000 baseline offset (MPC
# only; Earth Search already applies it). Verified Pearson r=0.99 on B4/B8.

import contextlib as _contextlib
import json as _json
import logging as _logging
import os as _os
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

_log = _logging.getLogger("farmdar.sentinel")

# GEE EPSG:4326 convention (30 m == 0.000269494585.. deg). Tiles snap to a single
# global grid so every tile co-registers and the mosaic is seam-free.
_DEG_PER_M = 1.0 / 111319.49
_TARGET_CRS = "EPSG:4326"

# ---- STAC providers -------------------------------------------------------
# Each provider maps the canonical band names below to its own asset keys.
_PROVIDERS = {
    "mpc": {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-2-l2a",
        "scl_asset": "SCL",
        "cloud_field": "eo:cloud_cover",
        "sign": True,
        "harmonize_offset": 1000,   # MPC ships raw DN w/ +1000 processing-baseline offset
    },
    "earthsearch": {
        "url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "scl_asset": "scl",
        "cloud_field": "eo:cloud_cover",
        "sign": False,
        "harmonize_offset": 0,      # already harmonized -> do NOT subtract again
    },
}

# ---- Canonical Sentinel-2 band registry -----------------------------------
# canonical name -> per-provider asset key + friendly aliases + native res (m).
# `bands=` accepts the canonical name, any alias, or the raw Sxx code (case /
# separator insensitive), e.g. "nir", "NIR", "b8", "B08" all resolve to NIR.
SENTINEL2_BANDS = {
    "coastal":  {"mpc": "B01", "earthsearch": "coastal",  "res_m": 60, "aliases": ["b1", "b01", "aerosol"]},
    "blue":     {"mpc": "B02", "earthsearch": "blue",     "res_m": 10, "aliases": ["b2", "b02"]},
    "green":    {"mpc": "B03", "earthsearch": "green",    "res_m": 10, "aliases": ["b3", "b03"]},
    "red":      {"mpc": "B04", "earthsearch": "red",      "res_m": 10, "aliases": ["b4", "b04"]},
    "rededge1": {"mpc": "B05", "earthsearch": "rededge1", "res_m": 20, "aliases": ["b5", "b05", "rededge"]},
    "rededge2": {"mpc": "B06", "earthsearch": "rededge2", "res_m": 20, "aliases": ["b6", "b06"]},
    "rededge3": {"mpc": "B07", "earthsearch": "rededge3", "res_m": 20, "aliases": ["b7", "b07"]},
    "nir":      {"mpc": "B08", "earthsearch": "nir",      "res_m": 10, "aliases": ["b8", "b08"]},
    "nir08":    {"mpc": "B8A", "earthsearch": "nir08",    "res_m": 20, "aliases": ["b8a", "narrownir"]},
    "nir09":    {"mpc": "B09", "earthsearch": "nir09",    "res_m": 60, "aliases": ["b9", "b09", "watervapour", "watervapor"]},
    "swir16":   {"mpc": "B11", "earthsearch": "swir16",   "res_m": 20, "aliases": ["b11", "swir1"]},
    "swir22":   {"mpc": "B12", "earthsearch": "swir22",   "res_m": 20, "aliases": ["b12", "swir2"]},
}

DEFAULT_BANDS = ("blue", "green", "red", "nir")

# alias/canonical/raw-code -> canonical, all normalized (lowercase, no separators)
_BAND_LOOKUP = {}
for _canon_name, _meta in SENTINEL2_BANDS.items():
    _BAND_LOOKUP[_canon_name] = _canon_name
    _BAND_LOOKUP[_meta["mpc"].lower()] = _canon_name          # raw Sxx code, e.g. b08
    _BAND_LOOKUP[_meta["earthsearch"].lower()] = _canon_name  # provider name, e.g. nir
    for _a in _meta["aliases"]:
        _BAND_LOOKUP[_a] = _canon_name


def resolve_bands(bands):
    """Normalize a user band list to canonical Sentinel-2 names (order-preserving,
    de-duplicated). Accepts friendly names, provider names, or raw Sxx codes,
    case- and separator-insensitive.

    >>> resolve_bands(["Blue", "B08", "NIR", "red"])
    ['blue', 'nir', 'red']
    """
    if isinstance(bands, str):
        bands = [bands]
    out, seen = [], set()
    for b in bands:
        key = str(b).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        canon = _BAND_LOOKUP.get(key)
        if canon is None:
            raise ValueError(
                f"Unknown band {b!r}. Valid names: {sorted(SENTINEL2_BANDS)} "
                f"(aliases and raw codes like B08/B8A also accepted)."
            )
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    if not out:
        raise ValueError("`bands` resolved to an empty list.")
    return out


def _set_gdal_http_env():
    """Tune GDAL's /vsicurl reader for pulling COGs over HTTP from public blob
    stores (Planetary Computer / Earth Search). Every value uses setdefault so a
    caller's own environment always wins.

    GDAL_HTTP_MULTIRANGE=SERIAL is the important one. Planetary Computer's Azure
    blob does not honour bundled multi-range HTTP requests: it answers GDAL's
    default single multi-range GET with a non-multipart 206, so GDAL logs
    `Could not find 'Content-Type: multipart/byteranges; boundary='` /
    `Request ... failed with response_code=206` and mis-slices the response. At
    low concurrency it retries and these are harmless; under many parallel workers
    the mis-slice turns into truncated reads -- `TIFFFillTile: got N bytes,
    expected M`, `TIFFReadEncodedTile() failed`, `Chunk and warp failed` -- that
    abort the load and fail the whole tile. SERIAL makes GDAL issue one bounded
    GET per byte-range, which Azure serves correctly: no multipart parsing, no
    truncation, warnings and tile failures both gone.

    NOTE: do NOT use SINGLE_GET here -- it fetches the entire span between the
    lowest and highest requested offset in one GET, which for scattered COG reads
    massively over-downloads and stalls. MERGE_CONSECUTIVE_RANGES=YES coalesces
    adjacent ranges so SERIAL issues fewer requests.

    The rest: retry transient failures instead of aborting; skip the sibling-blob
    directory listing on every open (big speedup for remote COGs); keep re-read
    chunks in a small in-memory cache."""
    # Force away from an unset OR the dangerous SINGLE_GET value (SINGLE_GET
    # over-fetches the whole span and stalls); a deliberate YES/NO/SERIAL is kept.
    if _os.environ.get("GDAL_HTTP_MULTIRANGE", "").upper() in ("", "SINGLE_GET"):
        _os.environ["GDAL_HTTP_MULTIRANGE"] = "SERIAL"
    _os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    _os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "10")
    _os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
    _os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    _os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    _os.environ.setdefault("VSI_CACHE", "TRUE")
    _os.environ.setdefault("VSI_CACHE_SIZE", "67108864")  # 64 MB


# ---- AOI handling ---------------------------------------------------------
def _load_aoi_geometry(aoi):
    """Return a shapely geometry in EPSG:4326 from any of:
      * path to a vector file (.shp/.geojson/.json/.gpkg/...) readable by geopandas
      * a shapely geometry (assumed already lon/lat / EPSG:4326)
      * a GeoJSON dict (Feature / FeatureCollection / geometry)
      * a bbox as (minx, miny, maxx, maxy)
    """
    from shapely.geometry import box, shape
    from shapely.geometry.base import BaseGeometry

    if isinstance(aoi, BaseGeometry):
        return aoi
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4 and all(
        isinstance(v, (int, float)) for v in aoi
    ):
        return box(*aoi)
    if isinstance(aoi, dict):
        t = aoi.get("type")
        if t == "FeatureCollection":
            from shapely.ops import unary_union
            return unary_union([shape(f["geometry"]) for f in aoi["features"]])
        if t == "Feature":
            return shape(aoi["geometry"])
        return shape(aoi)  # raw geometry dict
    if isinstance(aoi, str):
        import geopandas as gpd
        gdf = gpd.read_file(aoi).to_crs(4326)
        geom = gdf.geometry
        return geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    raise TypeError(
        f"Unsupported `aoi` type {type(aoi).__name__}. Pass a file path, shapely "
        f"geometry, GeoJSON dict, or (minx, miny, maxx, maxy) bbox."
    )


# ---- date windows / grid snapping -----------------------------------------
def _date_windows(start, end, step):
    curr = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if step and step > 0:
        while curr < end_dt:
            nxt = min(curr + timedelta(days=step), end_dt)
            yield curr.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
            curr = nxt
    else:  # step<=0 -> single window over the whole range
        yield start, end


def _snap(v, step):
    import numpy as np
    return np.floor(v / step) * step


def _build_tiles(aoi_geom, tile_deg, res_deg):
    """Cut the AOI bbox into uniform integer-pixel tiles on a single global-grid-
    snapped origin; keep cells intersecting the AOI. Every tile is exactly
    `tile_px` square and shares exact edges with neighbours -> seamless mosaic."""
    import numpy as np
    from shapely.geometry import box
    from rasterio.transform import from_bounds
    from odc.geo.geobox import GeoBox

    minx, miny, maxx, maxy = aoi_geom.bounds
    tile_px = max(1, int(round(tile_deg / res_deg)))
    tile_span = tile_px * res_deg
    x0 = _snap(minx, res_deg)
    y0 = _snap(miny, res_deg)
    ncols = int(np.ceil((maxx - x0) / tile_span))
    nrows = int(np.ceil((maxy - y0) / tile_span))
    tiles, tid = [], 0
    for i in range(ncols):
        for j in range(nrows):
            bx0, by0 = x0 + i * tile_span, y0 + j * tile_span
            bx1, by1 = bx0 + tile_span, by0 + tile_span
            if box(bx0, by0, bx1, by1).intersects(aoi_geom):
                gb = GeoBox((tile_px, tile_px),
                            from_bounds(bx0, by0, bx1, by1, tile_px, tile_px), _TARGET_CRS)
                tid += 1
                tiles.append({"tile_id": tid, "bbox": (bx0, by0, bx1, by1),
                              "geobox": gb, "nx": tile_px, "ny": tile_px})
    return tiles


# ---- STAC / compositing ---------------------------------------------------
def _open_catalog(provider):
    import pystac_client
    p = _PROVIDERS[provider]
    modifier = None
    if p["sign"]:
        import planetary_computer
        modifier = planetary_computer.sign_inplace
    return pystac_client.Client.open(p["url"], modifier=modifier), p


def _scl_keep_mask(scl):
    return (scl < 8) & (scl != 3)   # drops cloud 8/9, cirrus 10, snow 11, shadow 3


def _harmonize_offset(p, items):
    """Baseline-aware version of the provider's `harmonize_offset`.

    ESA introduced the +1000 DN radiometric offset with processing baseline 04.00
    (scenes acquired from 2022-01-25). Planetary Computer serves the raw DN, so the
    offset must be subtracted -- but ONLY for scenes that actually carry it.
    Subtracting it from older scenes would darken them by 1000 DN and distort every
    index. Earth Search already harmonizes, so its offset is 0 either way.

    Returns the provider offset only when the scenes involved are baseline >= 04.00
    (decided by majority when a window straddles the switch, which in practice it
    never does -- ESA reprocessed the whole archive to a single baseline)."""
    off = p.get("harmonize_offset", 0)
    if not off or not items:
        return 0
    newer = 0
    for it in items:
        bl = it.properties.get("s2:processing_baseline")
        try:
            if bl is not None:
                newer += float(bl) >= 4.0
                continue
        except (TypeError, ValueError):
            pass
        # No usable baseline property: fall back to the acquisition date. Note that
        # ESA reprocessed the older archive to baseline >= 04.00 (offset included),
        # so this branch is only reached for providers that omit the property.
        dt = getattr(it, "datetime", None)
        newer += 1 if dt is None else dt.strftime("%Y-%m-%d") >= "2022-01-25"
    return off if newer * 2 >= len(items) else 0


def _provider_assets(provider, band_order):
    """canonical band -> this provider's asset key, in order."""
    return [SENTINEL2_BANDS[b][provider] for b in band_order]


# =============================================================================
# S3 read-through cache (optional)
# =============================================================================
# When the FARMDAR_S2_CACHE_BUCKET env var is set, every Sentinel-2 scene asset
# read is served from s3://{bucket}/s2/{item_id}/{asset}.tif. First reader pays
# the public NAT/cross-region download and seeds the object; subsequent readers
# (any user, any DAG, any notebook, any pod in the same VPC) hit S3 via the
# gateway endpoint -- same region, free. Failure to cache never breaks the
# caller: we fall through to the original public href and pay NAT once. Old,
# unread objects expire via the bucket's S3 lifecycle policy -- pruning is not
# this module's concern.
_S2_CACHE_PREFIX = "s2"
_S2_CACHE_S3 = None


def _s2_cache_bucket():
    """Cache bucket from env, or None to disable."""
    return _os.environ.get("FARMDAR_S2_CACHE_BUCKET") or None


def _s2_cache_client():
    """Lazy singleton boto3 S3 client; nothing imported unless caching is on."""
    global _S2_CACHE_S3
    if _S2_CACHE_S3 is None:
        import boto3
        from botocore.config import Config as _BotoConfig
        _S2_CACHE_S3 = boto3.client(
            "s3",
            config=_BotoConfig(retries={"max_attempts": 5, "mode": "adaptive"}),
        )
    return _S2_CACHE_S3


def _s2_cache_key(item_id, asset_name):
    return f"{_S2_CACHE_PREFIX}/{item_id}/{asset_name}.tif"


def _s2_seed_and_rewrite_asset(item, asset_name, bucket, s3):
    """Ensure item.assets[asset_name] is served from s3://{bucket}/... .

    HIT -> just rewrite `.href`. MISS -> stream the current href into the
    cache first. Any error is logged and swallowed so the caller falls through
    to the original public URL."""
    from botocore.exceptions import ClientError
    key = _s2_cache_key(item.id, asset_name)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        item.assets[asset_name].href = f"s3://{bucket}/{key}"
        _log.debug("s2-cache HIT %s/%s", item.id, asset_name)
        return
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchKey", "NotFound"):
            _log.warning("s2-cache HEAD error %s/%s: %s", item.id, asset_name, e)
            return  # unknown error -> fall through to public URL
    # MISS: stream from current (possibly SAS-signed) href into S3. `requests`
    # is used instead of an S3 CopyObject because MPC hrefs are Azure blobs
    # (not S3) and Earth Search public buckets refuse cross-account CopyObject.
    src = item.assets[asset_name].href
    try:
        import requests
        with requests.get(src, stream=True, timeout=120) as r:
            r.raise_for_status()
            s3.upload_fileobj(r.raw, bucket, key)
        item.assets[asset_name].href = f"s3://{bucket}/{key}"
        _log.info("s2-cache MISS seeded %s/%s", item.id, asset_name)
    except Exception as e:  # noqa: BLE001
        # Non-fatal: original href is untouched, odc-stac still reads via NAT.
        _log.warning(
            "s2-cache seed failed %s/%s: %s (falling back to public URL)",
            item.id, asset_name, e,
        )


def _s2_seed_and_rewrite_items(items, asset_names):
    """Route the given asset names on each item through the S3 cache, if enabled.

    No-op when FARMDAR_S2_CACHE_BUCKET is unset, so unmodified consumers keep
    the exact prior behaviour. Otherwise threads across items -- cache seeds
    are I/O-bound and parallelize well; concurrent races on the same key are
    fine (last write wins, wasted bandwidth only)."""
    bucket = _s2_cache_bucket()
    if not bucket or not items:
        return
    s3 = _s2_cache_client()
    # Small, bounded parallelism; per-item work is HTTP GET + S3 PUT.
    with ThreadPoolExecutor(max_workers=min(8, len(items) * len(asset_names) or 1)) as ex:
        futures = []
        for it in items:
            for asset in asset_names:
                if asset in it.assets:
                    futures.append(ex.submit(_s2_seed_and_rewrite_asset, it, asset, bucket, s3))
        for f in as_completed(futures):
            # Individual failures are already logged inside the worker; the
            # future.result() call surfaces any unexpected exception without
            # aborting the batch.
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                _log.warning("s2-cache worker crashed: %s", e)


def _composite_window(catalog, p, provider, geobox, bbox, start, end, cloud_lt, band_order):
    """Median composite of `band_order` over a window; returns (dict{band: ndarray}, prov).

    GEE parity: GEE filterDate(start, end) is half-open [start, end); a STAC
    "start/end" range is closed. Explicit midnight timestamps make the end
    exclusive so a scene on the window end date lands in the next window."""
    import numpy as np
    from odc.stac import load as odc_load

    scl_asset = p["scl_asset"]
    band_assets = _provider_assets(provider, band_order)

    search = catalog.search(collections=[p["collection"]], bbox=list(bbox),
                            datetime=f"{start}T00:00:00Z/{end}T00:00:00Z",
                            query={p["cloud_field"]: {"lt": cloud_lt}})
    items = list(search.items())
    ny, nx = geobox.shape
    prov = {"window_start": start, "window_end": end, "n_scenes": len(items),
            "scenes": [{"id": it.id, "datetime": it.properties.get("datetime"),
                        "cloud": it.properties.get(p["cloud_field"])} for it in items]}
    if not items:
        nan = np.full((ny, nx), np.nan, dtype="float32")
        prov["valid_pct"] = 0.0
        return {b: nan.copy() for b in band_order}, prov

    # Route asset reads through the S3 cache when FARMDAR_S2_CACHE_BUCKET is set.
    _s2_seed_and_rewrite_items(items, [*band_assets, scl_asset])

    # groupby="id" keeps every granule as its own observation (GEE medians each
    # scene separately); "solar_day" would merge same-day granules and shift it.
    ds = odc_load(items, bands=[*band_assets, scl_asset], geobox=geobox,
                  resampling="nearest", groupby="id", dtype="float32", chunks={})
    keep = _scl_keep_mask(ds[scl_asset])
    off = _harmonize_offset(p, items)
    out = {}
    ref_valid = None
    for b, asset in zip(band_order, band_assets):
        arr = ds[asset].where(keep).median("time", skipna=True).values
        arr = np.where(np.isnan(arr), np.nan, np.clip(arr - off, 0, None)).astype("float32")
        out[b] = arr
        if ref_valid is None:
            ref_valid = ~np.isnan(arr)
    prov["valid_pct"] = round(100 * float(np.mean(ref_valid)), 2)
    return out, prov


def _pick_provider(providers):
    for prov in providers:
        try:
            catalog, p = _open_catalog(prov)
            return catalog, p, prov
        except Exception as e:  # noqa: BLE001
            _log.warning("provider %s open failed: %s", prov, e)
    return None, None, None


# =============================================================================
# Public API
# =============================================================================
def fetch_sentinel_composite(aoi, start, end, bands=DEFAULT_BANDS, res_m=10,
                             cloud_lt=80, providers=("mpc", "earthsearch"),
                             out_path=None, max_pixels=40_000_000):
    """Fetch a single cloud-masked median Sentinel-2 composite over an AOI.

    The simplest entry point -- ideal for JupyterHub / interactive use. It builds
    ONE median composite of the requested `bands` over the whole date range in a
    single grid, and returns the band arrays in memory (optionally writing a
    multi-band GeoTIFF).

    Parameters
    ----------
    aoi : str | shapely geometry | GeoJSON dict | (minx, miny, maxx, maxy)
        Area of interest. File paths are read with geopandas (any OGR format).
    start, end : str
        "YYYY-MM-DD". `end` is treated as exclusive (GEE parity).
    bands : sequence[str], default ("blue", "green", "red", "nir")
        Band names to include. Friendly names, provider names, or raw Sxx codes
        (e.g. "nir", "NIR", "B8", "B08") -- see `SENTINEL2_BANDS`.
    res_m : int, default 10
        Output pixel size in metres.
    cloud_lt : int, default 80
        Keep only scenes with eo:cloud_cover < this before pixel-level SCL masking.
    providers : sequence[str], default ("mpc", "earthsearch")
        STAC catalogs to try, in order.
    out_path : str, optional
        If given, write a multi-band GeoTIFF (band order == `bands`) to this path.
    max_pixels : int, default 40M
        Guard against accidentally huge AOIs. Raise if width*height exceeds this.

    Returns
    -------
    dict with keys:
        "bands"       : {band_name: 2-D float32 ndarray (NaN = no data)}
        "band_order"  : list[str] canonical band names, matches array/file order
        "transform"   : affine.Affine of the output grid (EPSG:4326)
        "crs"         : "EPSG:4326"
        "shape"       : (height, width)
        "bounds"      : (minx, miny, maxx, maxy)
        "provider"    : which catalog served the data
        "provenance"  : scene ids / dates / cloud / valid_pct
        "out_path"    : path written (or None)

    Example
    -------
    >>> out = fetch_sentinel_composite("field.geojson", "2025-04-01", "2025-04-30")
    >>> red, nir = out["bands"]["red"], out["bands"]["nir"]
    >>> ndvi = (nir - red) / (nir + red)
    """
    import numpy as np
    from rasterio.transform import from_bounds
    from odc.geo.geobox import GeoBox

    _set_gdal_http_env()
    band_order = resolve_bands(bands)
    geom = _load_aoi_geometry(aoi)
    minx, miny, maxx, maxy = geom.bounds
    res_deg = res_m * _DEG_PER_M

    # snap bounds outward to the global res grid so pixels align across calls
    bx0, by0 = _snap(minx, res_deg), _snap(miny, res_deg)
    bx1 = bx0 + np.ceil((maxx - bx0) / res_deg) * res_deg
    by1 = by0 + np.ceil((maxy - by0) / res_deg) * res_deg
    nx = max(1, int(round((bx1 - bx0) / res_deg)))
    ny = max(1, int(round((by1 - by0) / res_deg)))
    if nx * ny > max_pixels:
        raise ValueError(
            f"AOI would produce {nx}x{ny}={nx * ny:,} px (> max_pixels={max_pixels:,}). "
            f"Use a coarser res_m, a smaller AOI, or fetch_sentinel_imagery() (tiled)."
        )

    geobox = GeoBox((ny, nx), from_bounds(bx0, by0, bx1, by1, nx, ny), _TARGET_CRS)
    catalog, p, provider = _pick_provider(providers)
    if catalog is None:
        raise RuntimeError(f"Could not open any STAC provider from {providers}.")

    bands_out, prov = _composite_window(
        catalog, p, provider, geobox, (bx0, by0, bx1, by1), start, end, cloud_lt, band_order)

    result = {
        "bands": bands_out,
        "band_order": band_order,
        "transform": geobox.transform,
        "crs": _TARGET_CRS,
        "shape": (ny, nx),
        "bounds": (bx0, by0, bx1, by1),
        "provider": provider,
        "provenance": prov,
        "out_path": None,
    }

    if out_path:
        import rasterio
        arrs = [np.nan_to_num(bands_out[b], nan=0.0).astype("float32") for b in band_order]
        profile = {"driver": "GTiff", "height": ny, "width": nx, "count": len(band_order),
                   "dtype": "float32", "crs": _TARGET_CRS, "transform": geobox.transform,
                   "nodata": 0, "compress": "deflate", "predictor": 2, "tiled": True,
                   "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, (b, arr) in enumerate(zip(band_order, arrs), 1):
                dst.write(arr, i)
                dst.set_band_description(i, b)
        result["out_path"] = out_path
        _log.info("wrote %s (%d bands, %dx%d, provider=%s)", out_path, len(band_order), nx, ny, provider)

    return result


def _process_tile(tile, start, end, step, cloud_lt, out_dir, providers, run_tag, band_order):
    """Build + write one tile's multi-band time-stack and manifest. Resume-safe."""
    import numpy as np
    import rasterio

    tid = tile["tile_id"]
    res_m = int(round(abs(tile["geobox"].transform[0]) / _DEG_PER_M))
    out_tif = _os.path.join(out_dir, f"sentinel_{res_m}m_tile_{tid:04d}.tif")
    out_man = out_tif.replace(".tif", ".manifest.json")
    if _os.path.exists(out_tif) and _os.path.exists(out_man):
        return {"tile_id": tid, "path": out_tif, "status": "skipped_exists"}

    catalog, p, provider = _pick_provider(providers)
    if catalog is None:
        return {"tile_id": tid, "status": "failed_no_provider"}

    gb = tile["geobox"]
    windows = list(_date_windows(start, end, step))
    ny, nx = gb.shape
    nb = len(band_order)
    stack = np.zeros((len(windows) * nb, ny, nx), dtype="float32")
    names, window_prov = [], []
    t0 = _time.time()
    for wi, (s, e) in enumerate(windows):
        bands_out, wp = _composite_window(catalog, p, provider, gb, tile["bbox"],
                                          s, e, cloud_lt, band_order)
        edate = e.replace("-", "_")
        for bi, b in enumerate(band_order):
            stack[wi * nb + bi] = bands_out[b]
            names.append(f"{b}_{edate}")
        window_prov.append(wp)

    out = np.clip(np.nan_to_num(stack, nan=0.0), 0, 65535).astype("uint16")
    bx0, by0, bx1, by1 = tile["bbox"]
    profile = {"driver": "GTiff", "height": ny, "width": nx, "count": out.shape[0],
               "dtype": "uint16", "crs": gb.crs, "transform": gb.transform,
               "compress": "deflate", "predictor": 2, "tiled": True,
               "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
    tmp = out_tif + ".tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        for i in range(out.shape[0]):
            dst.write(out[i], i + 1)
            dst.set_band_description(i + 1, names[i])
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
    _os.replace(tmp, out_tif)

    manifest = {
        "tile_id": tid, "run_tag": run_tag, "provider": provider,
        "crs": str(gb.crs), "bbox": [bx0, by0, bx1, by1], "shape": [ny, nx],
        "resolution_deg": (bx1 - bx0) / nx, "bands": names,
        "band_order": band_order, "date_range": [start, end], "step_days": step,
        "cloud_lt": cloud_lt, "harmonize_offset": p.get("harmonize_offset", 0),
        "windows": window_prov, "seconds": round(_time.time() - t0, 1),
    }
    with open(out_man, "w") as f:
        _json.dump(manifest, f, indent=2)
    return {"tile_id": tid, "path": out_tif, "status": "written",
            "seconds": manifest["seconds"], "provider": provider}


def _with_tile_retries(worker, tile, tile_retries, **kw):
    """Run `worker(tile, **kw)`, retrying transient (mostly network) failures
    within the run so a flaky connection doesn't permanently fail a tile. A failed
    attempt writes nothing (the tile GeoTIFF is only committed once all windows
    succeed), so each retry safely rebuilds the whole tile. Backoff grows
    3s,6s,12s,... up to 30s. After the attempts are exhausted the tile is left
    failed for a later resume run."""
    tid = tile["tile_id"]
    attempts = max(1, tile_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            r = worker(tile, **kw)
            if not str(r.get("status", "")).startswith("failed"):
                if attempt > 1:
                    r["attempts"] = attempt
                return r                          # success (written / skipped_exists)
            last = r.get("status")                # returned-failed (e.g. no provider)
        except Exception as e:  # noqa: BLE001 -- retry transient read/connection errors
            last = f"{type(e).__name__}: {e}"
        if attempt == attempts:
            _log.error("tile %04d failed after %d attempt(s): %s", tid, attempt, last)
            return {"tile_id": tid, "status": f"failed after {attempt} attempt(s)",
                    "attempts": attempt}
        delay = min(30, 3 * (2 ** (attempt - 1)))
        _log.warning("tile %04d attempt %d/%d failed (%s); retrying in %ds",
                     tid, attempt, attempts, last, delay)
        _time.sleep(delay)


def _build_vrt(out_dir, vrt_name="sentinel.vrt", pattern="sentinel_*m_tile_*.tif"):
    """Stitch all tiles into one seamless VRT (gdalbuildvrt if present, else exact
    XML fallback -- tiles are co-registered so the manual VRT is exact)."""
    import glob
    import subprocess
    tifs = sorted(glob.glob(_os.path.join(out_dir, pattern)))
    if not tifs:
        return None
    vrt = _os.path.join(out_dir, vrt_name)
    try:
        subprocess.run(["gdalbuildvrt", "-overwrite", vrt, *tifs], check=True,
                       capture_output=True)
        return vrt
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _build_vrt_manual(tifs, vrt)


def _build_vrt_manual(tifs, vrt):
    import xml.etree.ElementTree as ET
    import rasterio
    metas = []
    for t in tifs:
        with rasterio.open(t) as s:
            metas.append((t, s.bounds, s.transform, s.width, s.height, s.count,
                          s.crs, [s.dtypes[0]], s.descriptions, s.nodata))
    res = abs(metas[0][2][0])
    minx = min(m[1].left for m in metas); maxx = max(m[1].right for m in metas)
    miny = min(m[1].bottom for m in metas); maxy = max(m[1].top for m in metas)
    W = int(round((maxx - minx) / res)); H = int(round((maxy - miny) / res))
    nbands, crs, dtype, descs = metas[0][5], metas[0][6], metas[0][7][0], metas[0][8]
    nodata = metas[0][9]
    gdal_dt = {"uint16": "UInt16", "float32": "Float32", "uint8": "Byte"}.get(dtype, "UInt16")

    root = ET.Element("VRTDataset", rasterXSize=str(W), rasterYSize=str(H))
    ET.SubElement(root, "SRS").text = crs.to_wkt()
    ET.SubElement(root, "GeoTransform").text = f"{minx}, {res}, 0.0, {maxy}, 0.0, {-res}"
    for b in range(1, nbands + 1):
        vb = ET.SubElement(root, "VRTRasterBand", dataType=gdal_dt, band=str(b))
        if descs and descs[b - 1]:
            ET.SubElement(vb, "Description").text = descs[b - 1]
        if nodata is not None:   # carry the tiles' nodata so the VRT masks like they do
            ET.SubElement(vb, "NoDataValue").text = repr(nodata)
        for (path, bnds, _tr, w, h, _c, _crs, _dt, _d, _nd) in metas:
            xoff = int(round((bnds.left - minx) / res))
            yoff = int(round((maxy - bnds.top) / res))
            src = ET.SubElement(vb, "SimpleSource")
            ET.SubElement(src, "SourceFilename", relativeToVRT="1").text = _os.path.basename(path)
            ET.SubElement(src, "SourceBand").text = str(b)
            ET.SubElement(src, "SrcRect", xOff="0", yOff="0", xSize=str(w), ySize=str(h))
            ET.SubElement(src, "DstRect", xOff=str(xoff), yOff=str(yoff), xSize=str(w), ySize=str(h))
    ET.ElementTree(root).write(vrt, encoding="utf-8", xml_declaration=False)
    return vrt


def _clip_mosaic_to_geom(src_path, geom, clipped_path, all_touched=False, nodata=0):
    """Clip a rectangular mosaic (VRT or GeoTIFF) to the AOI polygon -> one
    standalone GeoTIFF with a sharp boundary (pixels outside set to nodata=0).
    Also materializes a virtual VRT into a real file.

    Returns the written path, or None if the clip failed -- it never raises,
    because a finished tile run must not be discarded just because this final
    convenience step could not complete.

    NOTE: rasterio reads the clipped extent into memory; for a very large AOI
    prefer the streaming `gdalwarp -cutline aoi -crop_to_cutline`."""
    import rasterio
    from rasterio.mask import mask as _rio_mask
    try:
        with rasterio.open(src_path) as src:
            out_img, out_tf = _rio_mask(src, [geom.__geo_interface__], crop=True,
                                        nodata=nodata, all_touched=all_touched, filled=True)
            descriptions = src.descriptions
            profile = src.profile.copy()
        profile.update(driver="GTiff", height=out_img.shape[1], width=out_img.shape[2],
                       transform=out_tf, nodata=nodata, compress="deflate", predictor=2,
                       tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        with rasterio.open(clipped_path, "w", **profile) as dst:
            dst.write(out_img)
            for _i, _desc in enumerate(descriptions, 1):
                if _desc:
                    dst.set_band_description(_i, _desc)
        _log.info("clipped mosaic -> %s (%d bands, %dx%d)", clipped_path,
                  out_img.shape[0], out_img.shape[2], out_img.shape[1])
        return clipped_path
    except Exception as e:  # noqa: BLE001 -- clipping must not discard a finished tile run
        _log.error("clip_to_aoi failed (%s); tiles + VRT are still available", e)
        return None


def fetch_sentinel_imagery(aoi, start, end, bands=DEFAULT_BANDS, out_dir=None,
                           step=8, res_m=10, tile_deg=0.1, cloud_lt=80, workers=8,
                           providers=("mpc", "earthsearch"), run_tag=None,
                           build_vrt_mosaic=True, clip_to_aoi=True,
                           clip_all_touched=False, tile_retries=2):
    """Build a tiled, seamless Sentinel-2 time-series baseline over any AOI.

    The full builder ported from the YieldPro `s2_baseline.py` DAG, made
    framework-independent (no Airflow / Celery / S3). Use this for large AOIs:
    the AOI is cut into global-grid-snapped tiles, each tile is composited per
    time window and written as a multi-band COG + provenance manifest, then all
    tiles are stitched into one seamless VRT. Threaded over tiles and
    resume-safe (existing tiles are skipped), so long runs survive crashes.

    For a quick single composite (one array in memory) use
    :func:`fetch_sentinel_composite` instead.

    Parameters
    ----------
    aoi : str | shapely geometry | GeoJSON dict | (minx, miny, maxx, maxy)
        Area of interest (file paths read with geopandas).
    start, end : str
        "YYYY-MM-DD". `end` exclusive (GEE parity).
    bands : sequence[str], default ("blue", "green", "red", "nir")
        Bands per time window (see `SENTINEL2_BANDS`). Output band count is
        len(bands) * n_windows.
    out_dir : str, optional
        Where tiles/manifests/VRT are written. Defaults to a fresh temp dir.
    step : int, default 8
        Compositing window length in days. step<=0 -> one window over [start, end).
    res_m : int, default 10
        Output pixel size in metres.
    tile_deg : float, default 0.1
        Tile edge length in degrees.
    cloud_lt : int, default 80
        Scene-level eo:cloud_cover cutoff.
    workers : int, default 8
        Threads over tiles (I/O-bound).
    providers : sequence[str], default ("mpc", "earthsearch")
        STAC catalogs to try per tile, in order.
    run_tag : str, optional
        Provenance tag stored in each manifest.
    build_vrt_mosaic : bool, default True
        Whether to stitch a seamless VRT after tiles are written.
    clip_to_aoi : bool, default True
        After stitching, clip the (rectangular) mosaic to the AOI polygon and
        write a single GeoTIFF `sentinel_clipped.tif` with a sharp boundary
        (pixels outside the polygon set to nodata=0). This also materializes the
        virtual VRT into a standalone file. For a bbox AOI it is effectively a
        crop to the bbox. NOTE: rasterio reads the clipped extent into memory --
        for a very large AOI this can be heavy; set False and use the streaming
        `gdalwarp -cutline aoi -crop_to_cutline` instead.
    clip_all_touched : bool, default False
        Edge rule for the clip. False -> crisp edge (pixel kept only if its centre
        is inside the polygon). True -> keep any pixel the boundary touches.
    tile_retries : int, default 2
        Extra attempts per tile within the run when a tile hits a transient
        (mostly network) error -- ConnectTimeout, CURL "could not connect",
        truncated/aborted reads. Each retry waits a growing backoff (3s, 6s, 12s,
        ... capped at 30s). Rides through flaky connections so a blip doesn't
        permanently fail a tile; anything still failing after the attempts is left
        for a later resume run. Set 0 to disable.

    Returns
    -------
    dict: {"out_dir", "tiles", "bands", "index_csv", "vrt", "clipped",
           "elapsed_min", "results"}  ("clipped" is None unless clip_to_aoi).
    """
    import tempfile

    _set_gdal_http_env()
    band_order = resolve_bands(bands)
    out_dir = out_dir or tempfile.mkdtemp(prefix="s2_sentinel_")
    _os.makedirs(out_dir, exist_ok=True)
    run_tag = run_tag or "run"
    res_deg = res_m * _DEG_PER_M

    geom = _load_aoi_geometry(aoi)
    tiles = _build_tiles(geom, tile_deg, res_deg)
    _log.info("AOI -> %d tiles @ %dm (tile=%.2fdeg, bands=%s, %d workers)",
              len(tiles), res_m, tile_deg, band_order, workers)

    results = []
    t0 = _time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_with_tile_retries, _process_tile, t, tile_retries,
                          start=start, end=end, step=step, cloud_lt=cloud_lt,
                          out_dir=out_dir, providers=providers, run_tag=run_tag,
                          band_order=band_order):
                t["tile_id"] for t in tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            tid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 -- one tile failing must not kill the run
                _log.error("tile %04d failed: %s", tid, e)
                r = {"tile_id": tid, "status": f"failed: {type(e).__name__}"}
            results.append(r)
            _log.info("[%d/%d] tile %04d: %s%s", i, len(tiles), r["tile_id"],
                      r["status"], f" ({r.get('seconds')}s)" if r.get("seconds") else "")

    index_csv = _os.path.join(out_dir, "index.csv")
    with open(index_csv, "w") as f:
        f.write("tile_id,status,provider,seconds,path\n")
        for r in sorted(results, key=lambda x: x["tile_id"]):
            f.write(f"{r['tile_id']},{r['status']},{r.get('provider', '')},"
                    f"{r.get('seconds', '')},{r.get('path', '')}\n")
    # VRT is needed both as the mosaic output and as the source for clipping.
    vrt = _build_vrt(out_dir) if (build_vrt_mosaic or clip_to_aoi) else None
    _log.info("DONE %d tiles in %.1f min. out_dir=%s vrt=%s",
              len(tiles), (_time.time() - t0) / 60, out_dir, vrt)

    # Clip the rectangular mosaic to the AOI polygon -> one GeoTIFF with a sharp
    # boundary (pixels outside the polygon set to nodata=0); also materializes the
    # virtual VRT into a standalone file. `geom` is the same AOI used for tiling.
    clipped = None
    if clip_to_aoi and vrt:
        clipped = _clip_mosaic_to_geom(vrt, geom, _os.path.join(out_dir, "sentinel_clipped.tif"),
                                       all_touched=clip_all_touched)

    return {"out_dir": out_dir, "tiles": len(tiles), "bands": band_order,
            "index_csv": index_csv, "vrt": vrt, "clipped": clipped,
            "elapsed_min": round((_time.time() - t0) / 60, 1), "results": results}


# =============================================================================
# Static imagery: single-acquisition ("cleanup") image over a date range
# =============================================================================
#
# The functions above build MEDIAN composites -- every clear pixel in a window is
# averaged. That is right for a time series, but wrong when you need one crisp,
# self-consistent snapshot (a classifier input, a base map, a cleanup image):
# medians blur field boundaries and mix acquisition dates pixel by pixel.
#
# `fetch_sentinel_static_*` instead picks REAL acquisition dates and layers them:
#
#   1. Score every date in [start, end) over the AOI -- how much of the AOI it
#      covers, and how much of that is cloud-free.
#   2. Pick the best date (the anchor) -- the single least-cloudy acquisition.
#   3. An AOI wider than one Sentinel-2 granule/orbit (a district like Sheikhupura
#      or Gujranwala) is never fully imaged on one date: consecutive relative
#      orbits image it days apart, so the anchor covers only ~80-90% of it. So
#      pick a 2nd date that fills what the anchor is missing while staying
#      temporally close (so crops look the same in both halves).
#   4. Mosaic them LAYERED: the best date on top, the next one only showing
#      through where the top has cloud/shadow/no-data -- and so on. Every output
#      pixel therefore comes from exactly ONE acquisition, all bands consistent.
#
# `mask_clouds=False` keeps step 4 from touching cloud: cloudy pixels stay exactly
# as observed and a lower layer only fills ground the layers above never imaged.
# That is what a classifier with a cloud/haze class wants -- swapping in another
# date's ground under a cloud would give it pixels that contradict their label.
# The dates are still chosen by cloud either way, so the anchor stays the
# least-cloudy acquisition.
#
# This is the open-source equivalent of the GEE `get_optimal_composite` pattern
# (anchor by lowest cloud, penalty = |days from anchor| + cloud/5, sorted mosaic),
# with two improvements available: the cloud/coverage score can be measured on the
# AOI itself rather than taken from whole-granule metadata, and the 2nd date can be
# chosen for how much NEW ground it adds instead of by penalty alone.
#
#     from farmdar.sentinel import fetch_sentinel_static_imagery
#     s = fetch_sentinel_static_imagery("sheikhupura.shp", "2025-05-01", "2025-05-16",
#                                       bands=["blue","green","red","rededge1","nir","ndvi"],
#                                       out_dir="/tmp/static")
#     s["dates"]        # ['2025-05-13', '2025-05-14']  (top, below)
#     s["clipped"]      # static_clipped.tif

# SCL classes rejected when building a static image. Stricter than
# `_scl_keep_mask` used by the median compositor: a single-date image has no other
# observation to fall back on, so no-data (0) and saturated/defective (1) must be
# treated as holes for the next date to fill, not silently kept.
#   0 no-data | 1 saturated/defective | 3 cloud shadow | 8 cloud medium prob
#   9 cloud high prob | 10 thin cirrus | 11 snow/ice
_SCL_DROP_STATIC = (0, 1, 3, 8, 9, 10, 11)

# Hole classes when `mask_clouds=False`: only pixels that carry no observation at
# all. Cloud, shadow, cirrus and snow are kept as real values -- what a classifier
# with a cloud/haze class needs, since substituting another date's ground there
# would feed it pixels that no longer match their label.
#   0 no-data | 1 saturated/defective
_SCL_DROP_NODATA = (0, 1)

# Normalized-difference indices available as virtual output bands: (a - b)/(a + b).
# Their inputs are fetched automatically even when not requested for output.
INDEX_BANDS = {
    "ndvi":  ("nir", "red"),        # vegetation
    "gndvi": ("nir", "green"),      # vegetation, green-based
    "ndre":  ("nir", "rededge1"),   # red-edge / chlorophyll
    "ndwi":  ("green", "nir"),      # open water
    "mndwi": ("green", "swir16"),   # open water, SWIR-based
    "ndmi":  ("nir", "swir16"),     # canopy moisture
    "nbr":   ("nir", "swir22"),     # burn ratio
}


def _norm_band(b):
    return str(b).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


@_contextlib.contextmanager
def _quiet_rasterio():
    """Silence rasterio's NotGeoreferencedWarning around odc-stac loads.

    odc-stac warps through intermediate in-memory arrays that carry no geotransform
    of their own; rasterio warns about it even though the geobox we pass is what
    actually georeferences the result. Nothing else is suppressed."""
    import warnings
    try:
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError:  # pragma: no cover -- very old rasterio
        yield
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        yield


def resolve_static_bands(bands):
    """Split a requested band list into what must be downloaded vs computed.

    Accepts everything `resolve_bands` accepts plus the virtual index bands in
    `INDEX_BANDS` (``"ndvi"``, ``"ndmi"``, ...). Index inputs are added to the
    fetch list automatically even if the caller does not want them in the output.

    Returns
    -------
    (fetch_bands, index_bands, output_order)
        fetch_bands  : canonical spectral bands to download (fetch order)
        index_bands  : requested index names
        output_order : the caller's bands, de-duplicated, in the requested order
                       -- this is the output band order.

    >>> resolve_static_bands(["B4", "nir", "ndvi"])
    (['red', 'nir'], ['ndvi'], ['red', 'nir', 'ndvi'])
    """
    if isinstance(bands, str):
        bands = [bands]
    output_order, index_bands, spectral, seen = [], [], [], set()
    for b in bands:
        key = _norm_band(b)
        if key in INDEX_BANDS:
            if key not in seen:
                seen.add(key)
                output_order.append(key)
                index_bands.append(key)
        else:
            canon = resolve_bands([b])[0]
            if canon not in seen:
                seen.add(canon)
                output_order.append(canon)
                spectral.append(canon)
    if not output_order:
        raise ValueError("`bands` resolved to an empty list.")
    fetch_bands = list(spectral)
    for idx in index_bands:
        for need in INDEX_BANDS[idx]:
            if need not in fetch_bands:
                fetch_bands.append(need)   # required input, not necessarily output
    return fetch_bands, index_bands, output_order


def _solar_day(item):
    """Local ("solar") acquisition day of a STAC item, as 'YYYY-MM-DD'.

    Matches odc-stac's `groupby="solar_day"`: UTC shifted by longitude/15 hours.
    Granules of one overpass are then guaranteed to share a day label even when
    the swath straddles UTC midnight, so grouping here and loading with
    groupby="solar_day" always agree."""
    from shapely.geometry import shape
    try:
        lon = shape(item.geometry).centroid.x
    except Exception:  # noqa: BLE001 -- footprint missing/invalid; UTC day is close enough
        lon = 0.0
    return (item.datetime + timedelta(hours=lon / 15.0)).strftime("%Y-%m-%d")


def _search_items(catalog, p, bbox, start, end, cloud_lt):
    """All scenes intersecting bbox in [start, end) below the cloud cutoff.
    `end` is exclusive (GEE `filterDate` parity)."""
    search = catalog.search(collections=[p["collection"]], bbox=list(bbox),
                            datetime=f"{start}T00:00:00Z/{end}T00:00:00Z",
                            query={p["cloud_field"]: {"lt": cloud_lt}})
    return list(search.items())


def _items_for_solar_day(catalog, p, bbox, day, cloud_lt):
    """Scenes belonging to one solar day. Searched with a +/-1 day UTC pad and then
    filtered by `_solar_day`, so a granule whose UTC date differs from its solar
    date is still picked up. Re-searched per tile so signed asset URLs stay fresh
    on long runs."""
    d0 = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    d1 = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")
    return [it for it in _search_items(catalog, p, bbox, d0, d1, cloud_lt)
            if _solar_day(it) == day]


def _scan_geobox(geom, scan_res_m, max_scan_pixels, target_pixels=250_000, floor_res_m=20.0):
    """Coarse grid over the AOI, used only for scoring dates and estimating layer
    matching -- neither needs full resolution.

    `scan_res_m=None` (the default) sizes the grid to the AOI: roughly
    `target_pixels` cells, never finer than `floor_res_m`. A fixed resolution would
    be wrong at both ends -- 200 m gives a district a sensible ~250k-cell grid but
    leaves a single field with a couple of dozen cells, far too few to measure cloud
    cover or fit a radiometric correction on. Always coarsens further if the result
    would exceed `max_scan_pixels`."""
    import numpy as np
    from rasterio.transform import from_bounds
    from odc.geo.geobox import GeoBox

    minx, miny, maxx, maxy = geom.bounds
    if scan_res_m is None:
        area_m2 = ((maxx - minx) / _DEG_PER_M) * ((maxy - miny) / _DEG_PER_M)
        res_m = max(floor_res_m, float(np.sqrt(max(area_m2, 1.0) / target_pixels)))
    else:
        res_m = float(scan_res_m)
    while True:
        res_deg = res_m * _DEG_PER_M
        bx0, by0 = _snap(minx, res_deg), _snap(miny, res_deg)
        bx1 = bx0 + np.ceil((maxx - bx0) / res_deg) * res_deg
        by1 = by0 + np.ceil((maxy - by0) / res_deg) * res_deg
        nx = max(1, int(round((bx1 - bx0) / res_deg)))
        ny = max(1, int(round((by1 - by0) / res_deg)))
        if nx * ny <= max_scan_pixels:
            break
        res_m *= 2
        _log.info("scan grid too large; coarsening scan_res_m -> %dm", int(res_m))
    return (GeoBox((ny, nx), from_bounds(bx0, by0, bx1, by1, nx, ny), _TARGET_CRS),
            (bx0, by0, bx1, by1), int(res_m))


def _scan_date_masks(catalog, p, provider, geobox, bbox, by_day, cloud_lt, scl_drop,
                     cloud_metric, workers):
    """Per-date boolean 'usable pixel' masks on the coarse scan grid.

    cloud_metric='aoi'   -- read the SCL band for each date and mark pixels that are
                            cloud/shadow/snow/no-data free. Costs a few small reads
                            but measures the AOI itself, so partial-granule coverage
                            and cloud are captured together.
    cloud_metric='scene' -- no reads at all: rasterize the granule footprints. Cloud
                            comes from `eo:cloud_cover` metadata (whole-granule, the
                            GEE `CLOUDY_PIXEL_PERCENTAGE` equivalent).

    Returns {date: (usable_mask, footprint_mask)}."""
    import numpy as np

    shape_ = geobox.shape
    if cloud_metric == "scene":
        from rasterio.features import geometry_mask
        from shapely.geometry import shape as _shape
        out = {}
        for day, items in by_day.items():
            geoms = [_shape(it.geometry) for it in items]
            foot = ~geometry_mask(geoms, out_shape=shape_, transform=geobox.transform,
                                  all_touched=True)
            out[day] = (foot, foot)   # cloud is metadata-only in this mode
        return out

    from odc.stac import load as odc_load
    scl_asset = p["scl_asset"]

    def one(day):
        items = by_day[day]
        with _quiet_rasterio():
            ds = odc_load(items, bands=[scl_asset], geobox=geobox, resampling="nearest",
                          groupby="solar_day", dtype="float32", chunks={})
            scl = ds[scl_asset]
            scl = scl.isel(time=0).values if "time" in scl.dims else scl.values
        foot = np.isfinite(scl) & (scl != 0)
        usable = foot & ~np.isin(scl, scl_drop)
        return day, usable, foot

    out = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, d): d for d in by_day}
        for fut in as_completed(futs):
            try:
                day, usable, foot = fut.result()
            except Exception as e:  # noqa: BLE001 -- a date we cannot read is simply not a candidate
                _log.warning("scan failed for %s (%s); dropping it from the candidates",
                             futs[fut], e)
                continue
            out[day] = (usable, foot)
    return out


def select_static_dates(aoi, start, end, n_dates=2, cloud_lt=80,
                        providers=("mpc", "earthsearch"), cloud_metric="aoi",
                        selection_mode="greedy", scan_res_m=None,
                        max_scan_pixels=4_000_000, max_day_gap=None,
                        target_coverage=99.0, min_gain=0.5, cloud_penalty_divisor=5.0,
                        anchor_tolerance=1.0, coverage_basis="usable",
                        scl_drop=_SCL_DROP_STATIC, workers=8):
    """Choose the acquisition date(s) that make the best static image of an AOI.

    Every date in the range is scored over the AOI, the least-cloudy one becomes
    the anchor (layer 1, on top), and further dates are added only to fill what the
    anchor is missing -- cloud holes, and the parts of a multi-granule AOI the
    anchor's orbit never imaged.

    Parameters
    ----------
    aoi : str | shapely geometry | GeoJSON dict | (minx, miny, maxx, maxy)
    start, end : str
        "YYYY-MM-DD". `end` exclusive (GEE parity).
    n_dates : int, default 2
        Maximum number of dates to layer. 1 = strict single-date image.
    cloud_lt : int, default 80
        Drop scenes whose `eo:cloud_cover` is above this before scoring.
    cloud_metric : {"aoi", "scene"}, default "aoi"
        "aoi" measures cloud + coverage on the AOI from the SCL band (accurate;
        costs a coarse read per date). "scene" uses granule metadata and footprint
        geometry only (instant; matches GEE's CLOUDY_PIXEL_PERCENTAGE behaviour).
    selection_mode : {"greedy", "penalty"}, default "greedy"
        "greedy"  -- anchor = most usable AOI pixels, then each further date is the
                     one adding the most NEW ground, ties broken toward the
                     temporally closest date. Best for AOIs spanning several
                     granules/orbits.
        "penalty" -- GEE parity: anchor = lowest cloud, then rank every other date
                     by |days from anchor| + cloud_pct / `cloud_penalty_divisor`.
    scan_res_m : int, optional
        Resolution of the scoring grid (not the output). Default None = sized to the
        AOI (~250k cells, never finer than 20 m).
    max_day_gap : int, optional
        Reject dates more than this many days from the anchor -- the hard version of
        "the two images must not be far apart".
    target_coverage : float, default 99.0
        Stop adding dates once this % of the AOI is usable.
    min_gain : float, default 0.5
        Stop when the best remaining date would add less than this % of new AOI.
    cloud_penalty_divisor : float, default 5.0
        GEE's tunable: 5.0 means 5% cloud costs as much as being 1 day away.
    anchor_tolerance : float, default 1.0
        Greedy mode only. Dates whose usable AOI % is within this much of the best
        are treated as tied for anchor and the least-cloudy of them wins, rather
        than letting a 0.1% coverage difference decide (and drag the partner date
        further away in time).
    coverage_basis : {"usable", "footprint"}, default "usable"
        What a further date has to contribute to earn its place -- and therefore what
        `coverage_pct` counts. "usable" = cloud-free AOI ground, for a mosaic that
        masks cloud. "footprint" = imaged ground regardless of cloud, for a mosaic
        built with `mask_clouds=False`, where a lower layer only ever fills what the
        anchor's orbit did not reach. The anchor is picked on cloud either way, so
        you still get the least-cloudy image.

    Returns
    -------
    dict with keys:
        "dates"        : selected dates, BEST FIRST -- this is the layering order
        "anchor"       : the anchor date (== dates[0])
        "coverage_pct" : % of the AOI usable after layering the selected dates
        "selected"     : per-layer detail (rank, date, gain_pct, cloud_pct, ...)
        "candidates"   : every scored date, ranked by penalty
        "provider", "cloud_metric", "selection_mode", "scan_res_m"

    Example
    -------
    >>> sel = select_static_dates("sheikhupura.shp", "2025-05-01", "2025-05-16")
    >>> sel["dates"], round(sel["coverage_pct"], 1)
    (['2025-05-13', '2025-05-14'], 100.0)
    """
    import numpy as np
    from rasterio.features import geometry_mask

    if cloud_metric not in ("aoi", "scene"):
        raise ValueError("cloud_metric must be 'aoi' or 'scene'")
    if selection_mode not in ("greedy", "penalty"):
        raise ValueError("selection_mode must be 'greedy' or 'penalty'")
    if coverage_basis not in ("usable", "footprint"):
        raise ValueError("coverage_basis must be 'usable' or 'footprint'")
    basis = 0 if coverage_basis == "usable" else 1   # index into (usable, footprint)

    _set_gdal_http_env()
    geom = _load_aoi_geometry(aoi)
    catalog, p, provider = _pick_provider(providers)
    if catalog is None:
        raise RuntimeError(f"Could not open any STAC provider from {providers}.")

    items = _search_items(catalog, p, geom.bounds, start, end, cloud_lt)
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes over the AOI in [{start}, {end}) with cloud < {cloud_lt}%.")
    by_day = {}
    for it in items:
        by_day.setdefault(_solar_day(it), []).append(it)
    _log.info("static: %d scenes on %d dates in [%s, %s)", len(items), len(by_day), start, end)

    geobox, bbox, used_res = _scan_geobox(geom, scan_res_m, max_scan_pixels)
    aoi_mask = ~geometry_mask([geom.__geo_interface__], out_shape=geobox.shape,
                              transform=geobox.transform, all_touched=True)
    aoi_px = int(aoi_mask.sum())
    if aoi_px == 0:   # AOI thinner than one scan pixel -> fall back to the bbox
        aoi_mask = np.ones(geobox.shape, bool)
        aoi_px = aoi_mask.size

    masks = _scan_date_masks(catalog, p, provider, geobox, bbox, by_day, cloud_lt,
                             scl_drop, cloud_metric, workers)
    if not masks:
        raise RuntimeError("Could not score any candidate date (all scans failed).")

    def pct(mask):
        return 100.0 * float(np.count_nonzero(mask & aoi_mask)) / aoi_px

    cand = {}
    for day, (usable, foot) in masks.items():
        its = by_day[day]
        cov = pct(foot)
        valid = pct(usable)
        if cloud_metric == "aoi":
            cloud = round(100.0 * (1.0 - valid / cov), 2) if cov > 0 else 100.0
        else:  # granule metadata; usable == footprint so derive valid from it
            cloud = round(sum(i.properties.get(p["cloud_field"], 100) for i in its) / len(its), 2)
            valid = round(cov * (1.0 - cloud / 100.0), 2)
        cand[day] = {"date": day, "n_scenes": len(its), "coverage_pct": round(cov, 2),
                     "valid_pct": round(valid, 2), "cloud_pct": cloud}

    def as_day(d):
        return datetime.strptime(d, "%Y-%m-%d")

    if selection_mode == "penalty":   # GEE: sort('CLOUD_COVER_STD').first()
        anchor = min(cand, key=lambda d: (cand[d]["cloud_pct"], -cand[d]["valid_pct"], d))
    else:
        # Most usable AOI ground wins -- but dates within `anchor_tolerance` % of the
        # best are a photo finish (same orbit, one revisit apart), so let the cleaner
        # one anchor. That also tends to pull the whole selection tighter in time:
        # the runner-up date is chosen relative to the anchor, so a marginally
        # "bigger" anchor is a bad trade if it pushes its partner days further away.
        best_valid = max(c["valid_pct"] for c in cand.values())
        near = [d for d, c in cand.items() if c["valid_pct"] >= best_valid - anchor_tolerance]
        anchor = min(near, key=lambda d: (cand[d]["cloud_pct"], -cand[d]["valid_pct"], d))
    a_day = as_day(anchor)
    for d, c in cand.items():
        c["days_from_anchor"] = abs((as_day(d) - a_day).days)
        c["penalty"] = round(c["days_from_anchor"] + c["cloud_pct"] / cloud_penalty_divisor, 3)

    eligible = [d for d in cand if d != anchor
                and (max_day_gap is None or cand[d]["days_from_anchor"] <= max_day_gap)]

    # What the lower layers are actually there to fill: cloud-free ground when the
    # mosaic masks clouds, bare granule footprint when it keeps them.
    covered = masks[anchor][basis] & aoi_mask
    chosen = [dict(cand[anchor], rank=1, gain_pct=round(pct(covered), 2))]
    while len(chosen) < n_dates and eligible:
        if pct(covered) >= target_coverage:
            break
        if selection_mode == "penalty":
            nxt = min(eligible, key=lambda d: (cand[d]["penalty"], d))
        else:
            # most NEW ground first; near-ties (0.1%) go to the closest date in time
            nxt = max(eligible, key=lambda d: (round(pct(masks[d][basis] & ~covered), 1),
                                               -cand[d]["penalty"]))
        gain = pct(masks[nxt][basis] & ~covered)
        if selection_mode == "greedy" and gain < min_gain:
            _log.info("static: stopping -- best remaining date %s adds only %.2f%%", nxt, gain)
            break
        eligible.remove(nxt)
        covered = covered | (masks[nxt][basis] & aoi_mask)
        chosen.append(dict(cand[nxt], rank=len(chosen) + 1, gain_pct=round(gain, 2)))

    coverage = round(pct(covered), 2)
    dates = [c["date"] for c in chosen]
    _log.info("static: dates=%s (anchor=%s) -> %.2f%% of AOI usable", dates, anchor, coverage)
    if coverage < target_coverage and len(chosen) >= n_dates:
        _log.warning("static: %.2f%% AOI coverage with %d date(s); raise n_dates to fill the rest",
                     coverage, n_dates)

    return {"dates": dates, "anchor": anchor, "coverage_pct": coverage, "selected": chosen,
            "candidates": sorted(cand.values(), key=lambda c: c["penalty"]),
            "provider": provider, "cloud_metric": cloud_metric,
            "selection_mode": selection_mode, "coverage_basis": coverage_basis,
            "scan_res_m": used_res,
            "n_candidate_dates": len(cand), "date_range": [start, end]}


def _robust_match(x, y, method, gain_limits):
    """Coefficients (gain, offset) mapping values `x` onto reference `y`.

    Uses medians and MADs, not least squares: regressing one date on another has an
    r well below 1 (real ground change, different view/sun angle), so an OLS slope
    is attenuated toward zero and would visibly flatten the corrected layer's
    contrast. Matching the median and the spread instead preserves contrast and
    only removes the systematic brightness/haze difference."""
    import numpy as np
    lo_x, hi_x = np.percentile(x, [1, 99])
    lo_y, hi_y = np.percentile(y, [1, 99])
    keep = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    if keep.sum() < 100:
        keep = np.ones_like(x, dtype=bool)
    x, y = x[keep], y[keep]
    mx, my = float(np.median(x)), float(np.median(y))
    if method == "median":
        return 1.0, my - mx
    sx = float(np.median(np.abs(x - mx)))
    sy = float(np.median(np.abs(y - my)))
    gain = float(np.clip(sy / sx, *gain_limits)) if sx > 0 else 1.0
    return gain, my - gain * mx


def _layer_match_coeffs(catalog, p, provider, geom, dates, fetch_bands, cloud_lt, scl_drop,
                        method="linear", scan_res_m=None, max_scan_pixels=4_000_000,
                        min_overlap_frac=0.02, min_overlap_px=400,
                        gain_limits=(0.5, 2.0), workers=8):
    """Per-band radiometry correction bringing each lower layer onto the top layer.

    Two acquisitions days apart differ in haze, sun angle and view angle -- over
    Sheikhupura the second date runs ~190 DN darker in every band -- so a layered
    mosaic shows a visible tonal seam along the orbit boundary even though both
    dates are cloud-free. Where the dates overlap (usually most of the AOI, since
    consecutive orbits overlap heavily) that difference can be measured directly and
    removed from the lower layer.

    Estimated ONCE for the whole AOI on a coarse grid and then applied identically
    by every tile -- computing it per tile would make tiles disagree and trade an
    orbit seam for tile seams.

    Returns {date: {band: (gain, offset)}} for the lower layers only; dates with too
    little overlap are simply absent (left uncorrected)."""
    import numpy as np
    from odc.stac import load as odc_load
    from rasterio.features import geometry_mask

    if len(dates) < 2:
        return {}
    geobox, bbox, _res = _scan_geobox(geom, scan_res_m, max_scan_pixels)
    aoi_mask = ~geometry_mask([geom.__geo_interface__], out_shape=geobox.shape,
                              transform=geobox.transform, all_touched=True)
    # Enough overlap to fit on: a fraction of the AOI, with an absolute floor so a
    # tiny AOI is not calibrated from a handful of pixels.
    min_overlap = max(min_overlap_px, int(min_overlap_frac * int(aoi_mask.sum())))
    assets = _provider_assets(provider, fetch_bands)
    scl_asset = p["scl_asset"]

    def load(day):
        items = _items_for_solar_day(catalog, p, bbox, day, cloud_lt)
        if not items:
            return day, None, None
        with _quiet_rasterio():
            ds = odc_load(items, bands=[*assets, scl_asset], geobox=geobox,
                          resampling="nearest", groupby="solar_day", dtype="float32",
                          chunks={})
            if "time" in ds.dims:
                ds = ds.isel(time=0)
            scl = ds[scl_asset].values
            arrs = {b: ds[a].values.astype("float32", copy=False)
                    for b, a in zip(fetch_bands, assets)}
        ok = np.isfinite(scl) & (scl != 0) & ~np.isin(scl, scl_drop) & aoi_mask
        off = _harmonize_offset(p, items)
        for b in fetch_bands:
            ok &= np.isfinite(arrs[b])
            arrs[b] = np.clip(arrs[b] - off, 0, None)
        return day, arrs, ok

    loaded = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(load, d) for d in dates]):
            try:
                day, arrs, ok = fut.result()
            except Exception as e:  # noqa: BLE001 -- no calibration is better than a bad one
                _log.warning("layer matching: scan failed (%s); that layer stays uncorrected", e)
                continue
            if arrs is not None:
                loaded[day] = (arrs, ok)

    top = dates[0]
    if top not in loaded:
        _log.warning("layer matching: top layer %s unreadable; skipping matching", top)
        return {}
    coeffs = {}
    for day in dates[1:]:
        if day not in loaded:
            continue
        # Reference is the top layer; if this layer barely overlaps it, fall back to
        # the nearest already-corrected layer above it.
        for ref in [top] + [d for d in dates[1:] if d != day and d in coeffs]:
            both = loaded[day][1] & loaded[ref][1]
            n = int(np.count_nonzero(both))
            if n >= min_overlap:
                break
        if n < min_overlap:
            _log.warning("layer matching: %s overlaps the layers above it on only %d px "
                         "(< %d); leaving it uncorrected", day, n, min_overlap)
            continue
        c = {}
        for b in fetch_bands:
            x = loaded[day][0][b][both]
            y = loaded[ref][0][b][both]
            if ref != top and ref in coeffs:          # compare against corrected values
                g0, o0 = coeffs[ref][b]
                y = y * g0 + o0
            c[b] = _robust_match(x, y, method, gain_limits)
        coeffs[day] = c
        _log.info("layer matching: %s -> %s on %d px: %s", day, ref, n,
                  {b: (round(g, 3), round(o, 1)) for b, (g, o) in c.items()})
    return coeffs


def _static_layers(catalog, p, provider, geobox, bbox, dates, fetch_bands, index_bands,
                   output_order, cloud_lt, scl_drop, match_coeffs=None):
    """Layered single-date mosaic over one geobox: `dates[0]` on top.

    A pixel is taken from the first date in `dates` that has a usable observation
    there, so all of its bands come from the SAME acquisition -- no spectral mixing
    between dates, which is the whole point of a static image.

    Returns (bands dict {name: float32 array, NaN = no data}, source_rank uint8
    array where 0 = never filled, provenance list)."""
    import numpy as np
    from odc.stac import load as odc_load

    ny, nx = geobox.shape
    assets = _provider_assets(provider, fetch_bands)
    scl_asset = p["scl_asset"]

    stack = {b: np.full((ny, nx), np.nan, dtype="float32") for b in fetch_bands}
    filled = np.zeros((ny, nx), dtype=bool)
    source = np.zeros((ny, nx), dtype="uint8")
    prov = []

    for rank, day in enumerate(dates, 1):
        items = _items_for_solar_day(catalog, p, bbox, day, cloud_lt)
        if not items:
            prov.append({"rank": rank, "date": day, "n_scenes": 0, "added_pct": 0.0})
            continue
        # Route asset reads through the S3 cache when FARMDAR_S2_CACHE_BUCKET is set.
        _s2_seed_and_rewrite_items(items, [*assets, scl_asset])
        with _quiet_rasterio():
            ds = odc_load(items, bands=[*assets, scl_asset], geobox=geobox,
                          resampling="nearest", groupby="solar_day", dtype="float32",
                          chunks={})
            if "time" in ds.dims:
                ds = ds.isel(time=0)   # all items share a solar day -> exactly one slice

            scl = ds[scl_asset].values
            raw = {b: ds[a].values.astype("float32", copy=False)
                   for b, a in zip(fetch_bands, assets)}

        usable = np.isfinite(scl) & (scl != 0) & ~np.isin(scl, scl_drop)
        for b in fetch_bands:
            usable &= np.isfinite(raw[b])   # a band with no data is a hole, not a value

        take = usable & ~filled
        off = _harmonize_offset(p, items)
        match = (match_coeffs or {}).get(day)
        for b in fetch_bands:
            vals = np.clip(raw[b][take] - off, 0, None)
            if match and b in match:                  # put this layer on the top layer's scale
                g, o = match[b]
                vals = np.clip(vals * g + o, 0, None)
            stack[b][take] = vals
        source[take] = rank
        filled |= take
        prov.append({"rank": rank, "date": day, "n_scenes": len(items),
                     "added_pct": round(100.0 * float(np.count_nonzero(take)) / (ny * nx), 2),
                     "harmonize_offset": off,
                     "match": {b: [round(g, 4), round(o, 2)] for b, (g, o) in match.items()}
                     if match else None})

    out = {}
    for b in output_order:
        if b in INDEX_BANDS:
            hi, lo = (stack[x] for x in INDEX_BANDS[b])
            with np.errstate(divide="ignore", invalid="ignore"):
                out[b] = ((hi - lo) / (hi + lo)).astype("float32")
        else:
            out[b] = stack[b]
    return out, source, prov


def _resolve_static_dates(aoi_or_geom, start, end, dates, sel_kwargs):
    """Return (dates, selection dict). `dates` given -> manual mode (no scoring),
    mirroring the GEE notebook's manual_top_date / manual_bottom_date override."""
    if dates:
        if isinstance(dates, str):
            dates = [dates]
        dates = list(dates)
        return dates, {"dates": dates, "anchor": dates[0], "selection_mode": "manual",
                       "coverage_pct": None, "selected": [
                           {"rank": i, "date": d} for i, d in enumerate(dates, 1)]}
    sel = select_static_dates(aoi_or_geom, start, end, **sel_kwargs)
    return sel["dates"], sel


def fetch_sentinel_static_composite(aoi, start, end, bands=DEFAULT_BANDS, res_m=10,
                                    cloud_lt=80, providers=("mpc", "earthsearch"),
                                    out_path=None, max_pixels=40_000_000, dates=None,
                                    n_dates=2, mask_clouds=True, scl_drop=_SCL_DROP_STATIC,
                                    add_source_band=False, match_layers="median",
                                    **selection_kwargs):
    """Static single-acquisition Sentinel-2 image over a small AOI, in memory.

    The static counterpart of :func:`fetch_sentinel_composite`: instead of a median
    over the window it returns the least-cloudy real acquisition, with further
    dates layered underneath only where that one has holes. Ideal for interactive
    / notebook use; use :func:`fetch_sentinel_static_imagery` for large AOIs.

    Parameters
    ----------
    aoi : str | shapely geometry | GeoJSON dict | (minx, miny, maxx, maxy)
    start, end : str
        "YYYY-MM-DD", `end` exclusive.
    bands : sequence[str], default ("blue", "green", "red", "nir")
        Spectral bands and/or index bands (`INDEX_BANDS`: "ndvi", "ndmi", ...).
        Index inputs are downloaded automatically.
    res_m : int, default 10
    cloud_lt : int, default 80
    out_path : str, optional
        Write a float32 multi-band GeoTIFF (band order == `bands`, NaN = no data).
    max_pixels : int, default 40M
        Guard against accidentally huge AOIs.
    dates : sequence[str], optional
        Skip date selection and layer exactly these dates, first = top.
    n_dates : int, default 2
        Max dates to layer when selecting automatically.
    mask_clouds : bool, default True
        True  -- cloud, shadow, cirrus and snow in a layer are holes, filled from the
                 next date. Gives the cleanest picture of the ground.
        False -- those pixels are kept exactly as observed, and a lower layer only
                 fills what the layers above never imaged (outside their granule
                 footprint). Use this when the image feeds a classifier that has a
                 cloud/haze class: substituting another date's ground under a cloud
                 hands the model pixels that no longer match their label, and mixes
                 two acquisitions inside one scene.
        Either way the dates themselves are still chosen by cloud, so the anchor is
        the least-cloudy acquisition available.
    add_source_band : bool, default False
        Append a "source_rank" band: which layer (1 = top) each pixel came from,
        0 = no usable observation. Useful to QA the mosaic.
    match_layers : {"median", "linear", None}, default "median"
        Put the lower layers on the top layer's radiometry, measured where they
        overlap, so the mosaic has no tonal seam. "median" shifts brightness only
        (the difference between two nearby dates is mostly additive haze, so this
        removes ~85% of the seam and cannot distort contrast). "linear" also matches
        the spread -- slightly better on some bands, but it can compress the lower
        layer's contrast by ~10%. None disables matching (GEE parity).
    **selection_kwargs
        Passed to :func:`select_static_dates` (`cloud_metric`, `selection_mode`,
        `max_day_gap`, `target_coverage`, `min_gain`, `scan_res_m`, ...).

    Returns
    -------
    dict: {"bands", "band_order", "source_rank", "dates", "selection", "transform",
           "crs", "shape", "bounds", "provider", "provenance", "out_path"}

    Example
    -------
    >>> s = fetch_sentinel_static_composite("field.geojson", "2025-05-01", "2025-05-16",
    ...                                     bands=["red", "nir", "ndvi"])
    >>> s["dates"], s["bands"]["ndvi"].shape
    """
    import numpy as np
    from rasterio.transform import from_bounds
    from odc.geo.geobox import GeoBox

    _set_gdal_http_env()
    fetch_bands, index_bands, output_order = resolve_static_bands(bands)
    geom = _load_aoi_geometry(aoi)
    minx, miny, maxx, maxy = geom.bounds
    res_deg = res_m * _DEG_PER_M

    bx0, by0 = _snap(minx, res_deg), _snap(miny, res_deg)
    bx1 = bx0 + np.ceil((maxx - bx0) / res_deg) * res_deg
    by1 = by0 + np.ceil((maxy - by0) / res_deg) * res_deg
    nx = max(1, int(round((bx1 - bx0) / res_deg)))
    ny = max(1, int(round((by1 - by0) / res_deg)))
    if nx * ny > max_pixels:
        raise ValueError(
            f"AOI would produce {nx}x{ny}={nx * ny:,} px (> max_pixels={max_pixels:,}). "
            f"Use a coarser res_m, a smaller AOI, or fetch_sentinel_static_imagery() (tiled).")

    # Cloud always decides WHICH dates are used; `mask_clouds` decides whether it also
    # decides which PIXELS a lower layer may replace.
    layer_drop = tuple(scl_drop) if mask_clouds else _SCL_DROP_NODATA
    sel_kwargs = dict(selection_kwargs)
    sel_kwargs.setdefault("n_dates", n_dates)
    sel_kwargs.setdefault("cloud_lt", cloud_lt)
    sel_kwargs.setdefault("providers", providers)
    sel_kwargs.setdefault("scl_drop", scl_drop)
    sel_kwargs.setdefault("coverage_basis", "usable" if mask_clouds else "footprint")
    dates, selection = _resolve_static_dates(geom, start, end, dates, sel_kwargs)

    geobox = GeoBox((ny, nx), from_bounds(bx0, by0, bx1, by1, nx, ny), _TARGET_CRS)
    catalog, p, provider = _pick_provider(providers)
    if catalog is None:
        raise RuntimeError(f"Could not open any STAC provider from {providers}.")

    # `scl_drop` (not layer_drop) here on purpose: the radiometric offset must never be
    # fitted over cloudy pixels, even when the mosaic keeps them.
    match_coeffs = (_layer_match_coeffs(catalog, p, provider, geom, dates, fetch_bands,
                                        cloud_lt, scl_drop, method=match_layers)
                    if match_layers and len(dates) > 1 else {})

    bands_out, source, prov = _static_layers(catalog, p, provider, geobox,
                                             (bx0, by0, bx1, by1), dates, fetch_bands,
                                             index_bands, output_order, cloud_lt, layer_drop,
                                             match_coeffs=match_coeffs)
    filled_pct = round(100.0 * float(np.count_nonzero(source)) / source.size, 2)
    result = {
        "bands": bands_out, "band_order": output_order, "source_rank": source,
        "dates": dates, "selection": selection, "transform": geobox.transform,
        "crs": _TARGET_CRS, "shape": (ny, nx), "bounds": (bx0, by0, bx1, by1),
        "provider": provider, "provenance": {"layers": prov, "filled_pct": filled_pct,
                                             "match_layers": match_layers if match_coeffs else None},
        "out_path": None,
    }

    if out_path:
        import rasterio
        names = list(output_order) + (["source_rank"] if add_source_band else [])
        profile = {"driver": "GTiff", "height": ny, "width": nx, "count": len(names),
                   "dtype": "float32", "crs": _TARGET_CRS, "transform": geobox.transform,
                   "nodata": np.nan, "compress": "deflate", "predictor": 2, "tiled": True,
                   "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, b in enumerate(output_order, 1):
                dst.write(bands_out[b].astype("float32"), i)
                dst.set_band_description(i, b)
            if add_source_band:
                dst.write(source.astype("float32"), len(names))
                dst.set_band_description(len(names), "source_rank")
            dst.update_tags(dates=",".join(dates), provider=provider)
        result["out_path"] = out_path
        _log.info("wrote %s (%d bands, %dx%d, dates=%s)", out_path, len(names), nx, ny, dates)

    return result


def _process_static_tile(tile, dates, cloud_lt, out_dir, providers, run_tag, fetch_bands,
                         index_bands, output_order, scl_drop, dtype, index_scale,
                         add_source_band, match_coeffs=None):
    """Build + write one static tile (single multi-band COG + manifest). Resume-safe."""
    import numpy as np
    import rasterio

    tid = tile["tile_id"]
    res_m = int(round(abs(tile["geobox"].transform[0]) / _DEG_PER_M))
    out_tif = _os.path.join(out_dir, f"static_{res_m}m_tile_{tid:04d}.tif")
    out_man = out_tif.replace(".tif", ".manifest.json")
    if _os.path.exists(out_tif) and _os.path.exists(out_man):
        return {"tile_id": tid, "path": out_tif, "status": "skipped_exists"}

    catalog, p, provider = _pick_provider(providers)
    if catalog is None:
        return {"tile_id": tid, "status": "failed_no_provider"}

    gb = tile["geobox"]
    t0 = _time.time()
    bands_out, source, prov = _static_layers(catalog, p, provider, gb, tile["bbox"], dates,
                                             fetch_bands, index_bands, output_order,
                                             cloud_lt, scl_drop, match_coeffs=match_coeffs)
    ny, nx = gb.shape
    names = list(output_order) + (["source_rank"] if add_source_band else [])
    # index_scale=None -> scale only when the output is integer: uint16 cannot carry
    # a [-1, 1] index, float32 can and should keep the true values.
    scale = (10000 if dtype == "uint16" else 1) if index_scale is None else index_scale
    planes = []
    for b in output_order:
        arr = bands_out[b]
        if b in INDEX_BANDS and scale != 1:
            arr = arr * scale           # GEE parity: indices carried as scaled ints
        planes.append(arr)
    if add_source_band:
        planes.append(source.astype("float32"))

    if dtype == "uint16":
        # nodata/negatives -> 0, matching GEE's .max(0).uint16(). Note this floors
        # negative index values (water, bare soil) at 0; use dtype="float32" to keep them.
        out = np.stack([np.clip(np.nan_to_num(a, nan=0.0), 0, 65535) for a in planes]).astype("uint16")
        nodata = 0
    else:
        out = np.stack(planes).astype("float32")
        nodata = np.nan

    bx0, by0, bx1, by1 = tile["bbox"]
    profile = {"driver": "GTiff", "height": ny, "width": nx, "count": out.shape[0],
               "dtype": out.dtype.name, "crs": gb.crs, "transform": gb.transform,
               "nodata": nodata, "compress": "deflate", "predictor": 2, "tiled": True,
               "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
    tmp = out_tif + ".tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        for i in range(out.shape[0]):
            dst.write(out[i], i + 1)
            dst.set_band_description(i + 1, names[i])
        dst.update_tags(dates=",".join(dates))
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
    _os.replace(tmp, out_tif)

    manifest = {
        "tile_id": tid, "run_tag": run_tag, "provider": provider, "crs": str(gb.crs),
        "bbox": [bx0, by0, bx1, by1], "shape": [ny, nx], "resolution_deg": (bx1 - bx0) / nx,
        "bands": names, "band_order": output_order, "fetch_bands": fetch_bands,
        "index_bands": index_bands, "index_scale": scale if index_bands else None,
        "dtype": out.dtype.name, "dates": dates, "cloud_lt": cloud_lt,
        "scl_drop": list(scl_drop), "clouds_kept": sorted(set(_SCL_DROP_STATIC) - set(scl_drop)),
        "layers": prov,
        "filled_pct": round(100.0 * float(np.count_nonzero(source)) / (ny * nx), 2),
        "seconds": round(_time.time() - t0, 1),
    }
    with open(out_man, "w") as f:
        _json.dump(manifest, f, indent=2)
    return {"tile_id": tid, "path": out_tif, "status": "written",
            "seconds": manifest["seconds"], "provider": provider,
            "filled_pct": manifest["filled_pct"]}


def fetch_sentinel_static_imagery(aoi, start, end, bands=DEFAULT_BANDS, out_dir=None,
                                  res_m=10, tile_deg=0.1, cloud_lt=80, workers=8,
                                  providers=("mpc", "earthsearch"), run_tag=None,
                                  build_vrt_mosaic=True, clip_to_aoi=True,
                                  clip_all_touched=False, tile_retries=2, dates=None,
                                  n_dates=2, mask_clouds=False, scl_drop=_SCL_DROP_STATIC,
                                  dtype="uint16", index_scale=None, add_source_band=False,
                                  match_layers="median", **selection_kwargs):
    """Tiled, seamless static (single-acquisition) Sentinel-2 image over any AOI.

    The static counterpart of :func:`fetch_sentinel_imagery`, and the one to use for
    a district-sized AOI. Dates are chosen ONCE for the whole AOI and then applied
    identically to every tile, so the mosaic is temporally consistent as well as
    geometrically seamless -- the tiles cannot disagree about which acquisition
    they show. Threaded and resume-safe: re-running with the same `out_dir` skips
    finished tiles.

    Parameters
    ----------
    aoi : str | shapely geometry | GeoJSON dict | (minx, miny, maxx, maxy)
    start, end : str
        "YYYY-MM-DD", `end` exclusive.
    bands : sequence[str], default ("blue", "green", "red", "nir")
        Spectral and/or index bands. The FAO static-classifier set, for example, is
        ``["blue", "green", "red", "rededge1", "nir", "ndvi"]`` (B2,B3,B4,B5,B8,NDVI).
    out_dir : str, optional
        Tiles / manifests / VRT destination. Defaults to a fresh temp dir -- pass a
        stable path to get resume behaviour.
    res_m, tile_deg, cloud_lt, workers, providers, run_tag
        As in :func:`fetch_sentinel_imagery`.
    build_vrt_mosaic : bool, default True
        Stitch the tiles into `static.vrt`.
    clip_to_aoi : bool, default True
        Also write `static_clipped.tif`, the mosaic cut to the AOI polygon.
    tile_retries : int, default 2
        Extra attempts per tile on transient network errors (3s, 6s, 12s ... backoff).
    dates : sequence[str], optional
        Manual override -- layer exactly these dates, first = top. Skips selection.
    n_dates : int, default 2
        Max dates to layer when selecting automatically.
    dtype : {"uint16", "float32"}, default "uint16"
        Output type. uint16 matches the GEE export convention (reflectance DN,
        indices scaled by `index_scale`, negatives and no-data floored to 0).
        float32 keeps true index values and NaN no-data.
    index_scale : int, optional
        Multiplier applied to index bands (ndvi etc.) before writing. Default None
        picks it from `dtype`: 10000 for uint16 (which cannot hold a [-1, 1] index),
        1 for float32 (which keeps the true values). Pass a number to force it.
    mask_clouds : bool, default False
        False (the default here) -- cloud, shadow, cirrus and snow are kept exactly
                 as observed, and a lower layer only fills what the layers above
                 never imaged (outside their granule footprint). This is the default
                 because the tiled builder feeds classifiers that carry a cloud/haze
                 class: substituting another date's ground under a cloud hands the
                 model pixels that no longer match their label, and mixes two
                 acquisitions inside one scene.
        True  -- those pixels become holes, filled from the next date. Gives the
                 cleanest picture of the ground; use it for base maps and anything
                 where cloud is a nuisance rather than a class.
        Either way the dates themselves are still chosen by cloud, so the anchor is
        the least-cloudy acquisition available.

        NOTE: :func:`fetch_sentinel_static_composite` still defaults to True -- the
        interactive helper favours a clean look, this one favours model input.
    add_source_band : bool, default False
        Append a "source_rank" band (1 = top date, 2 = second, 0 = no data).
    match_layers : {"median", "linear", None}, default "median"
        Remove the tonal seam between dates by putting the lower layers on the top
        layer's radiometry, measured where they overlap. Estimated once for the
        whole AOI, so every tile applies the same correction. "median" shifts
        brightness only; "linear" also matches spread but can compress contrast;
        None disables it (GEE parity).
    **selection_kwargs
        Passed to :func:`select_static_dates`.

    Returns
    -------
    dict: {"out_dir", "tiles", "bands", "dates", "selection", "index_csv", "vrt",
           "clipped", "elapsed_min", "results"}

    Example
    -------
    >>> s = fetch_sentinel_static_imagery(
    ...     "sheikhupura.shp", "2025-05-01", "2025-05-16",
    ...     bands=["blue", "green", "red", "rededge1", "nir", "ndvi"],
    ...     out_dir="/data/sheikhupura_static")
    >>> s["dates"], s["clipped"]
    """
    import tempfile

    if dtype not in ("uint16", "float32"):
        raise ValueError("dtype must be 'uint16' or 'float32'")

    _set_gdal_http_env()
    fetch_bands, index_bands, output_order = resolve_static_bands(bands)
    out_dir = out_dir or tempfile.mkdtemp(prefix="s2_static_")
    _os.makedirs(out_dir, exist_ok=True)
    run_tag = run_tag or "run"
    res_deg = res_m * _DEG_PER_M

    geom = _load_aoi_geometry(aoi)

    # One date selection for the whole AOI -> every tile shows the same acquisitions.
    layer_drop = tuple(scl_drop) if mask_clouds else _SCL_DROP_NODATA
    sel_kwargs = dict(selection_kwargs)
    sel_kwargs.setdefault("n_dates", n_dates)
    sel_kwargs.setdefault("cloud_lt", cloud_lt)
    sel_kwargs.setdefault("providers", providers)
    sel_kwargs.setdefault("scl_drop", scl_drop)
    sel_kwargs.setdefault("coverage_basis", "usable" if mask_clouds else "footprint")
    sel_kwargs.setdefault("workers", workers)
    dates, selection = _resolve_static_dates(geom, start, end, dates, sel_kwargs)

    # Radiometric matching is estimated once, AOI-wide, and handed to every tile --
    # per-tile estimates would disagree and produce tile seams.
    match_coeffs = {}
    if match_layers and len(dates) > 1:
        catalog, p, provider = _pick_provider(providers)
        if catalog is None:
            raise RuntimeError(f"Could not open any STAC provider from {providers}.")
        match_coeffs = _layer_match_coeffs(catalog, p, provider, geom, dates, fetch_bands,
                                           cloud_lt, scl_drop, method=match_layers,
                                           workers=workers)

    tiles = _build_tiles(geom, tile_deg, res_deg)
    _log.info("AOI -> %d tiles @ %dm (tile=%.2fdeg, bands=%s, dates=%s, %d workers)",
              len(tiles), res_m, tile_deg, output_order, dates, workers)

    results = []
    t0 = _time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_with_tile_retries, _process_static_tile, t, tile_retries,
                          dates=dates, cloud_lt=cloud_lt, out_dir=out_dir,
                          providers=providers, run_tag=run_tag, fetch_bands=fetch_bands,
                          index_bands=index_bands, output_order=output_order,
                          scl_drop=layer_drop, dtype=dtype, index_scale=index_scale,
                          add_source_band=add_source_band, match_coeffs=match_coeffs):
                t["tile_id"] for t in tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            tid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 -- one tile failing must not kill the run
                _log.error("tile %04d failed: %s", tid, e)
                r = {"tile_id": tid, "status": f"failed: {type(e).__name__}"}
            results.append(r)
            _log.info("[%d/%d] tile %04d: %s%s", i, len(tiles), r["tile_id"], r["status"],
                      f" ({r.get('seconds')}s, {r.get('filled_pct')}% filled)"
                      if r.get("seconds") else "")

    index_csv = _os.path.join(out_dir, "static_index.csv")
    with open(index_csv, "w") as f:
        f.write("tile_id,status,provider,seconds,filled_pct,path\n")
        for r in sorted(results, key=lambda x: x["tile_id"]):
            f.write(f"{r['tile_id']},{r['status']},{r.get('provider', '')},"
                    f"{r.get('seconds', '')},{r.get('filled_pct', '')},{r.get('path', '')}\n")
    with open(_os.path.join(out_dir, "static_selection.json"), "w") as f:
        _json.dump(selection, f, indent=2)

    vrt = (_build_vrt(out_dir, "static.vrt", "static_*m_tile_*.tif")
           if (build_vrt_mosaic or clip_to_aoi) else None)
    clipped = None
    if clip_to_aoi and vrt:
        import numpy as np
        clipped = _clip_mosaic_to_geom(vrt, geom, _os.path.join(out_dir, "static_clipped.tif"),
                                       all_touched=clip_all_touched,
                                       nodata=0 if dtype == "uint16" else np.nan)
    _log.info("DONE %d tiles in %.1f min. dates=%s out_dir=%s vrt=%s",
              len(tiles), (_time.time() - t0) / 60, dates, out_dir, vrt)

    return {"out_dir": out_dir, "tiles": len(tiles), "bands": output_order, "dates": dates,
            "selection": selection, "match_coeffs": match_coeffs,
            "index_csv": index_csv, "vrt": vrt, "clipped": clipped,
            "elapsed_min": round((_time.time() - t0) / 60, 1), "results": results}


if __name__ == "__main__":
    import argparse
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch open-source Sentinel-2 imagery (GEE-free).")
    ap.add_argument("--aoi", required=True, help="Vector file path or 'minx,miny,maxx,maxy' bbox.")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--bands", default="blue,green,red,nir",
                    help="Comma-separated band names (default: blue,green,red,nir).")
    ap.add_argument("--out", default=None, help="Output dir (tiled) or file .tif (composite).")
    ap.add_argument("--mode", choices=["composite", "tiled", "static", "static-tiled",
                                       "static-dates"], default="composite")
    ap.add_argument("--n-dates", type=int, default=2,
                    help="Max acquisition dates to layer (static modes).")
    ap.add_argument("--dates", default=None,
                    help="Comma-separated dates to layer, first=top (skips selection).")
    ap.add_argument("--selection-mode", choices=["greedy", "penalty"], default="greedy")
    ap.add_argument("--cloud-metric", choices=["aoi", "scene"], default="aoi")
    ap.add_argument("--max-day-gap", type=int, default=None)
    ap.add_argument("--match-layers", choices=["median", "linear", "none"], default="median",
                    help="Radiometrically match lower layers to the top one (static modes).")
    ap.add_argument("--keep-clouds", action="store_true",
                    help="Keep cloud/shadow pixels instead of filling them from another "
                         "date; lower layers then only fill unimaged ground (static modes).")
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--res-m", type=int, default=10)
    ap.add_argument("--tile-deg", type=float, default=0.1)
    ap.add_argument("--cloud-lt", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tile-retries", type=int, default=2,
                    help="Extra attempts per tile on transient network errors (tiled mode).")
    ap.add_argument("--no-clip", action="store_true",
                    help="Skip clipping the mosaic to the AOI polygon (tiled mode).")
    a = ap.parse_args()

    _aoi = a.aoi
    if "," in _aoi and not _os.path.exists(_aoi):
        _aoi = tuple(float(x) for x in _aoi.split(","))
    _bands = [b for b in a.bands.split(",") if b.strip()]

    _dates = [d for d in a.dates.split(",") if d.strip()] if a.dates else None
    _sel = dict(selection_mode=a.selection_mode, cloud_metric=a.cloud_metric,
                max_day_gap=a.max_day_gap)
    _match = None if a.match_layers == "none" else a.match_layers
    _mask_clouds = not a.keep_clouds

    if a.mode == "static-dates":
        _r = select_static_dates(_aoi, a.start, a.end, n_dates=a.n_dates,
                                 cloud_lt=a.cloud_lt, workers=a.workers, **_sel)
        print(f"dates={_r['dates']} anchor={_r['anchor']} coverage={_r['coverage_pct']}% "
              f"(from {_r['n_candidate_dates']} candidate dates, metric={_r['cloud_metric']})")
        for c in _r["selected"]:
            print(f"  rank {c['rank']}: {c['date']}  cloud={c.get('cloud_pct')}%  "
                  f"cover={c.get('coverage_pct')}%  adds={c.get('gain_pct')}%")
    elif a.mode == "static":
        _r = fetch_sentinel_static_composite(_aoi, a.start, a.end, bands=_bands, res_m=a.res_m,
                                             cloud_lt=a.cloud_lt, out_path=a.out,
                                             dates=_dates, n_dates=a.n_dates,
                                             match_layers=_match, mask_clouds=_mask_clouds,
                                             **_sel)
        print(f"static: dates={_r['dates']} provider={_r['provider']} shape={_r['shape']} "
              f"bands={_r['band_order']} filled%={_r['provenance']['filled_pct']} "
              f"out={_r['out_path']}")
    elif a.mode == "static-tiled":
        _r = fetch_sentinel_static_imagery(_aoi, a.start, a.end, bands=_bands, out_dir=a.out,
                                           res_m=a.res_m, tile_deg=a.tile_deg,
                                           cloud_lt=a.cloud_lt, workers=a.workers,
                                           tile_retries=a.tile_retries, dates=_dates,
                                           n_dates=a.n_dates, clip_to_aoi=not a.no_clip,
                                           match_layers=_match, mask_clouds=_mask_clouds,
                                           **_sel)
        print(f"static-tiled: dates={_r['dates']} {_r['tiles']} tiles, {_r['elapsed_min']} min, "
              f"bands={_r['bands']} out_dir={_r['out_dir']} clipped={_r['clipped']}")
    elif a.mode == "composite":
        _r = fetch_sentinel_composite(_aoi, a.start, a.end, bands=_bands, res_m=a.res_m,
                                      cloud_lt=a.cloud_lt, out_path=a.out)
        print(f"composite: provider={_r['provider']} shape={_r['shape']} "
              f"bands={_r['band_order']} valid%={_r['provenance'].get('valid_pct')} "
              f"out={_r['out_path']}")
    else:
        _r = fetch_sentinel_imagery(_aoi, a.start, a.end, bands=_bands, out_dir=a.out,
                                    step=a.step, res_m=a.res_m, tile_deg=a.tile_deg,
                                    cloud_lt=a.cloud_lt, workers=a.workers,
                                    tile_retries=a.tile_retries, clip_to_aoi=not a.no_clip)
        print(f"tiled: {_r['tiles']} tiles, {_r['elapsed_min']} min, bands={_r['bands']} "
              f"out_dir={_r['out_dir']} vrt={_r['vrt']} clipped={_r['clipped']}")