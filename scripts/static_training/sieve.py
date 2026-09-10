"""The sieve, as one definition instead of two.

Lifted verbatim out of `Model_Execution_Pipeline_v3.0.ipynb` so scripts and the
notebook cannot drift apart. It removes blobs below a pixel count, and then undoes any
merge where a non-target clump was absorbed into the target class without being
completely surrounded by it, which is what keeps a speck of background from being
swallowed by the field it happens to sit beside.

Sizes are in pixels, and at 10 m a pixel is 0.0247 acres:

    20 px = 0.49 acres      6 px = 0.15 acres
"""

import numpy as np
import rasterio
from rasterio.features import sieve
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from pathlib import Path
from typing import List

def apply_strict_directional_sieve(
    input_raster_path: str,
    target_classes: List[int],
    min_pixel_size: int = 15,
    connectivity: int = 4,
    nodata_val: int = 255
) -> str:
    """
    Applies a strict asymmetric Sieve filter using morphological connected components.
    A Non-Target clump is ONLY allowed to merge into the Target class group if it is 
    completely surrounded by Target pixels. Touching NoData or any other class aborts the merge.
    """
    in_path = Path(input_raster_path)
    out_name = f"{in_path.stem}_strict_sieve_multiclass_p{min_pixel_size}{in_path.suffix}"
    out_path = in_path.parent / out_name

    # --- NEW: Skip processing if output already exists ---
    if out_path.exists():
        print(f"[Skipped] Strict Sieved raster already exists at:\n{out_path}")
        return str(out_path)

    print(f"Loading categorical map for Strict Sieving: {in_path.name}")
    with rasterio.open(in_path) as src:
        meta = src.profile
        data = src.read(1)
        
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)
            meta.update(dtype=rasterio.uint8)

    valid_mask = (data != nodata_val).astype(np.uint8)

    print(f"Phase 1: Base Sieve Filter (Removing blobs < {min_pixel_size} pixels)...")
    sieved_data = sieve(
        data, 
        size=min_pixel_size, 
        connectivity=connectivity, 
        mask=valid_mask
    )

    print("Phase 2: Enforcing Strict Topological Encapsulation...")
    
    # Define topological boundaries using the entire group of target classes
    is_target_orig = np.isin(data, target_classes)
    is_target_sieved = np.isin(sieved_data, target_classes)
    
    # 1. Identify all pixels that changed from Non-Target -> ANY Target Class
    changed_mask = (~is_target_orig) & is_target_sieved
    
    # 2. Define "Bad Neighbors": Any pixel in original data that is NOT in the target group.
    # We exclude the `changed_mask` pixels themselves so clumps don't flag themselves.
    bad_neighbors = (~is_target_orig) & ~changed_mask
    
    # 3. Create a structuring element that matches your sieve connectivity
    struct = generate_binary_structure(2, 1) if connectivity == 4 else generate_binary_structure(2, 2)
    
    # 4. Dilate the bad neighbors by 1 pixel to create a "collision zone"
    bad_borders = binary_dilation(bad_neighbors, structure=struct)
    
    # 5. Find pixels inside our changed clumps that intersect the collision zone
    touched_by_bad = changed_mask & bad_borders
    
    # 6. Assign a unique ID to every distinct clump in the changed_mask
    labeled_clumps, num_features = label(changed_mask, structure=struct)
    
    if num_features > 0:
        bad_labels = np.unique(labeled_clumps[touched_by_bad])
        bad_labels = bad_labels[bad_labels != 0] 
        
        if len(bad_labels) > 0:
            print(f"  -> Reverting {len(bad_labels)} illegal edge-merges...")
            revert_mask = np.isin(labeled_clumps, bad_labels)
            sieved_data[revert_mask] = data[revert_mask]

    # Strictly enforce NoData limits
    sieved_data[valid_mask == 0] = nodata_val

    print("Writing topologically-enforced output to disk...")
    meta.update(compress='lzw')
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(sieved_data, 1)

    print(f"SUCCESS: Strict Sieved raster saved to:\n{out_path}")
    return str(out_path)



