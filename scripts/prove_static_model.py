"""Run the static model over Al-Moiz Unit 1 on several dates and compare.

The acceptance test for the rebuild. The deployed model keeps 85.0% of the RF cane
mask on 10 August and 61.8% on 30 August, a 23-point swing across twenty days of a
standing crop. A model that has seen the season should not care nearly as much.

    python3 prove_static_model.py                       # deployed model
    python3 prove_static_model.py --model ../model_files/fao_cane_xgb_model_v4.json
    python3 prove_static_model.py --compare             # both, side by side

Recall here means the share of RF-cane pixels the static model keeps, and false
positives the share of RF-non-cane pixels it claims. Neither is ground truth; they
measure agreement with the time-series model, which is the thing the static stage
is meant to refine rather than contradict.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from static_training import domain_check, inference  # noqa: E402

log = logging.getLogger("prove_static_model")

CROPSCAN = SCRIPTS_DIR.parent
TEST_DIR = CROPSCAN / "data" / "test_data" / "almoiz_unit_1_test_feature_1" / "cane_2026"
RF_MAP = TEST_DIR / "almoiz_unit_1_test_feature_1_rf_classification_map.tif"
IMAGE_DIR = CROPSCAN / "data" / "images"
IMAGE_STEM = "cropscan_cane_Al-Moiz-Unit-1-SM-AOI-2025_2026-"
DEFAULT_MODEL = CROPSCAN / "model_files" / "fao_cane_xgb_model.json"
V4_MODEL = CROPSCAN / "model_files" / "fao_cane_xgb_model_v4.json"

#: The 2026 acquisitions already downloaded and visually checked for cloud.
DATES = ["May-07", "Jun-06", "Jul-06", "Aug-10", "Aug-30", "Sep-04"]

BAND_ORDER = ("B2", "B3", "B4", "B5", "B8", "NDVI")


def _aligned_image(tag: str, work_dir: Path) -> Path:
    """Reproject one dated image onto the RF map's grid so masks line up."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    out = work_dir / f"almoiz_{tag}.tif"
    if out.exists():
        return out
    with rasterio.open(RF_MAP) as ref:
        profile, transform, shape, crs = ref.profile, ref.transform, ref.shape, ref.crs
    with rasterio.open(IMAGE_DIR / f"{IMAGE_STEM}{tag}.tif") as src:
        with WarpedVRT(src, crs=crs, transform=transform,
                       width=shape[1], height=shape[0], resampling=Resampling.nearest) as vrt:
            data = vrt.read()
    profile.update(count=len(BAND_ORDER), dtype="uint16", nodata=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)
    return out


def evaluate(model_path: Path, dates: Sequence[str], work_dir: Path,
             thresholds: Optional[Sequence[float]] = None) -> pd.DataFrame:
    import rasterio

    with rasterio.open(RF_MAP) as src:
        rf = src.read(1)
    cane_mask, other_mask = rf == 1, rf == 4

    sidecar_path = inference.sidecar_for(model_path)
    sidecar = domain_check.load_sidecar(sidecar_path) if sidecar_path.exists() else None

    rows: List[Dict] = []
    for tag in dates:
        image = _aligned_image(tag, work_dir)
        with rasterio.open(image) as src:
            array = src.read()

        verdict = None
        if sidecar is not None:
            verdict = domain_check.check_raster(array, list(BAND_ORDER), sidecar, mask=cane_mask)

        cane = inference.classify_raster(
            image, model_path, work_dir / f"cls_{tag}.tif", work_dir / f"prob_{tag}.tif",
            mask_array=cane_mask, enforce_domain=False,
        )
        other = inference.classify_raster(
            image, model_path, work_dir / f"clsN_{tag}.tif", None,
            mask_array=other_mask, enforce_domain=False,
        )

        row = {
            "date": tag,
            "shift_iqr": None if verdict is None else round(verdict.score, 2),
            "verdict": None if verdict is None else verdict.level,
            "threshold": cane.threshold,
            "recall": round(100 * cane.positive_fraction, 1),
            "false_pos": round(100 * other.positive_fraction, 1),
        }
        row["separation"] = round(row["recall"] - row["false_pos"], 1)
        rows.append(row)
        log.info("%s: shift=%s recall=%.1f%% fp=%.1f%%",
                 tag, row["shift_iqr"], row["recall"], row["false_pos"])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--compare", action="store_true",
                        help="score the deployed model and the v4 model side by side")
    parser.add_argument("--dates", nargs="*", default=DATES)
    parser.add_argument("--work-dir", type=Path,
                        default=Path("/home/da638081/.claude/jobs/2e46c5cc/tmp/prove"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("static_training.inference").setLevel(logging.WARNING)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    models = [("deployed", DEFAULT_MODEL)]
    if args.compare:
        if not V4_MODEL.exists():
            raise SystemExit(f"{V4_MODEL} not found; train it first")
        models.append(("v4", V4_MODEL))
    elif args.model != DEFAULT_MODEL:
        models = [(args.model.stem, args.model)]

    tables = []
    for name, path in models:
        log.info("\n=== %s (%s)", name, path.name)
        table = evaluate(path, args.dates, args.work_dir / name)
        table.insert(0, "model", name)
        tables.append(table)
        log.info("\n%s", table.to_string(index=False))

    combined = pd.concat(tables, ignore_index=True)
    out = args.work_dir / "comparison.csv"
    combined.to_csv(out, index=False)
    log.info("\nwritten to %s", out)

    if len(tables) > 1:
        log.info("\nrecall spread across dates (lower is better):")
        for name, table in zip([m[0] for m in models], tables):
            spread = table.recall.max() - table.recall.min()
            log.info("  %-10s %.1f points  (min %.1f%%, max %.1f%%)",
                     name, spread, table.recall.min(), table.recall.max())


if __name__ == "__main__":
    main()
