"""ANSWERED: SWIR does not help this classifier. Kept as the record of that test.

The question was whether B11 and B12 could close the August gap. Leave-one-month-out
had shown August to be the hardest month, ROC-AUC 0.847 against 0.899 for November,
and the reason looked like a missing measurement rather than domain shift: in August
cotton, rice and maize are all green too, and SWIR is where canopy water separates
them while everything looks alike in the visible and NIR.

Per-feature separability supports that reasoning. On August imagery NDMI reaches a
Cohen's d of 1.785 against NDVI's 1.570, so the SWIR indices really do carry signal.

The model still does not want them. Leave-one-AOI-out over six mills, training on
November and testing on the held-out mill's August:

    feature set              mean August AUC   change   folds won
    deployed six bands                0.8807        -           -
    plus NDRE and GNDVI               0.8858   +0.0051         4/6
    plus B11 and B12                  0.8673   -0.0134         3/6
    plus SWIR indices too             0.8549   -0.0258         2/6

Adding features makes it monotonically worse. With one acquisition window in
training, extra inputs mostly buy new ways to fit November detail that does not
survive to August. A single-fold version of this test on three mills reported
+0.027 for SWIR; the fold-to-fold standard deviation is 0.03 to 0.05, so that was
noise. Cross-validate before believing a feature-set result on this data.

    python3 swir_pilot.py export
    python3 swir_pilot.py evaluate
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import build_static_training_set as B  # noqa: E402
import sentinel  # noqa: E402
from static_training import config as st_config  # noqa: E402
from static_training import extract as st_extract  # noqa: E402
from static_training import train as st_train  # noqa: E402
from static_training import validate as st_validate  # noqa: E402

log = logging.getLogger("swir_pilot")

OUT = st_config.FAO_ROOT / "cane" / "static_model" / "swir_pilot"
IMAGES = OUT / "images"
PARQUETS = OUT / "parquets"

#: A spread of mills rather than neighbours, so the answer is not one district's
#: quirk. Each must have both an August and a November acquisition already scouted.
PILOT_AOIS = ["ALMoiz1", "jdw1", "sheikhoo", "Khanewal_dist", "hunza2", "f7"]

FETCH_BANDS = ["blue", "green", "red", "rededge1", "nir", "swir16", "swir22", "ndvi"]
BAND_ORDER = ["B2", "B3", "B4", "B5", "B8", "B11", "B12", "NDVI"]

BASE_FEATURES = ["B2", "B3", "B4", "B5", "B8", "NDVI"]
SWIR_FEATURES = BASE_FEATURES + ["B11", "B12", "NDMI", "R1112"]


def _pairs() -> List[tuple[str, str]]:
    """One August and one November date per pilot AOI."""
    chosen: List[tuple[str, str]] = []
    for aoi, date in B.selected_pairs():
        if aoi in PILOT_AOIS and date.startswith("2025-08"):
            chosen.append((aoi, date))
    for aoi, date, _ in B._existing_images():
        if aoi in PILOT_AOIS and date.startswith("2025-11"):
            chosen.append((aoi, date))
    return sorted(set(chosen))


def stage_export(workers: int) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    aois = {a["aoi"]: a for a in B.load_aois()}
    todo = [(a, d) for a, d in _pairs() if not (IMAGES / f"{a}_{d}.tif").exists()]
    log.info("exporting %d eight-band images", len(todo))

    for i, (aoi, date) in enumerate(todo, 1):
        staging = IMAGES / f".staging_{aoi}_{date}"
        try:
            result = sentinel.fetch_sentinel_static_imagery(
                aoi=aois[aoi]["geometry"], start=date, end=date,
                bands=FETCH_BANDS, out_dir=str(staging), res_m=10, tile_deg=0.2,
                dates=[date], n_dates=1, mask_clouds=False, workers=workers,
                build_vrt_mosaic=True, clip_to_aoi=True, dtype="uint16",
            )
            clipped = result.get("clipped")
            if clipped and Path(clipped).exists():
                Path(clipped).replace(IMAGES / f"{aoi}_{date}.tif")
                log.info("[%d/%d] %s %s", i, len(todo), aoi, date)
        except Exception as exc:
            log.error("[%d/%d] %s %s failed: %s", i, len(todo), aoi, date, exc)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


def stage_extract() -> None:
    from shapely.ops import unary_union

    PARQUETS.mkdir(parents=True, exist_ok=True)
    cfg = st_config.get("cane")
    aois = {a["aoi"]: a for a in B.load_aois()}
    labels = B.load_label_polygons()
    index = labels.sindex

    for image in sorted(IMAGES.glob("*.tif")):
        aoi, date = image.stem.rsplit("_", 1)
        out = PARQUETS / f"{aoi}_{date}.parquet"
        if out.exists():
            continue
        geom = aois[aoi]["geometry"]
        hits = list(index.query(geom, predicate="intersects"))
        positive = unary_union(labels.geometry.iloc[hits].values).intersection(geom) if hits else None
        result = st_extract.extract_from_raster(
            image, positive, geom, aoi, date,
            crop_label=cfg.crop_label, background_label=cfg.background_label,
            band_order=BAND_ORDER, max_pixels_per_class=200_000,
        )
        if result.ok:
            result.frame.to_parquet(out, index=False)
            log.info("%s %s: %d rows", aoi, date, len(result.frame))


def stage_evaluate() -> None:
    files = sorted(PARQUETS.glob("*.parquet"))
    if not files:
        raise SystemExit("nothing extracted yet")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frame["month"] = frame.date.str.slice(0, 7)
    log.info("pilot rows: %d, AOIs: %d, months: %s",
             len(frame), frame.aoi.nunique(), sorted(frame.month.unique()))

    august = frame[frame.month.str.endswith("-08")]
    november = frame[~frame.month.str.endswith("-08")]
    if august.empty or november.empty:
        raise SystemExit("need both August and November rows")

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    rows = []
    for name, features in [("6 bands, no SWIR", BASE_FEATURES), ("with SWIR", SWIR_FEATURES)]:
        booster, report, _ = st_train.fit(
            november, features, num_boost_round=300, early_stopping_rounds=25
        )
        matrix = st_train.feature_matrix(august, features)
        dtest = xgb.DMatrix(matrix, feature_names=list(features))
        prob = booster.predict(dtest)
        truth = (august.label.to_numpy() == 1).astype(int)
        precision, recall, f1 = st_validate.binary_scores(truth, prob, 0.5)
        rows.append({
            "features": name,
            "n_features": len(features),
            "august_auc": round(float(roc_auc_score(truth, prob)), 4),
            "recall@0.5": round(recall, 4),
            "precision@0.5": round(precision, 4),
            "f1@0.5": round(f1, 4),
        })
        log.info("%s -> August AUC %.4f", name, rows[-1]["august_auc"])

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "swir_pilot.csv", index=False)
    log.info("trained on November, tested on August:\n%s", table.to_string(index=False))
    gain = table.august_auc.iloc[-1] - table.august_auc.iloc[0]
    log.info("SWIR changes August AUC by %+.4f", gain)
    log.info("worth re-exporting everything at eight bands: %s",
             "yes" if gain >= 0.02 else "not on this evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export", "extract", "evaluate", "all"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sentinel").setLevel(logging.WARNING)

    if args.stage in ("export", "all"):
        stage_export(args.workers)
    if args.stage in ("extract", "all"):
        stage_extract()
    if args.stage in ("evaluate", "all"):
        stage_evaluate()


if __name__ == "__main__":
    main()
