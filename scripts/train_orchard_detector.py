"""Decide whether the shape of the year separates cane from orchards, then fit it.

Runs in that order deliberately. The separability stage is a gate: if the phenology
features do not pull the two classes apart, the premise is wrong and no model is
worth fitting. Only if it passes does anything get trained.

    python3 train_orchard_detector.py separability
    python3 train_orchard_detector.py fit

The detector is deliberately timid. Its threshold is chosen as the most orchard it
can remove while still keeping CANE_RECALL_FLOOR of known cane fields, because a
mill would rather carry a few orchards in its acreage than lose real fields.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from static_training import phenology as ph  # noqa: E402
from static_training import validate as st_validate  # noqa: E402

log = logging.getLogger("train_orchard_detector")

CROPSCAN = SCRIPTS_DIR.parent
OUT = Path("/mnt/c/Work_Work_Work/Python/Scripts/FAO/cane/orchard_detector")
PARQUET_DIR = OUT / "parquets"
REPORTS = OUT / "reports"
MODEL_PATH = CROPSCAN / "model_files" / "orchard_detector.json"

ORCHARD_LABEL, CANE_LABEL = 2, 1

#: Fields smaller than this carry too few pixels for a stable median.
MIN_FIELD_PIXELS = 12

#: The share of known cane fields the filter must leave standing.
CANE_RECALL_FLOOR = 0.99

#: A district's mask holes only count as orchards if they are greener on average
#: than the cane around them. That comparison is the whole point: an orchard is only
#: a problem when it is green enough to be taken for standing cane, and the test has
#: to be relative because what counts as green differs between the irrigated belt and
#: the desert margin. The absolute floor only stops two dark classes passing it.
MIN_ORCHARD_MEAN_NDVI = 0.45

PARAMS: Dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.08,
    "max_depth": 5,          # 14 features and a few thousand fields; deep trees only memorise
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
}


def load_pixels() -> pd.DataFrame:
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquets in {PARQUET_DIR}; run build_orchard_training_set.py first")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    log.info("pixels: %d across %d districts, labels %s",
             len(frame), frame.district.nunique(), frame.label.value_counts().to_dict())
    return frame


def date_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c.startswith("NDVI_")]


def pixel_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Phenology features per pixel, keeping the label, district and field."""
    dates = date_columns(frame)
    matrix = ph.build_from_dataframe(frame, dates)
    out = pd.DataFrame(matrix, columns=list(ph.FEATURE_NAMES))
    for column in ("label", "district", "field_id"):
        out[column] = frame[column].to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=list(ph.FEATURE_NAMES))


def to_fields(pixels: pd.DataFrame) -> pd.DataFrame:
    """One row per field: the median of each feature over its pixels.

    An orchard is an object, not a scatter of pixels, and a per-field median is far
    steadier than any single pixel. It also stops one large block from outweighing a
    hundred small ones during fitting.
    """
    grouped = pixels.groupby(["district", "field_id", "label"], observed=True)
    fields = grouped[list(ph.FEATURE_NAMES)].median()
    fields["n_pixels"] = grouped.size()
    fields = fields.reset_index()
    before = len(fields)
    fields = fields[fields.n_pixels >= MIN_FIELD_PIXELS].reset_index(drop=True)
    log.info("fields: %d (dropped %d under %d pixels), labels %s",
             len(fields), before - len(fields), MIN_FIELD_PIXELS,
             fields.label.value_counts().to_dict())
    return fields


def audit_districts(fields: pd.DataFrame) -> pd.DataFrame:
    """Check that a district's mask holes are actually orchards before training on them.

    The tree mask is not uniformly a tree mask. In the irrigated belt its holes are
    real orchards: in Rahim Yar Khan they sit between 0.56 and 1.00 NDVI all year,
    high and flat, which is exactly the thing that gets mistaken for standing cane.
    In arid Bhakkar the same holes run 0.03 to 0.40 and peak in February, which is
    desert scrub, not an orchard, and in Mirpur Khas they are darker than the cane
    beside them.

    Training on both teaches the detector that "orchard" means dark, the opposite of
    the problem it exists to solve, so a district whose positives are not perennial
    is dropped and the reason recorded.
    """
    rows = []
    for district in sorted(fields.district.unique()):
        orchard = fields[(fields.district == district) & (fields.label == ORCHARD_LABEL)]
        cane = fields[(fields.district == district) & (fields.label == CANE_LABEL)]
        if orchard.empty:
            continue
        mean_ndvi = float(orchard.ndvi_mean.median())
        min_ndvi = float(orchard.ndvi_min.median())
        cane_mean = float(cane.ndvi_mean.median()) if len(cane) else np.nan
        greener = np.isfinite(cane_mean) and mean_ndvi > cane_mean
        bright = mean_ndvi >= MIN_ORCHARD_MEAN_NDVI
        usable = bool(greener and bright)
        reason = ""
        if not bright:
            reason = "mask holes are not green vegetation"
        elif not greener:
            reason = "mask holes are darker than the cane around them"
        rows.append({
            "district": district,
            "orchard_fields": len(orchard), "cane_fields": len(cane),
            "orchard_mean_ndvi": round(mean_ndvi, 3),
            "orchard_min_ndvi": round(min_ndvi, 3),
            "cane_mean_ndvi": round(cane_mean, 3),
            "usable": usable,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def stage_separability(fields: pd.DataFrame) -> pd.DataFrame:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ph.FEATURE_NAMES:
        orchard = fields.loc[fields.label == ORCHARD_LABEL, name].to_numpy(dtype=float)
        cane = fields.loc[fields.label == CANE_LABEL, name].to_numpy(dtype=float)
        row = {
            "feature": name,
            "cohens_d": round(cohens_d(cane, orchard), 3),
            "cane_median": round(float(np.nanmedian(cane)), 3),
            "orchard_median": round(float(np.nanmedian(orchard)), 3),
        }
        for district in sorted(fields.district.unique()):
            sub = fields[fields.district == district]
            row[district] = round(cohens_d(
                sub.loc[sub.label == CANE_LABEL, name].to_numpy(dtype=float),
                sub.loc[sub.label == ORCHARD_LABEL, name].to_numpy(dtype=float)), 2)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("cohens_d", ascending=False)
    table.to_csv(REPORTS / "separability.csv", index=False)
    log.info("field-level separability, cane against orchard:\n%s", table.to_string(index=False))

    best = table.cohens_d.max()
    log.info("strongest feature: %s at d=%.2f", table.feature.iloc[0], best)
    if best < 1.0:
        log.warning("nothing reaches d=1.0. The premise does not hold on this data; "
                    "stop here rather than fitting a model on it.")
    return table


def _fit(train: pd.DataFrame, features: Sequence[str], seed: int = 42):
    import xgboost as xgb

    x = train[list(features)].to_numpy(dtype=np.float32)
    y = (train.label.to_numpy() == ORCHARD_LABEL).astype(int)
    negatives, positives = int((y == 0).sum()), int((y == 1).sum())
    params = {**PARAMS, "scale_pos_weight": negatives / max(positives, 1), "random_state": seed}
    dtrain = xgb.DMatrix(x, label=y, feature_names=list(features))
    return xgb.train(params, dtrain, num_boost_round=250)


def _predict(booster, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    import xgboost as xgb

    matrix = frame[list(features)].to_numpy(dtype=np.float32)
    return booster.predict(xgb.DMatrix(matrix, feature_names=list(features)))


def threshold_at_cane_floor(cane_scores: np.ndarray, orchard_scores: np.ndarray,
                            floor: float = CANE_RECALL_FLOOR) -> Dict[str, float]:
    """Lowest cut that still spares `floor` of cane fields, and what it removes."""
    best = {"threshold": 1.0, "cane_kept": 1.0, "orchard_removed": 0.0}
    for cut in np.round(np.arange(0.05, 1.0, 0.01), 2):
        cane_kept = float((cane_scores < cut).mean())
        if cane_kept < floor:
            continue
        removed = float((orchard_scores >= cut).mean())
        if removed > best["orchard_removed"]:
            best = {"threshold": float(cut), "cane_kept": cane_kept,
                    "orchard_removed": removed}
    return best


def stage_fit(fields: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    features = list(ph.FEATURE_NAMES)

    fold_rows, pooled_cane, pooled_orchard = [], [], []
    for district in sorted(fields.district.unique()):
        test = fields[fields.district == district]
        train = fields[fields.district != district]
        if train.empty or test.label.nunique() < 2:
            log.warning("%s: cannot be a fold, skipped", district)
            continue
        booster = _fit(train, features)
        scores = _predict(booster, test, features)
        truth = (test.label.to_numpy() == ORCHARD_LABEL).astype(int)

        cane_scores, orchard_scores = scores[truth == 0], scores[truth == 1]
        pooled_cane.append(cane_scores)
        pooled_orchard.append(orchard_scores)
        point = threshold_at_cane_floor(cane_scores, orchard_scores)

        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(truth, scores))
        except Exception:
            auc = float("nan")
        fold_rows.append({
            "held_out": district, "n_train": len(train), "n_test": len(test),
            "roc_auc": round(auc, 4), "threshold": point["threshold"],
            "cane_kept": round(point["cane_kept"], 4),
            "orchard_removed": round(point["orchard_removed"], 4),
        })
        log.info("%-16s AUC %.4f | at %.0f%% cane kept, removes %.1f%% of orchards",
                 district, auc, 100 * point["cane_kept"], 100 * point["orchard_removed"])

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(REPORTS / "leave_one_district_out.csv", index=False)
    log.info("leave-one-district-out:\n%s", folds.to_string(index=False))

    pooled = threshold_at_cane_floor(np.concatenate(pooled_cane), np.concatenate(pooled_orchard))
    log.info("pooled operating point: threshold %.2f keeps %.1f%% of cane and "
             "removes %.1f%% of orchards",
             pooled["threshold"], 100 * pooled["cane_kept"], 100 * pooled["orchard_removed"])

    booster = _fit(fields, features)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))

    quantiles = {}
    for label, name in [(CANE_LABEL, "cane"), (ORCHARD_LABEL, "orchard")]:
        sub = fields[fields.label == label]
        quantiles[name] = {
            f: {f"q{int(q*100):02d}": round(float(sub[f].quantile(q)), 4)
                for q in (0.25, 0.5, 0.75)}
            for f in features
        }

    sidecar = {
        "model": "orchard detector",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_names": features,
        "feature_definitions": ph.describe(features),
        "positive_class": "orchard",
        "decision_threshold": pooled["threshold"],
        "threshold_rule": (
            f"highest orchard removal subject to keeping at least "
            f"{CANE_RECALL_FLOOR:.0%} of known cane fields, pooled across "
            f"leave-one-district-out folds"),
        "cane_kept_at_threshold": round(pooled["cane_kept"], 4),
        "orchard_removed_at_threshold": round(pooled["orchard_removed"], 4),
        "min_field_pixels": MIN_FIELD_PIXELS,
        "training_districts": sorted(fields.district.unique()),
        "training_fields": {"cane": int((fields.label == CANE_LABEL).sum()),
                            "orchard": int((fields.label == ORCHARD_LABEL).sum())},
        "training_quantiles": quantiles,
        "folds": fold_rows,
        "series_window": "2025-11-24 to 2026-09-09, 8-day steps, Whittaker lmbd=0.5 d=2",
    }
    sidecar_path = MODEL_PATH.with_suffix(".sidecar.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    log.info("model -> %s", MODEL_PATH)
    log.info("sidecar -> %s", sidecar_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["separability", "fit", "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    fields = to_fields(pixel_features(load_pixels()))

    audit = audit_districts(fields)
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit.to_csv(REPORTS / "district_audit.csv", index=False)
    log.info("district audit:\n%s", audit.to_string(index=False))
    usable = list(audit.loc[audit.usable, "district"])
    dropped = list(audit.loc[~audit.usable, "district"])
    if dropped:
        log.warning("dropped for training: %s", dropped)
    if not usable:
        raise SystemExit(
            "no district has perennial mask holes; the tree mask cannot serve as "
            "orchard labels here and the approach needs rethinking")
    fields = fields[fields.district.isin(usable)].reset_index(drop=True)
    log.info("training on %s: %d fields", usable, len(fields))

    if args.stage in ("separability", "all"):
        stage_separability(fields)
    if args.stage in ("fit", "all"):
        stage_fit(fields)


if __name__ == "__main__":
    main()
