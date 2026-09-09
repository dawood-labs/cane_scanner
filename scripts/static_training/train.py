"""Fit a static crop classifier and ship it with everything inference needs.

The model file alone is not enough. `fao_cane_xgb_model.json` carries its feature
names but nothing about the distribution it was fitted on, no decision threshold,
and no record of which dates it saw. Training picked an optimal threshold by
maximising F1 and then threw it away, so inference falls back to 0.5.

Every model this module writes gets a sidecar JSON beside it holding the feature
list, per-class training quantiles, the selected threshold and the acquisition
dates in the training pool. `domain_check` reads it to refuse out-of-distribution
imagery, and the execution pipeline reads the threshold instead of assuming one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import features as feat
from . import validate as val

QUANTILES = (0.02, 0.25, 0.50, 0.75, 0.98)

DEFAULT_PARAMS: Dict[str, object] = {
    # Starting point taken from the recorded winning Optuna trial for the
    # deployed cane model, so a retrain is comparable to what it replaces.
    "learning_rate": 0.0614,
    "max_depth": 10,
    "min_child_weight": 4,
    "subsample": 0.919,
    "colsample_bytree": 0.981,
    "gamma": 1.641,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
    "random_state": 42,
}


@dataclass
class TrainingReport:
    n_rows: int
    n_features: int
    positive_rows: int
    negative_rows: int
    scale_pos_weight: float
    threshold: float
    best_iteration: Optional[int]
    date_holdout: Optional[Dict[str, object]] = None
    aoi_holdout: Optional[Dict[str, object]] = None


def feature_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """Build the model input matrix from a table of source bands."""
    source = feat.source_from_dataframe(frame)
    return feat.build_matrix(source, feature_names)


#: Kept so existing callers of the private name keep working.
_matrix = feature_matrix


def _quantiles(matrix: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(feature_names):
        column = matrix[:, index]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        values = np.quantile(column, QUANTILES)
        out[name] = {f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, values)}
    return out


def fit(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    positive_label: int = 1,
    label_column: str = "label",
    params: Optional[Dict[str, object]] = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 30,
    validation_date: Optional[str] = None,
    date_column: str = "date",
    aoi_column: str = "aoi",
    seed: int = 42,
):
    """Fit one model, stopping early on data the model has not effectively seen.

    Name a `validation_date` and that acquisition is held out whole, which is the
    right choice when the question is how the model behaves on a new image. With no
    date named, whole AOI-date groups are held out instead. Either way the point is
    the same: neighbouring pixels in one field are near-duplicates, so a random
    per-pixel split puts copies of the same thing on both sides and early stopping
    keeps boosting long after the model has stopped learning anything transferable.
    """
    import xgboost as xgb

    params = {**DEFAULT_PARAMS, **(params or {})}

    if validation_date is not None:
        holdout_mask = frame[date_column] == validation_date
        if not holdout_mask.any():
            raise ValueError(f"no rows on validation_date {validation_date!r}")
        train_frame, valid_frame = frame[~holdout_mask], frame[holdout_mask]
    else:
        # Hold out whole AOI-date groups rather than scattered pixels. Neighbouring
        # pixels in one field are near-duplicates, so a random split puts copies of
        # the same thing on both sides and early stopping keeps boosting long past
        # the point where the model has stopped learning anything transferable.
        rng = np.random.default_rng(seed)
        group_columns = [c for c in (aoi_column, date_column) if c in frame.columns]
        if group_columns:
            groups = frame[group_columns].drop_duplicates()
            n_hold = max(1, int(round(len(groups) * 0.05)))
            picked = groups.iloc[rng.choice(len(groups), size=n_hold, replace=False)]
            holdout_mask = frame.set_index(group_columns).index.isin(
                picked.set_index(group_columns).index
            )
        else:
            holdout_mask = rng.random(len(frame)) < 0.05
        train_frame, valid_frame = frame[~holdout_mask], frame[holdout_mask]

    x_train = _matrix(train_frame, feature_names)
    y_train = (train_frame[label_column].to_numpy() == positive_label).astype(int)
    x_valid = _matrix(valid_frame, feature_names)
    y_valid = (valid_frame[label_column].to_numpy() == positive_label).astype(int)

    keep_train = feat.finite_mask(x_train)
    keep_valid = feat.finite_mask(x_valid)
    x_train, y_train = x_train[keep_train], y_train[keep_train]
    x_valid, y_valid = x_valid[keep_valid], y_valid[keep_valid]

    negatives, positives = int((y_train == 0).sum()), int((y_train == 1).sum())
    if positives == 0:
        raise ValueError("training split has no positive rows")
    params["scale_pos_weight"] = negatives / positives

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(feature_names))
    dvalid = xgb.DMatrix(x_valid, label=y_valid, feature_names=list(feature_names))
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    valid_prob = booster.predict(dvalid)
    threshold, scores = val.select_threshold(y_valid, valid_prob)

    report = TrainingReport(
        n_rows=len(x_train),
        n_features=len(feature_names),
        positive_rows=positives,
        negative_rows=negatives,
        scale_pos_weight=float(params["scale_pos_weight"]),
        threshold=threshold,
        best_iteration=getattr(booster, "best_iteration", None),
    )
    return booster, report, scores


def write_sidecar(
    path: Path | str,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    report: TrainingReport,
    crop: str,
    static_window: Sequence[str],
    outlier_caps: Dict[str, int],
    positive_label: int = 1,
    label_column: str = "label",
    date_column: str = "date",
    aoi_column: str = "aoi",
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Write the JSON that travels with the model."""
    positive = frame[frame[label_column] == positive_label]
    negative = frame[frame[label_column] != positive_label]

    payload: Dict[str, object] = {
        "crop": crop,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_names": list(feature_names),
        "feature_definitions": feat.describe(feature_names),
        "index_scale": feat.INDEX_SCALE,
        "positive_label": positive_label,
        "decision_threshold": report.threshold,
        "static_window": list(static_window),
        "outlier_caps": outlier_caps,
        "training_rows": int(len(frame)),
        # "pooled" covers a whole scene, which is what the domain guard sees when
        # no upstream mask has narrowed the pixels to crop candidates.
        "training_quantiles": {
            "positive": _quantiles(feature_matrix(positive, feature_names), feature_names),
            "negative": _quantiles(feature_matrix(negative, feature_names), feature_names),
            "pooled": _quantiles(feature_matrix(frame, feature_names), feature_names),
        },
        "report": asdict(report),
    }

    # Flat medians, which is what domain_check reads on every inference run.
    payload["training_medians"] = {
        role: {name: stats["q50"] for name, stats in payload["training_quantiles"][role].items()}
        for role in ("positive", "negative", "pooled")
    }

    if date_column in frame.columns:
        payload["training_dates"] = sorted(str(d) for d in frame[date_column].unique())
    if aoi_column in frame.columns:
        payload["training_aois"] = sorted(str(a) for a in frame[aoi_column].unique())
    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def make_fit_predict(feature_names: Sequence[str], **fit_kwargs):
    """Adapter so `validate.leave_one_group_out` can drive this trainer."""

    def fit_predict(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        booster, _, _ = fit(train_frame, feature_names, **fit_kwargs)
        x_test = _matrix(test_frame, feature_names)
        dtest = xgb.DMatrix(x_test, feature_names=list(feature_names))
        return booster.predict(dtest)

    return fit_predict
