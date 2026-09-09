"""Train the date-robust static cane model and prove it beats the one it replaces.

Three stages, each answering a question the current model cannot:

    stability   which features hold still when the acquisition date changes
    ablation    does multi-date training actually fix August, and does the new
                feature set help beyond that
    final       fit the chosen model and write its sidecar

The ablation is the point. It trains an October-November model, exactly the shape of
the deployed one, and scores it on August held out entirely; then trains the same
model on a table that also contains September and scores it the same way. Any claim
that this work helped has to survive that comparison.

Read recall at matched precision, not at 0.5. Two models sitting at different points
on the same curve will show a large recall difference that vanishes once both are
asked to tolerate the same number of false positives. On this data that distinction
turned an apparent 13.8-point gain into 1.5 points.

The feature set is settled and is the one the deployed model already uses. The
stability stage is a diagnostic that explains why a single-date model breaks; its
suggestion was tested and lost. So were SWIR and the extra indices. Pass --features
to try something else, but cross-validate it: fold-to-fold AUC varies by 0.03 to
0.05 here, which is wider than any feature-set effect measured so far.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from static_training import config as st_config  # noqa: E402
from static_training import stability as st_stability  # noqa: E402
from static_training import train as st_train  # noqa: E402
from static_training import validate as st_validate  # noqa: E402

log = logging.getLogger("train_static_model")

V4 = st_config.FAO_ROOT / "cane" / "static_model" / "v4"
MERGED = V4 / "master_multidate.parquet"
MODEL_DIR = st_config.CROPSCAN_ROOT / "model_files"
REPORT_DIR = V4 / "reports"

#: What the deployed model uses. The baseline everything is measured against.
LEGACY_FEATURES = ["B2", "B3", "B4", "B5", "B8", "NDVI"]

#: Candidates for the stability study: the raw bands plus one representative of
#: each band pair. A normalised difference and the corresponding ratio carry the
#: same information to a tree model, which splits on order and is blind to any
#: monotone transform: NDVI and B8/B4 are the same feature, so are NDRE and B8/B5,
#: and NDWI is simply GNDVI negated. Only the better-scaled member of each pair is
#: kept, so the ranking compares distinct information rather than the same signal
#: written three ways.
CANDIDATE_FEATURES = [
    "B2", "B3", "B4", "B5", "B8",
    "NDVI",   # B8 vs B4
    "GNDVI",  # B8 vs B3
    "NDRE",   # B8 vs B5
    "R43",    # B4 vs B3, the only visible-only contrast
    "R28",    # B2 vs B8
]

#: Rows per AOI-date kept when fitting, to bound memory and stop one big district
#: from dominating. Sampling is stratified by class inside each group.
ROWS_PER_GROUP = 60_000


def load(sample_per_group: int = ROWS_PER_GROUP, seed: int = 42) -> pd.DataFrame:
    """Read a class-balanced sample of the training table, one row group at a time.

    The merged table holds about 60 million rows and roughly 3.3 GB in pandas.
    Reading it whole and then sampling needs that much plus a copy, which is more
    than this 7.4 GB machine has once the editor's own processes are counted, and
    the kernel's OOM killer resolves it by taking the session down.

    The merge stage writes one row group per AOI-date, so streaming row groups is
    the same thing as streaming AOI-dates: each is sampled and released before the
    next is read, and the peak is one row group rather than the whole table.
    """
    import pyarrow.parquet as pq

    if not MERGED.exists():
        raise SystemExit(f"{MERGED} not found; run build_static_training_set.py merge first")

    handle = pq.ParquetFile(MERGED)
    rng = np.random.default_rng(seed)
    per_class = sample_per_group // 2
    parts: List[pd.DataFrame] = []
    total = 0

    for index in range(handle.metadata.num_row_groups):
        group = handle.read_row_group(index).to_pandas()
        total += len(group)
        for _, chunk in group.groupby("label", sort=False):
            if len(chunk) > per_class:
                chunk = chunk.iloc[rng.choice(len(chunk), size=per_class, replace=False)]
            parts.append(chunk)
        del group

    out = pd.concat(parts, ignore_index=True)
    del parts
    out["month"] = out["date"].astype(str).str.slice(0, 7).astype("category")
    for column in ("aoi", "date"):
        if out[column].dtype.name != "category":
            out[column] = out[column].astype("category")
    log.info("streamed %d rows from %d row groups, sampled to %d",
             total, handle.metadata.num_row_groups, len(out))
    log.info("AOIs: %d, dates: %d, memory %.0f MB",
             out.aoi.nunique(), out.date.nunique(),
             out.memory_usage(deep=True).sum() / 1e6)
    return out


def stage_stability(frame: pd.DataFrame) -> List[str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = st_config.get("cane")
    # Compare against November, the window the deployed model was fitted in, so the
    # drift figures say how far each feature has moved from what it already knows.
    # The grouping key is the month, so the reference has to be one too.
    months = sorted(frame.month.unique())
    reference = next((m for m in months if m.endswith("-11")), months[-1] if months else None)

    result = st_stability.report(
        frame, CANDIDATE_FEATURES, label_value=cfg.crop_label,
        reference_date=reference, date_column="month",
    )
    for name, table in result.items():
        table.to_csv(REPORT_DIR / f"stability_{name}.csv")
    log.info("feature stability, steadiest first:\n%s", result["ranking"].to_string())

    chosen = st_stability.suggest_feature_set(result["ranking"], max_range_pct=15.0, minimum=5)
    log.info("suggested feature set: %s", chosen)
    (REPORT_DIR / "chosen_features.json").write_text(json.dumps(chosen, indent=2))
    return chosen


def _recall_at_precision(truth: np.ndarray, prob: np.ndarray, target_precision: float) -> float:
    """Highest recall reachable while holding precision at or above the target.

    Sweeping the threshold traces one curve per model; this reads each curve at the
    same precision, which is the only way to compare two models that sit at
    different operating points.
    """
    best = 0.0
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        precision, recall, _ = st_validate.binary_scores(truth, prob, float(threshold))
        if precision >= target_precision:
            best = max(best, recall)
    return best


def _fold_scores(frame: pd.DataFrame, features: Sequence[str], group_column: str) -> Dict:
    fit_predict = st_train.make_fit_predict(
        features, num_boost_round=250, early_stopping_rounds=20
    )
    report = st_validate.leave_one_group_out(
        frame, group_column, fit_predict, positive_label=st_config.CANE.crop_label
    )
    return {"table": report.table(), "summary": report.summary()}


def _stability_suggestion() -> List[str]:
    """Whatever the stability stage last suggested, for the ablation's fourth arm."""
    path = REPORT_DIR / "chosen_features.json"
    return json.loads(path.read_text()) if path.exists() else LEGACY_FEATURES


def stage_ablation(frame: pd.DataFrame, chosen: Sequence[str]) -> None:
    """Does multi-date training fix August, and do better features add anything?"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    august = sorted(m for m in frame.month.unique() if m.endswith("-08"))
    if not august:
        raise SystemExit("no August rows; nothing to prove")
    # August is the test, so no arm may train on it. The two training pools differ
    # only by September, which is what makes the first two rows a clean measurement
    # of what one extra month buys.
    test = frame[frame.month.isin(august)]
    novemberish = frame[frame.month.str.endswith(("-10", "-11"))]
    multidate = frame[~frame.month.isin(august)]
    log.info("test months %s (%d rows); training pools %s and %s",
             august, len(test), sorted(novemberish.month.unique()),
             sorted(multidate.month.unique()))

    # Four arms, so the two questions separate cleanly. Rows one and two differ only
    # in the dates seen, isolating what multi-date training is worth. Rows two to
    # four differ only in the feature set, isolating what the stability study is
    # worth. The widest set is included because dropping a discriminative but drifty
    # feature such as NIR may cost more accuracy than the drift does.
    rows = []
    baseline_precision = None
    for label, train_frame, features in [
        ("Oct/Nov only, 6 raw features (the deployed shape)", novemberish, LEGACY_FEATURES),
        ("plus Sep, 6 raw features", multidate, LEGACY_FEATURES),
        ("plus Sep, all candidates", multidate, CANDIDATE_FEATURES),
        ("plus Sep, stability-suggested subset", multidate, _stability_suggestion()),
    ]:
        if train_frame.empty:
            continue
        booster, report, _ = st_train.fit(
            train_frame, features, num_boost_round=300, early_stopping_rounds=25
        )
        import xgboost as xgb

        matrix = st_train.feature_matrix(test, features)
        dtest = xgb.DMatrix(matrix, feature_names=list(features))
        prob = booster.predict(dtest)
        truth = (test.label.to_numpy() == st_config.CANE.crop_label).astype(int)
        at_half = st_validate.binary_scores(truth, prob, 0.5)
        row = {
            "setup": label,
            "n_train": len(train_frame),
            "n_features": len(features),
            "recall@0.5": round(at_half[1], 4),
            "precision@0.5": round(at_half[0], 4),
            "f1@0.5": round(at_half[2], 4),
            "best_f1": round(st_validate.binary_scores(truth, prob, report.threshold)[2], 4),
        }
        # Comparing recall at a fixed 0.5 is unfair when the models sit at different
        # points on the same curve. Reporting recall at the first arm's precision
        # asks the only question that matters: for the same tolerance of false
        # positives, how much more cane does each model find.
        if baseline_precision is None:
            baseline_precision = at_half[0]
        row["recall@matched_precision"] = round(
            _recall_at_precision(truth, prob, baseline_precision), 4
        )
        rows.append(row)
        log.info("%s -> f1@0.5 %.4f, recall@0.5 %.4f, recall at %.3f precision %.4f",
                 label, at_half[2], at_half[1], baseline_precision,
                 row["recall@matched_precision"])

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "ablation_august_holdout.csv", index=False)
    log.info("August held out entirely:\n%s", table.to_string(index=False))


def stage_grouped(frame: pd.DataFrame, chosen: Sequence[str]) -> None:
    for group_column, name in [("month", "month"), ("aoi", "aoi")]:
        result = _fold_scores(frame, chosen, group_column)
        result["table"].to_csv(REPORT_DIR / f"leave_one_{name}_out.csv", index=False)
        log.info("leave-one-%s-out:\n%s", name, result["table"].to_string(index=False))
        log.info("summary: %s", result["summary"])


def stage_final(frame: pd.DataFrame, chosen: Sequence[str]) -> None:
    cfg = st_config.get("cane")
    september = sorted(m for m in frame.month.unique() if m.endswith("-09"))
    validation_date = None
    if september:
        candidates = sorted(d for d in frame.date.unique() if d.startswith(september[-1]))
        validation_date = candidates[len(candidates) // 2] if candidates else None

    booster, report, scores = st_train.fit(
        frame, chosen, validation_date=validation_date,
        num_boost_round=600, early_stopping_rounds=30,
    )
    log.info("final fit: %s", report)
    log.info("threshold %.2f -> %s", report.threshold, scores)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "fao_cane_xgb_model_v4.json"
    booster.save_model(str(model_path))
    sidecar = st_train.write_sidecar(
        model_path.with_suffix(".sidecar.json"), frame, chosen, report,
        crop="cane", static_window=cfg.static_window, outlier_caps=cfg.outlier_caps,
        extra={"validation_date": validation_date, "validation_scores": scores},
    )
    log.info("model -> %s", model_path)
    log.info("sidecar -> %s", sidecar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stability", "ablation", "grouped", "final", "all"])
    parser.add_argument("--rows-per-group", type=int, default=ROWS_PER_GROUP)
    parser.add_argument(
        "--features", nargs="*", default=None,
        help="feature names for the grouped and final stages; defaults to the six "
             "the deployed model uses, which cross-validation favoured over every "
             "alternative tried",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    frame = load(args.rows_per_group)

    # The stability study is a diagnostic, not a chooser. Its suggestion was tested
    # and lost: restricting the model to date-stable features cost more accuracy
    # than the drift did, and so did adding SWIR or extra indices. Unless told
    # otherwise, the grouped and final stages use the set the ablation favoured.
    if args.stage in ("stability", "all"):
        suggested = stage_stability(frame)
        log.info("stability suggests %s; using %s unless --features says otherwise",
                 suggested, LEGACY_FEATURES)
    chosen = list(args.features) if args.features else LEGACY_FEATURES
    log.info("feature set in use: %s", chosen)

    if args.stage in ("ablation", "all"):
        stage_ablation(frame, chosen)
    if args.stage in ("grouped", "all"):
        stage_grouped(frame, chosen)
    if args.stage in ("final", "all"):
        stage_final(frame, chosen)


if __name__ == "__main__":
    main()
