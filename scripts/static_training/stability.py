"""Measure how much each candidate feature moves when the acquisition date changes.

A static classifier can only generalise across dates through features that hold
still across dates. This module ranks candidates by how far their central value
drifts, so the feature set is chosen from evidence rather than habit.

Two cautions, both learned the hard way on the cane data.

Measure drift inside each AOI and pool afterwards, never by comparing one date's
pooled median against another's. Cloud decides which AOIs are available on which
date, so an unpaired comparison reports the difference between two different sets
of districts and calls it seasonal drift. The paired and unpaired rankings of the
same cane data disagreed almost completely.

A steady feature set is not automatically a better one. On cane, the drift ranking
was real (raw blue moves several times as much as any normalised index) but
restricting the model to the steady features made it worse, not better, because it
threw away discriminative bands such as NIR. Once a model is trained across dates
it learns to handle the drift itself. Treat this module as a diagnostic that
explains a single-date model's failure, and let the ablation decide the feature set.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import features as feat


def per_date_centres(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    date_column: str = "date",
    label_column: Optional[str] = "label",
    label_value: object = None,
) -> pd.DataFrame:
    """Median of every feature, per acquisition date.

    Restricting to one class with `label_value` is usually what you want: the
    question is whether the *crop* looks the same on a new date, and a mixed-class
    median moves with class balance as well as with radiometry.
    """
    subset = frame
    if label_column is not None and label_value is not None:
        subset = subset[subset[label_column] == label_value]
    if subset.empty:
        raise ValueError("no rows left after filtering by label")

    rows: Dict[str, Dict[str, float]] = {}
    for date_value, group in subset.groupby(date_column):
        source = feat.source_from_dataframe(group)
        matrix = feat.build_matrix(source, feature_names)
        with np.errstate(invalid="ignore"):
            rows[str(date_value)] = {
                name: float(np.nanmedian(matrix[:, i]))
                for i, name in enumerate(feature_names)
            }
    return pd.DataFrame(rows).T.sort_index()


def deviation_table(
    centres: pd.DataFrame,
    reference: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Percent deviation of each feature on each date from a reference centre.

    With no reference given, the across-date median is used, which asks "how much
    does this feature wander" rather than "how far is it from training".
    """
    ref = centres.median(axis=0) if reference is None else reference
    ref = ref.reindex(centres.columns)
    if (ref == 0).any():
        raise ValueError(f"reference is zero for: {list(ref.index[ref == 0])}")
    return (centres - ref) / ref.abs() * 100.0


def rank_features(deviations: pd.DataFrame) -> pd.DataFrame:
    """Order features by how far they move between acquisitions, steadiest first.

    The spread across dates is the criterion, not the distance from any one date. A
    feature with a large but perfectly consistent seasonal offset is still unusable
    here: the model carries no date input, so it cannot tell which season a value
    belongs to, and the same number means two different things in August and
    November. What a date-agnostic model needs is features whose meaning holds.
    """
    worst = deviations.abs().max(axis=0)
    spread = deviations.max(axis=0) - deviations.min(axis=0)
    out = pd.DataFrame({"range_pct": spread, "worst_abs_pct": worst})
    return out.sort_values("range_pct")


def paired_deviations(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_value: object,
    reference_date: str,
    date_column: str = "date",
    label_column: str = "label",
    aoi_column: str = "aoi",
) -> pd.DataFrame:
    """Per-date drift measured inside each AOI, then pooled across AOIs.

    Comparing a pooled median on one date against a pooled median on another only
    isolates the date when both dates cover the same places. They rarely do: cloud
    decides which AOIs are available on which date, so a naive comparison reports
    the difference between two different sets of districts and calls it seasonal
    drift. Anchoring each AOI to its own reference date removes that confound.
    """
    subset = frame[frame[label_column] == label_value]
    per_aoi: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (aoi, date_value), group in subset.groupby([aoi_column, date_column]):
        source = feat.source_from_dataframe(group)
        matrix = feat.build_matrix(source, feature_names)
        with np.errstate(invalid="ignore"):
            per_aoi.setdefault(str(aoi), {})[str(date_value)] = {
                name: float(np.nanmedian(matrix[:, i]))
                for i, name in enumerate(feature_names)
            }

    records: List[Dict[str, object]] = []
    for aoi, by_date in per_aoi.items():
        base = by_date.get(reference_date)
        if base is None:
            continue  # this AOI has no reference acquisition to anchor against
        for date_value, centres in by_date.items():
            # The reference itself is kept as an explicit zero row, so the spread
            # across dates includes the anchor instead of floating free of it.
            row: Dict[str, object] = {"aoi": aoi, date_column: date_value}
            for name in feature_names:
                reference_value = base.get(name)
                if not reference_value:
                    continue
                row[name] = (centres[name] - reference_value) / abs(reference_value) * 100.0
            records.append(row)

    if not records:
        raise ValueError(
            f"no AOI has both {reference_date!r} and another date; cannot pair"
        )
    paired = pd.DataFrame(records)
    return paired.groupby(date_column)[list(feature_names)].median()


def report(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_value: object,
    reference_date: Optional[str] = None,
    date_column: str = "date",
    label_column: str = "label",
    aoi_column: Optional[str] = "aoi",
) -> Dict[str, pd.DataFrame]:
    """Centres, deviations and the ranking together.

    When `aoi_column` is given and a `reference_date` is named, drift is measured
    inside each AOI and then pooled, which is the only form that isolates the date.
    """
    centres = per_date_centres(
        frame, feature_names, date_column, label_column, label_value
    )
    if aoi_column and reference_date and aoi_column in frame.columns:
        deviations = paired_deviations(
            frame, feature_names, label_value, reference_date,
            date_column, label_column, aoi_column,
        )
    else:
        reference = centres.loc[reference_date] if reference_date else None
        deviations = deviation_table(centres, reference)
    return {
        "centres": centres,
        "deviations": deviations,
        "ranking": rank_features(deviations),
    }


def suggest_feature_set(
    ranking: pd.DataFrame,
    max_range_pct: float = 15.0,
    minimum: int = 4,
) -> List[str]:
    """Features steady enough to train on, keeping at least `minimum` of them."""
    keep = list(ranking.index[ranking["range_pct"] <= max_range_pct])
    if len(keep) < minimum:
        keep = list(ranking.index[:minimum])
    return keep
