"""Refuse to classify an image that sits outside the model's training distribution.

The 30 August 2026 failure was silent. The model returned confident labels for an
image whose blue reflectance sat above the 75th percentile of its training cane
distribution, and nothing in the pipeline noticed. This module is the check that
would have caught it.

The score is how far the image's per-feature medians sit from the training medians
in the sidecar, measured in interquartile ranges of the training distribution and
combined as a root mean square.

Calibrated on Al-Moiz Unit 1 against the deployed cane model, the score orders the
six available 2026 acquisitions the same way their recall does, and the default
thresholds land where the accuracy does:

    date          shift   verdict   recall vs RF mask
    10 Aug 2026    0.25   ok                    85.0%
    06 Jul 2026    0.31   ok                    80.5%
    30 Aug 2026    0.91   warn                  61.8%
    06 Jun 2026    2.08   refuse                30.8%
    04 Sep 2026    2.21   refuse                35.9%
    07 May 2026    2.53   refuse                14.2%
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import features as feat

#: Thresholds are in interquartile ranges of the training distribution, not
#: percent. A percentage cannot be compared across features, because a 20% move in
#: blue and a 20% move in NDVI mean completely different things. Measuring the
#: shift against the spread the model actually saw during training makes the
#: features commensurable and the thresholds transferable to another crop.
DEFAULT_REFUSE_ABOVE = 1.0
DEFAULT_WARN_ABOVE = 0.5


@dataclass
class DomainVerdict:
    score: float
    per_feature: Dict[str, float]
    level: str  # "ok", "warn" or "refuse"
    message: str
    #: Percent deviation per feature, kept for reporting; not used for the verdict.
    per_feature_pct: Dict[str, float] = None  # type: ignore[assignment]

    @property
    def ok(self) -> bool:
        return self.level == "ok"

    def __str__(self) -> str:  # pragma: no cover - display only
        worst = sorted(self.per_feature.items(), key=lambda kv: -abs(kv[1]))[:3]
        detail = ", ".join(f"{name} {value:+.2f}" for name, value in worst)
        return f"[{self.level}] shift {self.score:.2f} IQR ({detail}) - {self.message}"


def load_sidecar(path: Path | str) -> dict:
    """Read the JSON written next to a trained model by train.py."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _reference_quantiles(sidecar: dict, reference: str) -> Dict[str, Dict[str, float]]:
    """Training quantiles to compare against.

    "pooled" is not stored directly by older sidecars, so it is approximated by
    blending the two class distributions on the class balance the model was fitted
    with. That is exact for the median only when the classes overlap, but it is far
    closer to a mixed scene than either class alone.
    """
    quantiles = sidecar["training_quantiles"]
    if reference in quantiles:
        return quantiles[reference]
    if reference != "pooled":
        raise ValueError(f"unknown reference {reference!r}")

    positive, negative = quantiles["positive"], quantiles["negative"]
    weight = 1.0 / (1.0 + float(sidecar.get("report", {}).get("scale_pos_weight", 1.0) or 1.0))
    blended: Dict[str, Dict[str, float]] = {}
    for name, pos_stats in positive.items():
        neg_stats = negative.get(name)
        if not neg_stats:
            continue
        blended[name] = {
            key: weight * pos_stats[key] + (1.0 - weight) * neg_stats[key]
            for key in pos_stats
            if key in neg_stats
        }
    return blended


def score_matrix(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    training_quantiles: Dict[str, Dict[str, float]],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Shift of each feature's median from training, in IQR units and in percent."""
    in_iqr: Dict[str, float] = {}
    in_pct: Dict[str, float] = {}
    for index, name in enumerate(feature_names):
        stats = training_quantiles.get(name)
        if not stats:
            continue
        median = stats.get("q50")
        spread = stats.get("q75", 0.0) - stats.get("q25", 0.0)
        if median in (None, 0) or spread <= 0:
            continue
        with np.errstate(invalid="ignore"):
            observed = float(np.nanmedian(matrix[:, index]))
        in_iqr[name] = (observed - median) / spread
        in_pct[name] = (observed - median) / abs(median) * 100.0
    return in_iqr, in_pct


def check(
    matrix: np.ndarray,
    sidecar: dict,
    warn_above: float = DEFAULT_WARN_ABOVE,
    refuse_above: float = DEFAULT_REFUSE_ABOVE,
    reference: str = "pooled",
) -> DomainVerdict:
    """Compare a feature matrix against the training distribution in `sidecar`.

    `reference` must match what the matrix contains. A whole scene holds every land
    cover and belongs against "pooled"; pixels already narrowed to crop candidates
    by an upstream mask belong against "positive". Scoring a mixed scene against the
    crop-only distribution reports a shift that is mostly just the other land cover,
    and refuses images that are in fact fine.
    """
    feature_names = sidecar["feature_names"]
    quantiles = _reference_quantiles(sidecar, reference)
    in_iqr, in_pct = score_matrix(matrix, feature_names, quantiles)
    if not in_iqr:
        raise ValueError("sidecar carries no usable training quantiles")

    # Root mean square rather than a plain mean: one badly displaced feature is
    # enough to move a tree model into a region it never saw, and averaging would
    # let the steady features hide it.
    score = float(np.sqrt(np.mean([v * v for v in in_iqr.values()])))
    if score > refuse_above:
        level, message = "refuse", (
            "image is outside the distribution this model was fitted on; "
            "pick a date inside the crop's static window"
        )
    elif score > warn_above:
        level, message = "warn", (
            "image is drifting away from the training distribution; "
            "expect recall to fall"
        )
    else:
        level, message = "ok", "image is in distribution"
    return DomainVerdict(score, in_iqr, level, message, in_pct)


def check_raster(
    array: np.ndarray,
    band_order: Sequence[str],
    sidecar: dict,
    sample: Optional[int] = 2_000_000,
    valid_band: str = "B8",
    mask: Optional[np.ndarray] = None,
    reference: Optional[str] = None,
    **thresholds,
) -> DomainVerdict:
    """Same check, straight from a (band, row, col) raster read.

    Pixels whose `valid_band` is zero are treated as no-data and excluded, which
    matches how the rest of the pipeline reads these rasters. Pass the same `mask`
    the classifier will use, so the check scores exactly the pixels that will be
    classified; the reference distribution then defaults to the crop class, since a
    mask exists to select crop candidates.
    """
    source = feat.source_from_raster(array, band_order)
    keep = source[valid_band] > 0
    if mask is not None:
        keep &= np.asarray(mask).reshape(-1).astype(bool)
    if reference is None:
        reference = "positive" if mask is not None else "pooled"
    thresholds["reference"] = reference
    if not keep.any():
        raise ValueError("raster has no valid pixels")
    source = {name: values[keep] for name, values in source.items()}

    count = len(next(iter(source.values())))
    if sample and count > sample:
        picks = np.random.default_rng(0).choice(count, size=sample, replace=False)
        source = {name: values[picks] for name, values in source.items()}

    matrix = feat.build_matrix(source, sidecar["feature_names"])
    return check(matrix, sidecar, **thresholds)
