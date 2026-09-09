"""Run a static classifier over a raster, with the guard and threshold applied.

This replaces the notebook's `classify_large_image`, which called `.predict()` and
so silently used XGBoost's default 0.5 cut even though training had chosen a
different threshold and thrown it away. It also had no way of noticing that the
image it was handed sat outside the distribution the model was fitted on.

What changes here:

  * features come from `features.py`, the same module training uses, so the two
    cannot drift apart
  * the decision threshold comes from the model sidecar
  * the probability raster is written alongside the labels, so a threshold can be
    revisited without re-running inference over the whole AOI
  * the domain guard runs first and can refuse
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import domain_check
from . import features as feat

log = logging.getLogger(__name__)

DEFAULT_BAND_ORDER = ("B2", "B3", "B4", "B5", "B8", "NDVI")


@dataclass
class ClassificationResult:
    label_path: Path
    probability_path: Optional[Path]
    verdict: Optional[domain_check.DomainVerdict]
    threshold: float
    classified_pixels: int
    positive_pixels: int

    @property
    def positive_fraction(self) -> float:
        return self.positive_pixels / self.classified_pixels if self.classified_pixels else 0.0


def load_model(model_path: Path | str):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def sidecar_for(model_path: Path | str) -> Path:
    """Where the sidecar lives for a given model file."""
    model_path = Path(model_path)
    return model_path.with_suffix(".sidecar.json")


def classify_raster(
    raster_path: Path | str,
    model_path: Path | str,
    out_label_path: Path | str,
    out_probability_path: Optional[Path | str] = None,
    mask_array: Optional[np.ndarray] = None,
    mask_path: Optional[Path | str] = None,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    positive_out: int = 1,
    background_out: int = 4,
    nodata_out: int = 255,
    threshold: Optional[float] = None,
    enforce_domain: bool = True,
    block_size: int = 1024,
    domain_sample_pixels: int = 4_000_000,
) -> ClassificationResult:
    """Classify one raster, honouring the model's own threshold and domain limits.

    `mask_array` restricts classification to selected pixels, which is how the
    static model is used in the cane pipeline: it refines what the time-series
    model already called crop, rather than mapping from scratch. `mask_path` is the
    same thing read from disk, which is what the notebook already produces through
    `create_aligned_mask`, so the two can be swapped without changing that step.
    """
    import rasterio

    if mask_path is not None and mask_array is None:
        with rasterio.open(mask_path) as mask_src:
            mask_array = mask_src.read(1) == 1

    sidecar_path = sidecar_for(model_path)
    sidecar = domain_check.load_sidecar(sidecar_path) if sidecar_path.exists() else None
    if sidecar is None:
        log.warning("no sidecar beside %s; falling back to threshold 0.5 and no domain guard",
                    model_path)

    feature_names = list(sidecar["feature_names"]) if sidecar else list(band_order)
    if threshold is None:
        threshold = float(sidecar["decision_threshold"]) if sidecar else 0.5

    booster = load_model(model_path)

    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

        verdict = None
        if sidecar is not None:
            # Score exactly the pixels that will be classified, so the verdict is
            # about the model's actual input rather than the whole scene. Read a
            # decimated copy for this: a district mosaic will not fit in memory, and
            # a median does not need every pixel to be stable.
            step = max(1, int(np.sqrt((height * width) / domain_sample_pixels)))
            probe = src.read(out_shape=(src.count, height // step, width // step))
            probe_mask = None
            if mask_array is not None:
                probe_mask = np.asarray(mask_array)[::step, ::step][
                    : probe.shape[1], : probe.shape[2]
                ]
            verdict = domain_check.check_raster(
                probe, list(band_order), sidecar, mask=probe_mask
            )
            log.info("domain check: %s", verdict)
            if enforce_domain and verdict.level == "refuse":
                raise RuntimeError(
                    f"refusing to classify {Path(raster_path).name}: {verdict}"
                )

        label_profile = {**profile, "count": 1, "dtype": "uint8", "nodata": nodata_out,
                         "compress": "lzw", "tiled": True}
        prob_profile = {**profile, "count": 1, "dtype": "float32", "nodata": np.nan,
                        "compress": "lzw", "tiled": True}

        Path(out_label_path).parent.mkdir(parents=True, exist_ok=True)
        prob_writer = None
        classified = positives = 0

        with rasterio.open(out_label_path, "w", **label_profile) as label_dst:
            if out_probability_path:
                prob_writer = rasterio.open(out_probability_path, "w", **prob_profile)
            try:
                import xgboost as xgb

                for row_start in range(0, height, block_size):
                    rows = min(block_size, height - row_start)
                    window = rasterio.windows.Window(0, row_start, width, rows)
                    block = src.read(window=window)

                    source = feat.source_from_raster(block, list(band_order))
                    matrix = feat.build_matrix(source, feature_names)
                    usable = feat.finite_mask(matrix) & (source[band_order[4]] > 0)
                    if mask_array is not None:
                        usable &= mask_array[row_start:row_start + rows].reshape(-1).astype(bool)

                    labels = np.full(rows * width, nodata_out, dtype=np.uint8)
                    probabilities = np.full(rows * width, np.nan, dtype=np.float32)

                    if usable.any():
                        dmatrix = xgb.DMatrix(matrix[usable], feature_names=feature_names)
                        prob = booster.predict(dmatrix)
                        probabilities[usable] = prob
                        labels[usable] = np.where(prob >= threshold, positive_out, background_out)
                        classified += int(usable.sum())
                        positives += int((prob >= threshold).sum())

                    label_dst.write(labels.reshape(rows, width), 1, window=window)
                    if prob_writer is not None:
                        prob_writer.write(probabilities.reshape(rows, width), 1, window=window)
            finally:
                if prob_writer is not None:
                    prob_writer.close()

    log.info("classified %d pixels, %d positive (%.1f%%) at threshold %.2f",
             classified, positives, 100 * positives / classified if classified else 0.0, threshold)
    return ClassificationResult(
        Path(out_label_path),
        Path(out_probability_path) if out_probability_path else None,
        verdict, threshold, classified, positives,
    )


def probability_summary(probability_path: Path | str, thresholds: Sequence[float]) -> Dict[float, int]:
    """Positive pixel count at several thresholds, from a written probability raster.

    Lets a threshold be re-chosen against ground truth without re-running the model.
    """
    import rasterio

    with rasterio.open(probability_path) as src:
        data = src.read(1)
    finite = data[np.isfinite(data)]
    return {float(t): int((finite >= t).sum()) for t in thresholds}
