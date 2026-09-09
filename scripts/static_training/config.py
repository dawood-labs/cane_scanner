"""Per-crop configuration for date-robust static classifiers.

Adding a crop means adding an entry here. No other module in this package
contains crop-specific logic.

The key idea is `label_valid_days`. A label drawn on one acquisition date stays
true for some span around it, set by how long the crop occupies the field.
Sugarcane holds a field for roughly a year, so a November label still describes
the same field the previous June, and the same labelled polygons can be reused
against imagery from any date inside that span. Wheat occupies a field for four
months, so its span is far tighter. This is what makes multi-date training
possible without re-annotating anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Roots. Kept here so a machine move is a one-line change.
FAO_ROOT = Path("/mnt/c/Work_Work_Work/Python/Scripts/FAO")
CROPSCAN_ROOT = Path("/mnt/c/Work_Work_Work/Python/Scripts/cropscan")


@dataclass(frozen=True)
class LabelSource:
    """One set of label polygons with the AOI boundaries they were drawn inside."""

    name: str
    aoi_path: Path
    polygon_path: Path
    #: Date the labels describe, as YYYY-MM-DD. Label validity is measured from here.
    label_date: str
    #: Column on the polygon layer that marks a positive; None means every polygon
    #: in the file is a positive.
    positive_column: str | None = None
    positive_value: object = None


@dataclass(frozen=True)
class CropConfig:
    """Everything the training pipeline needs for one crop."""

    name: str
    crop_label: int
    background_label: int

    #: How far either side of its own date a label remains true, in days.
    label_valid_before_days: int
    label_valid_after_days: int

    #: Acquisition window the deployed model is allowed to run in. Mirrors
    #: Scripts_Dir/FAO/config.py so the two cannot drift apart unnoticed.
    static_window: Tuple[str, str]

    #: Ordered feature names, resolved by features.py. Training and inference
    #: both read this list, so they cannot disagree.
    feature_names: Sequence[str]

    #: Bands to request from sentinel.py when fetching imagery.
    fetch_bands: Sequence[str]

    #: Upper caps applied per source band to drop radiometric outliers. Derived
    #: from the distribution rather than invented; see docs in train.py.
    outlier_caps: Dict[str, int]

    label_sources: Sequence[LabelSource] = field(default_factory=tuple)

    #: Extra acquisition dates to sample, beyond each label's own date.
    #: Given as (year, month, day) strings; resolved against label validity.
    extra_sample_dates: Sequence[str] = field(default_factory=tuple)

    def is_label_valid_on(self, label_date: str, sample_date: str) -> bool:
        """Whether a label drawn on `label_date` still describes `sample_date`."""
        from datetime import date

        def _parse(value: str) -> date:
            year, month, day = (int(part) for part in value.split("-"))
            return date(year, month, day)

        delta = (_parse(sample_date) - _parse(label_date)).days
        return -self.label_valid_before_days <= delta <= self.label_valid_after_days


_CANE_STATIC_MODEL_DIR = FAO_ROOT / "cane" / "static_model"

CANE = CropConfig(
    name="cane",
    crop_label=1,
    background_label=4,
    # Sugarcane occupies the field for about twelve months. A label drawn in
    # early November is still true back to roughly the previous May, once the
    # ratoon or spring planting has established. Going further back reaches the
    # previous crop cycle, where the label stops meaning anything.
    label_valid_before_days=160,
    label_valid_after_days=30,
    static_window=("10-15", "11-25"),
    # Starting list. Phase 2 replaces this with whatever the stability study picks.
    feature_names=("B2", "B3", "B4", "B5", "B8", "NDVI"),
    fetch_bands=("blue", "green", "red", "rededge1", "nir", "ndvi"),
    # Reproduces master_filtered_scratch.parquet to within 0.05% of its row count.
    outlier_caps={"B2": 3976, "B3": 3940, "B4": 3000, "B5": 4345, "B8": 6000, "NDVI": 9571},
    label_sources=(),  # populated once the AOI and date list is confirmed
    extra_sample_dates=(),
)


# The remaining crops are declared but not yet trained through this package. They
# are here so the next one is a data exercise rather than a code exercise, and so
# the label-validity reasoning is written down while it is still fresh.

SPR_MAIZE = CropConfig(
    name="spr_maize",
    crop_label=1,
    background_label=8,  # spring maize uses 8, not 4; mirrors Scripts_Dir/FAO/config.py
    # Spring maize is sown in January or February and off the field by June, so it
    # occupies the ground for roughly five months. A label drawn in early May holds
    # back to about March, once the canopy is up, and forward to harvest. This is a
    # far tighter window than cane and the sampling has to respect it.
    label_valid_before_days=60,
    label_valid_after_days=35,
    static_window=("04-20", "05-20"),
    feature_names=("B2", "B3", "B4", "B5", "B8", "NDVI"),
    fetch_bands=("blue", "green", "red", "rededge1", "nir", "ndvi"),
    outlier_caps={},  # derive from its own training pool before first use
)

WHEAT = CropConfig(
    name="wheat",
    crop_label=1,
    background_label=4,
    # Wheat is sown in November and harvested in April. A mid-February label holds
    # back to about December and forward into March. Note the production config
    # gives wheat region-specific windows because Punjab and Sindh differ by two
    # weeks; that split will have to be carried into the sampling dates too.
    label_valid_before_days=75,
    label_valid_after_days=45,
    static_window=("01-20", "03-20"),
    feature_names=("B2", "B3", "B4", "B5", "B8", "NDVI"),
    fetch_bands=("blue", "green", "red", "rededge1", "nir", "ndvi"),
    outlier_caps={},
)


CROPS: Dict[str, CropConfig] = {
    "cane": CANE,
    "spr_maize": SPR_MAIZE,
    "wheat": WHEAT,
}


def get(crop: str) -> CropConfig:
    try:
        return CROPS[crop]
    except KeyError as exc:
        raise KeyError(
            f"no config for crop {crop!r}; known crops: {', '.join(sorted(CROPS))}"
        ) from exc


def sample_dates_for(crop: str, label_date: str) -> List[str]:
    """Which acquisition dates a label from `label_date` may legitimately be paired with."""
    cfg = get(crop)
    return [d for d in cfg.extra_sample_dates if cfg.is_label_valid_on(label_date, d)]
