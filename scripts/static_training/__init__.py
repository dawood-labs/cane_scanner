"""Date-robust static crop classification.

A static classifier fitted on one narrow acquisition window only works inside that
window. The cane model was fitted on 24 October to 8 November and lost a third of
its recall on a 30 August image of a standing crop. Every other static model in the
FAO config has the same shape and will behave the same way.

The methodology here is crop-agnostic. To add a crop you supply label polygons, AOI
boundaries, how long a label stays true, and candidate dates. Everything else is
shared:

    config        per-crop settings, including the label-validity window
    features      one definition of every feature, used by training and inference
    extract       multi-date pixel extraction carrying date and AOI
    stability     ranks features by how little they move between dates
    validate      leave-one-date-out and leave-one-AOI-out scoring
    train         fitting, threshold selection and the model sidecar
    domain_check  refuses imagery outside the model's training distribution
    inference     classify a raster with the guard and the model's own threshold
    phenology     shape-of-the-year features, for annual crop versus perennial
    orchard_filter  drops perennial fields from a crop map, after the time-series model
    orchard_mask    applies the never-cane orchard mask and protects intercropped blocks
"""

from . import (
    config,
    domain_check,
    extract,
    features,
    inference,
    orchard_filter,
    orchard_mask,
    phenology,
    stability,
    train,
    validate,
)

__all__ = [
    "config",
    "domain_check",
    "extract",
    "features",
    "inference",
    "orchard_filter",
    "orchard_mask",
    "phenology",
    "stability",
    "train",
    "validate",
]
