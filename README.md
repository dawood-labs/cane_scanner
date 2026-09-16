# cane_scanner

Sugarcane mapping from Sentinel-2 over the Punjab and Sindh mill belt.

Two models work together. A RandomForest reads a season of smoothed NDVI and maps
where cane grew; an XGBoost classifier then reads a single clear image and strips
out the orchards, trees and cotton the first one picked up along the way.

## Why the static training package exists

The static classifier was fitted on seven acquisitions, all inside one three-week
window from 24 October to 8 November. Its training table carried no date column, so
that was invisible, and its train/test split was random per pixel, so neighbouring
pixels of the same field sat on both sides of it and the reported score could not
fall.

Run on a 30 August image it discarded about a third of a standing crop. The crop was
there: 98.4% of the pixels the time-series model called cane still had smoothed NDVI
above 0.55 that week.

`scripts/static_training/` is the rebuild. It is crop-agnostic; cane is the first
crop through it. See its own [README](scripts/static_training/README.md) for the
method, the numbers, and what adding a second crop requires.

## Layout

| Path | What it does |
|---|---|
| `scripts/sentinel.py` | Sentinel-2 acquisition: STAC search, cloud-aware date selection, tiled median and single-date mosaics |
| `scripts/geo_inference_workers.py` | Whittaker smoothing and chunked time-series inference |
| `scripts/static_training/` | Date-robust static classification: features, extraction, validation, training, domain guard, inference |
| `scripts/build_static_training_set.py` | Scout dates, export imagery, extract labelled pixels, merge, verify |
| `scripts/train_static_model.py` | Feature stability, ablation, grouped validation, final fit |
| `scripts/prove_static_model.py` | Date-by-date comparison of two models over one AOI |
| `scripts/run_v4_on_test_feature.py` | Apply a rebuilt model to the Al-Moiz test feature and diff against the old output |
| `scripts/swir_pilot.py` | Record of a tested and rejected idea; see its docstring before repeating it |
| `scripts/Model_Execution_Pipeline_v3.0.ipynb` | The end-to-end execution pipeline |

## What is not in this repository

Imagery, label polygons, field boundaries and trained model binaries are all
excluded. They are large and most of them are client data. Paths in the code point
at a working tree that lives outside version control.

Credentials are excluded too. The Google Cloud service-account key used for the data
catalogue sits beside the scripts during development and is matched by `.gitignore`;
keep it that way, and prefer pointing `GOOGLE_APPLICATION_CREDENTIALS` at a file
outside the repository.

## Requirements

Python 3.12 with `rasterio`, `geopandas`, `pyogrio`, `shapely`, `pyarrow`,
`pandas`, `numpy`, `scipy`, `xgboost`, `scikit-learn`, `pystac-client`,
`odc-stac` and `google-cloud-storage` (models are fetched from GCS by
`scripts/model_store.py`).

## A note on memory

The training table runs to about 59 million rows. Reading it whole needs several
gigabytes plus a copy, which is more than a typical WSL instance has. The merge and
load paths stream row groups instead and peak under 1 GB; keep it that way if you
extend them.
