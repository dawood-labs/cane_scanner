# CLAUDE.md

Handover notes for whoever picks this project up next, human or Claude. Read this before
running anything: most of what is below was learned by breaking something.

## What this project is

Farmdar **CropScan** sugarcane mapping for sugar mill AOIs in Pakistan, from
Sentinel-2. This is CropScan work, not FAO work. For each mill AOI the client gets **six layers of cane field polygons**:

| folder | what it is |
|---|---|
| `01_timeseries` | RandomForest on a season of smoothed NDVI, alone |
| `02_static_<date>` | static XGBoost v4 on date 1, inside the time-series mask |
| `03_static_<date>` | the same on date 2 |
| `04_static_<date>` | the same on date 3 |
| `05_fused_union` | cane where any static date says cane |
| `06_fused_majority` | cane where 2 of the 3 dates agree |

**Recommended final layer: `06_fused_majority`.** Union inherits every date's false
positives. The client decision between union and majority is still open.

Two AOIs are done so far:

| AOI | data | static dates | state |
|---|---|---|---|
| Al-Moiz Unit 1 | `data/Al-Moiz-Unit-1-SM-AOI-2025/` | 10 Aug, 30 Aug, 9 Sep 2026 | done, **old code**: files named `fields_cane.gpkg`, no rescue pass (see below). Re-run the `label`, `gpkg`, `summary` stages to bring it up to date. |
| RYK | `RYK_data/RYK-AOI-2026/` | 10 Aug, 30 Aug, 14 Sep 2026 | done, current code |

Per-layer numbers live in `<aoi>/cane_2026/outputs/summary.csv`, not here: this
repository is public and the acreages are client results.

## Hard rules

- **The repo `dawood-labs/cane_scanner` is public.** Never commit client data.
  `data/`, `model_files/`, `RYK_data/`, rasters, shapefiles, GPKG and Parquet are all
  in `.gitignore`; put a new AOI's folder there too before the first `git add`.
- **`scripts/gcs_data_downloader_ee_farmdar.json` is a live GCP service-account key.**
  It must never be committed or pushed, to any remote. `.gitignore` matches it; never
  use `git add -f` or `git add .` without checking `git status` first.
- **Commit messages carry no Claude attribution** (no `Co-Authored-By: Claude`, no
  `Claude-Session:` line). Commits go out under the owner's identity.
- **Always `git fetch`/`git pull` before pushing.** Never force-push `main` without
  the owner saying so.
- **`dawood-labs/cropstack` is a different repo** holding FAO/cotton work on its `main`.
  Do not push this project's work there. It has never been run end to end.

## What a fresh clone needs

The code runs on a new AOI with no data from earlier AOIs. It needs:

1. **The new AOI shapefile and its field delineation shapefile**, passed with `--aoi`
   and `--delineation`. Always pass `--delineation`: the default points at an Al-Moiz
   path that will not exist.
2. **GCS read access to the models.** They are not in git (public repo, trained on
   client labels). They live in the private bucket and are downloaded on first use by
   `scripts/model_store.py` into `model_files/`, checksum-verified:

   | model | GCS |
   |---|---|
   | time-series RandomForest | `gs://farmdar_data_catalog/cropscan/cane/models/v4/best_rf_classifier_v4.joblib` |
   | static XGBoost v4 | `gs://farmdar_data_catalog/cropscan/cane/models/v4/fao_cane_xgb_model_v4.json` |
   | its sidecar (threshold, domain guard) | `gs://farmdar_data_catalog/cropscan/cane/models/v4/fao_cane_xgb_model_v4.sidecar.json` |

   Credentials, first found wins: `--gcs-key PATH` (or `GCS_KEY`),
   `GOOGLE_APPLICATION_CREDENTIALS`, a git-ignored `scripts/gcs_data_downloader*.json`,
   then `gcloud auth application-default login`. `python3 scripts/model_store.py`
   fetches all three up front (the RF is 187 MB, about 80 s).

Rules for the models in GCS:
- A new model goes in a **new version folder** (`v5/`) with its md5 added to `MODELS`
  in `model_store.py`. Never overwrite an object in `v4/`: old maps must stay
  reproducible, and the checksum would stop every run anyway.
- CropScan models live under `cropscan/cane/models/`. `fao_cane_model_file/` in the
  same bucket is **FAO work** (the old models `cropstack` uses); do not put CropScan
  files there or touch what is in it.
- The sidecar always travels with the static model. Without it inference silently falls
  back to threshold 0.5 and no domain guard; `ensure_model` downloads both together.

Imagery comes from public STAC catalogues (Planetary Computer, Earth Search), so the
GCS credentials are only for the models.

## Machine and memory

WSL2, 12 cores, about 11 GB RAM plus 8 GB swap. **Memory is the binding constraint,
not cores.** An OOM here kills the whole WSL session, so every heavy run goes under
`scripts/capped.sh`, a process-tree watchdog: it sums the RSS of the *whole process
tree* every second (children from `ProcessPoolExecutor` hold the memory, not the
parent), `kill -9`s the tree past `LIMIT_MB` (default 10000) and exits 99 with a
`WATCHDOG:` line. It always prints `PEAK_RSS_MB`; read that before raising any jobs
or tile-size flag. On a machine with a different amount of RAM, set `LIMIT_MB` to
roughly 1 GB under what `free -m` reports available.

Settings that are known to fit:

| stage | flags | peak |
|---|---|---|
| timeseries | `--jobs 6 --fetch-jobs 3` | fits under 10 GB |
| label, Al-Moiz (139 polygons/km²) | `--label-jobs 8 --tile-km 12` (defaults) | fits |
| label, RYK (249 polygons/km²) | `--label-jobs 6 --tile-km 8` | 6.5 GB |

Labelling memory scales with **delineation polygons per tile**, not tile count. RYK at
12 km x 8 jobs hit the watchdog at 10.1 GB. For a new AOI, compute polygons/km² of its
delineation first and shrink `--tile-km` if it is denser than Al-Moiz.

Shell traps met in practice:
- Never put `pkill -f "<pattern>"` and a relaunch of the same command in one shell
  command: the pattern matches the shell itself and kills it (exit 144).
- `--stage label` runs only labelling. `gpkg` and `summary` are separate stages.

## Running a new AOI

Inputs: the AOI shapefile and the field delineation shapefile (SAMGeo, basemap traced).

### 1. Pick the three static dates

`sentinel.select_static_dates(aoi, start, end, n_dates=1, cloud_metric="aoi")`
scores every Sentinel-2 date in a window by SCL cloud over the AOI and returns
`candidates` with `date`, `valid_pct`, `cloud_pct`, `coverage_pct`. The rule used for
RYK, one date per window:

| window | prefer the date nearest |
|---|---|
| 1 Aug to 15 Aug | 1 Aug |
| 20 Aug to 31 Aug | 31 Aug |
| 5 Sep to today | today |

Within a window keep candidates whose `valid_pct` is within 2 points of the best, then
take the one closest to the preferred edge. Cloud-free first, edge second. (For RYK,
2 Aug lost to 10 Aug at 29.6% cloud.) Eyeball the chosen scenes before a long run.

### 2. Run the stages

```bash
cd scripts
A=/path/to/AOI.shp
D=/path/to/Delineation.shp
DATES=2026-08-10,2026-08-30,2026-09-14
COMMON="--aoi $A --delineation $D --static-dates $DATES"
RUN="bash capped.sh python3 run_full_aoi.py"

$RUN --stage timeseries --jobs 6 --fetch-jobs 3 $COMMON
$RUN --stage sieve   $COMMON
$RUN --stage static  $COMMON
$RUN --stage fuse    $COMMON
$RUN --stage label   --label-jobs 6 --tile-km 8 $COMMON
$RUN --stage gpkg    $COMMON
$RUN --stage summary $COMMON
```

`--stage all` runs them in order. Every stage skips work already on disk, so a killed
run is resumed by running the same command again. Output goes to
`<AOI parent>/cane_2026/` unless `--out` is given. The time-series window
(`NDVI_START/END`, `INFER_START/END` at the top of `run_full_aoi.py`) is hard-coded to
the 2025-26 season; update it for a new season.

Rough times on RYK: timeseries ~35 min (fresh fetch), static ~2.5 min a date, fuse
seconds, labelling ~6.7 min a layer (~40 min for six), GPKG ~1.3 min a layer, summary
a few minutes.

### 3. Deliver

`outputs/<folder>/<folder>_cane.gpkg` is the deliverable, `<folder>_labelled.gpkg`
holds every polygon with its label. The layer *inside* each GPKG has the same name as
the file, because QGIS shows the internal layer name: six files that all contained a
layer called `fields_cane` were indistinguishable in QGIS.

## Pipeline, stage by stage

**timeseries** (`timeseries_pipeline.py`, `geo_inference_workers.py`, `sentinel.py`):
STAC fetch in 0.1° tiles (tiles missing the AOI are dropped), then the AOI is burned
into each raw tile as zeros before inference (`burn_aoi_into_tiles`, marked with a
`.aoiclip` sidecar). Zero is already the "missing" encoding, so outside pixels skip
the Whittaker smoothing and `predict` for free. Model:
`model_files/best_rf_classifier_v4.joblib`.

**sieve**: clip the map to the AOI polygon (a check now; the burn already did the
work), then a directional sieve at **6 px = 0.15 acres**
(`static_training/sieve.py`).

**static** (`static_pipeline.py`, `static_training/inference.py`): XGBoost
`fao_cane_xgb_model_v4.json` with its sidecar, **threshold 0.35**, only inside the
time-series cane mask, with the **domain guard** (`static_training/domain_check.py`)
refusing imagery outside the training distribution. See
`scripts/static_training/README.md` for why v4 exists: the old model was trained on
Oct/Nov dates only and failed in August.

**fuse**: union (`votes >= 1`) and majority (`votes >= 2`), sieved the same way.
`votes >= 2` is hard-coded for three dates; change it if the date count changes.

**label** (`label_field_polygons_tiled.py` -> `label_field_polygons.py`): the core
rule is **the delineation owns the geometry, the raster owns only the label**. Full
design and numbers in `scripts/static_training/FIELD_LABELLING.md`. In short:
- the delineation is repaired (`make_valid`, explode until nothing is nested,
  despike tails) and the repair is **cached** in `<delineation dir>/repair_cache/`,
  keyed on source mtime, tile window and `REPAIR_VERSION`. Bump `REPAIR_VERSION` by
  hand whenever `repair()` changes, or the cache silently serves old behaviour.
  `--rebuild-cache` forces it.
- crop fraction >= 0.85 or <= 0.15 is clean; mixed polygons get a straight cut along
  the field's long axis if it raises purity by >= 0.08, otherwise a majority label.
- overlaps resolve in favour of the smaller polygon.
- **orphan cane** (no delineation polygon over it) becomes a polygon derived from the
  raster: opening at radius 7 m, area floor 0.15 acres, then shape gates
  rectangularity >= 0.45 and Polsby-Popper >= 0.18.
- **rescue pass** (`_rescue`): a blob of >= 1 acre that fails the gates is eroded
  harder (14 m, then 21 m); each core is grown back clipped to the blob, largest
  first, and kept if it passes the gates. Added because a 35-acre cane field on RYK
  (pixel index 18487161) was joined to lace between traced fields and the whole blob
  was rejected. It recovers 170 to 271 acres per RYK layer with no new overlap.
- tiles run as separate processes (that is what actually returns memory), each reads
  a 1 km margin, and each polygon is owned by the tile holding its representative
  point.

Every output polygon carries `origin` (delineation / split / derived from crop map),
`decision`, `crop_fraction`, `pixels`, `is_crop`, `acres`.

**summary**: `outputs/summary.csv` with raster acres, polygon count, polygon acres,
capture % and **double counted** acres, plus a date-disagreement breakdown printed to
the log. Double counting is small (single-digit acres per layer), so no dissolve is
needed.

## Performance lessons

- **Never `union_all` a whole layer to measure overlap.** Use
  `label_field_polygons.overlap_acres()`: STRtree pair query + pairwise intersection
  area. Same answer, about 20x faster (26 s against 521 s); labelling a layer went
  from ~16 to 6.7 min once the tiled merge stopped unioning too.
- Cache any slow stage whose inputs do not change (the repair cache took 273 s to 2 s).
- Write GeoParquet during the run and GeoPackage once at the end (`--no-gpkg`, then
  `--stage gpkg`): Parquet takes seconds, GPKG about a minute a layer.
- Time stages before optimising: `cane_2026/stage_timings.json` records every stage.

## Known open items

- Al-Moiz outputs predate the rescue pass and the file/layer renaming; re-label it.
- Union vs majority as the final layer: waiting on a client/owner decision.
- Review `derived from crop map` and `crop inside another polygon` polygons in QGIS.
- The orchard never-cane mask (`static_training/orchard_mask.py`) is not in
  `run_full_aoi.py`. On RYK it would touch only ~0.05% of the majority layer, so it
  was left off; check it again for orchard-heavy AOIs.
- Labelling could skip tiles with no delineation up front (small win).
- **Drift hazard:** notebook `Model_Execution_Pipeline_v3.0.ipynb` cell 4 writes
  `scripts/geo_inference_workers.py` with `%%writefile`, and the two have already
  diverged. Running that cell overwrites the file on disk. Reconcile before editing
  either.
- There is an unexplained ~2,013-acre raster gap between `cropstack` and this repo on
  the same AOI.
- The docstring at the top of `run_full_aoi.py` still describes the older
  four-date/five-layer run; the code is current.

## Other docs

| file | covers |
|---|---|
| `README.md` | repository overview and layout |
| `scripts/static_training/README.md` | static v4 model: why, method, results |
| `scripts/static_training/FIELD_LABELLING.md` | labelling design in detail |
| `scripts/static_training/ORCHARD_MASK.md`, `ORCHARD_FINDINGS.md` | orchards |
| `scripts/static_training/SENTINEL1_DESIGN.md` | radar experiment, not deployed |
| `scripts/static_training/NOTEBOOK_INTEGRATION.md` | wiring v4 into the notebook |
