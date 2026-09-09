# Date-robust static crop classification

## Why this exists

The FAO cane static classifier was fitted on seven acquisition dates: 26 Oct 2024,
24 Oct 2025, and 2, 3, 5, 7 and 8 Nov 2025. Its 51-million-row training table has no
date column, so nobody could see that, and its validation split was random per pixel,
so pixels from one field landed on both sides of it and the reported score could not
fall.

Run on a 30 Aug 2026 image of Al-Moiz Unit 1, it discarded a third of the fields the
time-series model had called cane. The crop was standing: 98.4% of those pixels still
had smoothed NDVI above 0.55. The model was simply looking at reflectances it had
never been shown.

Every other static model in `Scripts_Dir/FAO/config.py` has the same shape. Cane is
allowed 15 Oct to 25 Nov, spring maize 20 Apr to 20 May, wheat 20 Jan to 20 Mar. All
of them will behave this way outside their window.

## The two findings that shape the design

**Model quality tracks distance from the training distribution.** Measured across the
six cloud-free 2026 acquisitions over Al-Moiz, the ranking of distribution shift and
the ranking of recall are the same:

| Date | Shift (IQR) | Recall vs RF mask |
|---|---|---|
| 10 Aug 2026 | 0.25 | 85.0% |
| 06 Jul 2026 | 0.31 | 80.5% |
| 30 Aug 2026 | 0.91 | 61.8% |
| 04 Sep 2026 | 2.21 | 35.9% |

**The model leans on the features that move most.** Worst-case drift of each
candidate feature across July to September, against its training median:

| Feature | Worst drift |
|---|---|
| B4/B3 ratio | 8.4% |
| B8 | 16.2% |
| NDVI | 17.0% |
| GNDVI | 17.2% |
| NDRE | 18.2% |
| B3 | 53.0% |
| B4 | 69.4% |
| B2 | 138.1% |

Five of the six model inputs are raw DN bands, and the visible ones are the least
stable quantities on the list.

## Six principles

1. **Label-validity window.** A label drawn on one date stays true for as long as the
   crop occupies the field. Cane holds ground for a year, so a November label still
   describes the same field the previous June. Wheat holds it for five months. This
   is what allows multi-date training with no re-annotation, and it is the first
   thing to get right for a new crop.
2. **Multi-date sampling.** Extract the same labelled geometry at several dates
   inside that window.
3. **Date-stable features.** Prefer normalised ratios to raw DN, and prove it per
   crop with `stability.py` instead of assuming it.
4. **Grouped validation.** Hold out whole dates and whole AOIs. A random pixel split
   measures nothing that matters.
5. **Model sidecar.** Every model ships with its feature list, per-class training
   quantiles, chosen threshold and valid window.
6. **Inference-time guard.** Score the image against the sidecar before classifying,
   and refuse when it is out of distribution.

## Adding a crop

Supply four things, all in `config.py`:

- **label polygons and AOI boundaries** as vector layers
- **`label_valid_before_days` / `label_valid_after_days`**, from the crop calendar
- **`static_window`**, mirroring the production config so the two cannot drift
- **candidate sample dates** inside the validity window

Then run the four stages. Nothing else is crop-specific.

```bash
python3 build_static_training_set.py scout     # cloud-free date per AOI per window
python3 build_static_training_set.py export    # fetch imagery for those dates
python3 build_static_training_set.py extract   # labelled pixels, carrying date and AOI
python3 build_static_training_set.py merge     # label-transfer check, then one table

python3 train_static_model.py stability        # rank features by date stability
python3 train_static_model.py ablation         # does multi-date training actually help
python3 train_static_model.py grouped          # leave-one-date-out, leave-one-AOI-out
python3 train_static_model.py final            # fit and write the sidecar
```

## What actually helped, and what did not

Measured on cane, holding out whole months and whole mills:

| Change | Effect |
|---|---|
| Domain guard at inference | catches the failure that started this, before it ships |
| Multi-date training | about +1.5 points of August recall at matched precision |
| Restricting to date-stable features | worse, by a wide margin |
| Adding SWIR bands and indices | worse, -0.013 to -0.026 mean AUC |
| Adding NDRE and GNDVI | +0.005 mean AUC, inside the fold-to-fold noise |

The feature set the model already had turned out to be the right one. What was
wrong was the dates it was trained on and the absence of any check at inference.

August is also intrinsically harder than November, ROC-AUC around 0.88 against 0.90,
because cotton, rice and maize are green then too. No feature set tested closed that
gap, and it should not be expected to: it is the crop calendar, not the model.

## Two traps worth knowing

**Do not include AOIs that overlap.** Three of the v3 cane AOIs sit on top of v1
jdw1, LSM and sheikhoo by 79 to 95%. Including both would put the same ground in the
training pool under two names and make leave-one-AOI-out report a score it has not
earned.

**Score the guard against what the classifier will actually see.** A whole scene
contains every land cover and belongs against the pooled training distribution.
Pixels already narrowed to crop candidates by an upstream mask belong against the
crop distribution. Comparing a mixed scene to the crop-only distribution reports a
shift that is mostly just the other land cover, and refuses images that are fine.

**Keep field-edge pixels out of both classes.** A 10 m pixel straddling a boundary
is part crop and part whatever borders it. Excluding it from the positives is only
half the job: it then falls through into the background and teaches the model that
a crop edge looks like background. On one AOI that was 24% of all rows, and they
were exactly the ambiguous ones, all landing on the negative side.

**Cross-validate before believing any feature-set result.** Fold-to-fold AUC varies
by 0.03 to 0.05 on this data, so a single train-test split can show a 0.027 gain
from a feature set that six-fold cross-validation shows to be a 0.013 loss.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | per-crop settings, including the label-validity window |
| `features.py` | one definition of every feature, used by training and inference |
| `extract.py` | multi-date pixel extraction carrying date and AOI |
| `stability.py` | ranks features by how little they move between dates |
| `validate.py` | leave-one-date-out and leave-one-AOI-out scoring |
| `train.py` | fitting, threshold selection and the sidecar |
| `domain_check.py` | refuses imagery outside the training distribution |
| `inference.py` | classify a raster with the guard and the model's own threshold |
