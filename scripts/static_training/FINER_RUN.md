# A finer sieve and both August dates

Three changes to the deployed cane map, all aimed at cane the current settings throw
away, and one of them is not obviously safe.

## What changed

**A 0.15-acre sieve instead of 0.5.** The time-series map loses 98 acres of cane to a
20-pixel sieve before the static model ever sees it. A 0.15-acre field is real here:
the delineation has a median field size of 0.77 acres and 1,110 polygons under a
quarter acre.

**Both August dates, unioned.** Cane cut between the 10th and the 30th is bare on the
later image and standing on the earlier one, so taking either date as cane recovers
those fields.

**The static model re-run against the finer mask.** This was not optional. The deployed
static output's nodata is exactly the coarse sieve's background, so the extra ground had
never been classified at all and re-masking the old output would have recovered nothing.

## What it did

| | 30 Aug, 0.5-acre sieve | both dates, 0.15-acre sieve |
|---|---|---|
| cane in the raster | 7,237 ac | 8,557 ac |
| ground classified at all | 8,303 ac | 9,455 ac |
| cane polygons | 5,836 | 6,771 |
| cane in polygons | 6,027 ac | 6,900 ac |

Both runs label the same delineation, so `source_fid` lines the fields up exactly:

| | polygons | acres |
|---|---|---|
| cane in both runs | 4,989 | 8,125 |
| became cane | 811 | 1,048 |
| stopped being cane | 132 | 159 |

Of the 1,408 raster acres gained, 1,038 were seen only on 10 August and 332 on both.

## Why the gain is credible, and where it is not

Every gained acre is inside the time-series map already. The static model only runs
within the sieved RandomForest mask, so nothing here is new ground: it is ground the
season-long model already called cane, which 30 August rejected and 10 August accepted.
The static model's job is to remove trees, orchards and cotton from that, not to find
new fields, and it has not invented any.

The domain guard says the same thing from the other side. 30 August comes back at 0.65
IQR with a warning that recall will fall, its B5 and B4 high and its NDVI low; 10 August
comes back at 0.19, in distribution. The date carrying the gain is the healthier one.

What this cannot show is whether a particular gained field was cane. The union takes
either date's positives, so it takes either date's false positives too. Before this
setting ships, a sample of the 811 newly-cane fields should be looked at against the
10 August image.

## One number that got worse

Cane reaching a cane-labelled polygon fell from 83.3% to 80.6%. The union map holds
more cane and holds it in more scattered places, so more of it lands inside polygons
that are still mostly something else and take the majority label: 549 acres against
414. Absolute cane rose and capture efficiency fell, and both are true at once.

## Rerunning

    python3 run_finer_pipeline.py --stage all
    python3 label_field_polygons.py \
        --crop-map .../finer_p6/static_union_Cls_v4_strict_sieve_multiclass_p6.tif \
        --out .../finer_p6/field_labelling
    python3 compare_field_runs.py

Outputs live in `cane_2026/finer_p6/`, alongside rather than over the deployed run.
