# Orchards in the cane map: what was measured, and what it means

Short version: the problem is real and now quantified, the optical approach was
tried properly and does not solve it, and the reason it does not is structural
rather than a matter of tuning. **Do not deploy `orchard_detector.json`.**

## The problem, measured

The time-series RandomForest calls orchards cane. Running it directly over five
district chips and intersecting its output with the delineated orchard blocks:

| District | Time-series cane | Of which orchard | Share |
|---|---|---|---|
| Multan | 970 ha | 93.5 ha | 9.6% |
| Rahim Yar Khan | 4,074 ha | 313.5 ha | 7.7% |
| Mirpur Khas | 1,692 ha | 108.3 ha | 6.4% |
| Khanewal | 748 ha | 7.7 ha | 1.0% |
| Bhakkar | 4,164 ha | 22.5 ha | 0.5% |

In the mango belt roughly one hectare in ten of what the time-series stage calls
cane is orchard. The static model then removes most of it: in the finished 2025
national map only 314 ha of orchard survives across all five districts, 0.01% to
0.81% of each district's orchard area. So the second stage is working, but not
completely, and what it leaves is what shows up in QGIS.

## What was built

A detector on the shape of the year. Sugarcane is planted or ratooned, ramps, peaks
and collapses at harvest; a perennial sits high and flat. Fourteen features capture
that: amplitude, minimum, how many 8-day steps fall below a bare-canopy level, the
sharpest fall and rise, the longest unbroken green run, and where in the year the
peak and trough sit.

It works, on the wrong question. Trained to tell whole orchard blocks from cane
polygons it reaches AUC 0.99 and removes 96% of orchards while keeping 99% of cane.

## Why that number was not real

Most orchard blocks are already classified correctly by the time-series model. Only
part of each block fools it. Training on whole blocks therefore measures a contrast
that mostly does not need making.

Repeated against the ground that actually fools the model, orchard pixels the
time-series stage itself called cane, the same features give:

| | whole blocks | the ground that actually fools the model |
|---|---|---|
| best Cohen's d | 2.94 | 1.28 |
| pooled AUC | 0.99 | 0.681 |
| orchard removed at 99% cane kept | 96% | 1.5% |

The trade-off curve on the real target:

| Threshold | Cane kept | Orchard removed | Cane fields lost |
|---|---|---|---|
| 0.74 | 99.0% | 1.5% | 19 |
| 0.50 | 97.8% | 4.0% | 43 |
| 0.30 | 95.3% | 7.5% | 93 |
| 0.20 | 93.2% | 9.5% | 134 |

Removing a tenth of the orchard costs seven per cent of the cane. The time-series
model maps about ten times more cane than orchard, so in Rahim Yar Khan that trade
is 30 ha of orchard recovered against 256 ha of cane destroyed. It is not close.

## Why optical cannot do this

The orchard ground that reaches this filter is, by construction, the ground whose
NDVI trace already looked like cane to a model reading NDVI. A second model reading
the same signal has almost nothing left to separate. This is not a threshold that
needs moving or a feature that is missing; it is the same measurement being asked
the same question twice.

## What would

Radar. Sentinel-1 measures structure rather than greenness, and a woody canopy and a
dense grass differ there even when their NDVI traces agree. Sentinel-1 RTC covers
these AOIs on Planetary Computer with 67 acquisitions across the season, VV and VH,
and it is cloud-independent.

`sentinel.py` cannot fetch it as it stands. Its chain is hardcoded to optical in
three places that each need a radar branch: the STAC search filters on
`eo:cloud_cover`, which Sentinel-1 items do not carry, so every scene is dropped;
every load path requires an SCL asset; and the tile writer clips to uint16 at zero,
which destroys backscatter in dB, all of which is negative. Orbit direction also has
to be carried through, since ascending and descending passes cannot be averaged
together. Perhaps 400 to 600 lines, plus retraining.

## Three corrections made along the way

**37,464 ha was wrong; the number is 314 ha.** The first figure counted the area of
every orchard block that *touched* a cane polygon, not the orchard area actually
inside one. Most blocks touch cane only at an edge.

**The layer was wrong.** The national cane map is the finished product, with the
static model already applied. Measuring against it asked how many orchards survive
the existing cleanup, not how many arrive at it. Everything downstream inherited
that mistake until the time-series stage was run directly.

**The unit of decision was wrong.** Grouping by mapped cane polygon buries an orchard:
a mapped polygon runs to thousands of hectares against an orchard block's 3.7, so the
orchard never moves the polygon's median. Over five chips that version cleared no
field at all and removed nothing, while 53% of the orchard pixels in Rahim Yar Khan
cleared the gate on their own. `orchard_filter` now groups adjacent gate-passing
pixels into orchard-sized patches when no delineation layer is available.

## What is still worth keeping

- **`orchards_called_cane.gpkg`** and the RF-stage rasters under `rf_stage/`: they
  locate the problem on the ground and are useful whatever is built next.
- **`phenology.py`**: the feature definitions are sound and would carry straight over
  to a radar or combined model.
- **`orchard_filter.py`**: the two-condition design, a relative score plus an absolute
  gate, is what stops a filter removing its top slice from a map that holds nothing
  to remove. It removed 58 fields from an orchard-free AOI before the gate existed.
- **The measurement harness**: `test_orchard_filter_where_it_matters.py` asks both
  halves of the question, what is caught and what is lost, and either alone misleads.
