# Labelling field polygons from a 10 m crop map

The client wants field boundaries, not the fuzzy 10 m edges Sentinel produces. The
SAMGeo delineation drawn on basemap imagery has those boundaries. The crop map knows
what is growing but not where anything ends. So the whole pipeline rests on one rule:

> **the delineation owns the geometry, the crop map owns only the label.**

A raster edge is never allowed to become an output edge. Everything else follows.

## What the delineation is actually like

Three awkward cases were known going in. An audit found several more, which is why
`audit_field_labelling.py` exists and runs before anything is decided.

| | count | note |
|---|---|---|
| polygons | 17,074 | median 0.77 acres, 19,699 acres in total |
| invalid geometry | 8,666 | half the layer |
| multipart | 5,415 | |
| with holes | 1,270 | |
| overlapping pairs | 11,088 | the overlaps themselves are slivers, not acreage |
| under a quarter acre | 1,110 | |

`make_valid` turns those 8,666 into GeometryCollections, and a collection can hold a
MultiPolygon inside it. One `explode` leaves the nested Multis intact, and filtering
to `Polygon` then discards them: 2,997 polygons and 1,532 acres of crop, which come
back as orphan blobs with rasterised edges. Exploding until nothing is nested takes
the layer from 17,074 features to 36,000 clean polygons and makes the repair lossless,
19,693 acres retained of 19,699 covered.

## The cases, and what is done about each

**Cleanly one class.** 12,810 polygons. Labelled and passed through untouched.

**Two or three real fields in one polygon.** 21.5% of the layer, 5,213 acres. The
first hypothesis was that these are small polygons crossed by a fuzzy 10 m edge, in
which case majority-labelling would be right and cutting would be inventing detail.
That was wrong: the mixed share *rises* with size, 13.2% under half an acre against
26.2% at two to five acres, and a straight cut beats majority-labelling by a median
0.167 in purity, with 75.7% gaining more than 0.10. So they are genuinely
multi-field and cutting is justified.

The cut is a straight line along the parent field's own orientation, taken from its
minimum rotated rectangle, because subdivisions in this landscape run parallel to the
field's edges. Pixels are projected onto that axis and the perpendicular one and the
line goes where the two classes separate best. 2,110 polygons were cut this way. A cut
must raise purity by at least 0.08 to be made at all; 1,388 fell short and were
majority-labelled instead, because a cut that adds nothing invents a boundary the
ground does not have.

**Crop with no polygon over it.** 499 acres in 663 blocks above the 0.15-acre floor.
These get a polygon derived from the crop map, simplified at half a pixel so the
rasterised staircase does not ship. They are tagged `derived from crop map` so a
reviewer can tell them apart from traced geometry at a glance. Below 0.15 acres the
blobs are 10 m edge effects rather than fields and are left alone.

**Overlapping polygons.** Resolved in favour of the smaller polygon, which keeps the
finest delineation intact and trims the coarser one around it. Letting the larger win
would erase exactly the detail the basemap tracing was done for. 17,328 polygons were
trimmed; the slivers that fell out hold 6.3 acres between them.

**Too small to judge.** 1,800 polygons carry fewer than ten pixels, where a crop
fraction is not a measurement. They keep their geometry and are flagged rather than
guessed at.

## What comes out

| origin | polygons | acres | of which cane |
|---|---|---|---|
| delineation, untouched | 15,998 | 16,686 | 3,844 |
| split from a mixed polygon | 4,235 | 3,006 | 1,742 |
| derived from the crop map | 663 | 499 | 499 |
| **total** | **20,896** | **20,191** | **6,084** |

82% of the cut pieces reach 0.8 purity or better and 34% reach 0.9 or better. Of the
7,237 acres of cane in the raster, 6,084 end up inside cane-labelled polygons. The
missing 16% is cane sitting in polygons that are mostly something else and were
labelled accordingly: it is the price of refusing to let a raster edge become an
output edge, and it is visible per polygon in `crop_fraction` rather than hidden.

## Output

    data/test_data/almoiz_unit_1_test_feature_1/cane_2026/field_labelling/
        fields_labelled.gpkg    every polygon with its label, fraction and decision
        fields_cane.gpkg        the cane polygons alone, which is the deliverable

Every feature carries `origin` (delineation, split, or derived from crop map),
`decision`, `crop_fraction` and `pixels`, so any polygon in the deliverable can be
traced back to why it is there.

## Rerunning

    python3 audit_field_labelling.py       # measure before deciding
    python3 label_field_polygons.py        # produce the layers

`--threshold` sets the crop fraction at which a polygon counts as crop; it defaults to
0.5, which given the 0.85/0.15 clean bands only affects the polygons a cut could not
improve.
