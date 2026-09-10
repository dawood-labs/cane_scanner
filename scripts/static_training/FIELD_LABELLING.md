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

## Shape, not just area

Inspection in QGIS is what caught this, and nothing in the logs would have. The layer
came back full of shapes that no field has: long thin ribbons, L-shapes, hooks, and
whiskers a metre or two wide hanging several metres off otherwise sensible polygons.
A 0.15-acre floor does not catch any of them, because a ribbon can be long.

One test settles all three: **a shape that disappears when eroded by half a field's
width was never a field.** It is applied three different ways, because the right
remedy differs.

**Derived polygons get an opening.** Eroded by 10 m and dilated back, which is a
repair rather than a rejection: the tentacles come off and whatever solid core is left
survives as the field. Then the area floor again, then two shape gates, how much of
its own rotated rectangle the polygon fills and how much perimeter it carries for its
area. Of 7,548 blobs of crop lying outside the delineation, 455 survive the opening,
368 clear the area floor, and 354 are field-shaped. The 94% that vanish are the strip
of disagreement between a traced boundary and a 10 m raster, which is itself evidence
that the delineation is the better geometry.

**Slivering cuts are abandoned rather than emitted.** A cut running nearly parallel to
the parent's own edge shaves off a needle: one such piece was 11 pixels and ran the
length of the field. If either side of a proposed cut fails to look like a field, the
cut is not made and the polygon stays whole with a majority label, flagged `mixed, cut
would leave a sliver`. 805 cuts were abandoned this way. Abandoning beats discarding,
which would lose the ground altogether.

**Tails are trimmed off everything.** Eroding by 4 m removes any spur narrower than
8 m, but eroding and dilating rounds every corner, which would be a worse change than
the one being fixed. So the opened body is dilated slightly past its own radius and
intersected back with the original, restoring the true edges and corners while leaving
the tails outside.

The first version of this had a clause saying "leave a polygon alone if it would lose
more than 10%", meant to protect genuinely narrow fields. It protected exactly the
wrong ones. The polygons that lose most to despiking are the ones that are mostly
tail: one was 3.95 acres with a compactness of 0.021 and a solid body under it, and
the clause was what kept its tails on. There is no such clause now. A polygon that
vanishes entirely under a 4 m erosion was a line rather than a field, and 1,087 of
them go; the rest keep their bodies. Afterwards a compactness floor of 0.10 catches
what is still line-shaped, 74 of them, and a 0.05-acre floor catches bodies too small
to be anything, 1,343.

This runs on traced polygons too. The delineation owns the geometry, but a squiggle
two metres wide and forty long is not geometry anybody drew on purpose.

One threshold was also wrong in the other direction. The rectangle-fill floor sat at
0.55, and a triangle fills exactly half its own rotated rectangle: triangular fields
are real and were being rejected. It sits at 0.45 now.

## What comes out

| origin | polygons | acres | of which cane |
|---|---|---|---|
| delineation, untouched | 15,919 | 17,131 | 4,465 |
| split from a mixed polygon | 2,634 | 2,113 | 1,174 |
| derived from the crop map | 331 | 279 | 279 |
| **total** | **18,884** | **19,523** | **5,918** |

No invalid geometry and no overlapping area. Of the 7,237 acres of cane in the raster,
5,875 end up inside cane-labelled polygons. The rest is cane sitting in polygons that
are mostly something else, plus the 167 acres that were only ever ribbons and needles:
real in the raster, but with no shape worth handing to a client.

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
