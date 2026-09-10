# The orchard mask, and the cane growing inside the orchards

## What this replaces

An earlier attempt trained a detector to recognise orchards from the shape of the
NDVI year. It is written up in `ORCHARD_FINDINGS.md` and it does not work: pooled
leave-one-district-out AUC 0.681, and at the operating point that keeps 99% of cane
it recovers about 30 ha of orchard in Rahim Yar Khan while destroying 256 ha of cane.
Sentinel-1 was tried next and came out at 0.790 against NDVI's 0.722 on the same
objects, but it is weakest exactly where the confused orchard is (Multan 0.654,
Mirpur Khas 0.688) and at a conservative operating point it removes less orchard than
NDVI does. Neither is deployable.

The reason both fail is now clear, and it is not a modelling problem. Growers in the
mango belt plant sugarcane between the tree rows. Cane under a mango canopy never
goes bare and never crashes at harvest the way an open field does, so every phenology
feature we have reads it as woody. A model that learned to remove it would be
learning to delete real fields. The confusion is not an artefact to be trained away;
some of that ground genuinely is both.

## What is done instead

Nothing is inferred. The 59-polygon manual tree mask has 67,004 interior rings and
those rings are delineated orchards, the only orchard labels that exist. Of them,
66,341 fall between 0.5 and 100 ha and are treated as orchard blocks; the rest are
riverine and forest belts running up to 14,667 ha.

Each block is then screened against the 2025 national sugarcane scan, which is the
only independent record of where cane actually grows. A block the scan has never seen
carrying cane becomes mask: crop mapped there is removed by rule, with no model and
no threshold. A block that does overlap 2025 cane is not masked at all. It becomes a
protected layer instead, and inside it the residual phenology filter is forbidden to
act, because that is where intercropping lives.

Two more safeguards. Every kept block is shrunk by one Sentinel pixel, so a mask edge
can never reach into the field next door. And the 2025 cane polygons are cut out of
the final mask geometry, so no mapped cane sits inside the mask whatever the
block-level tolerance is set to.

## Does the screen actually see anything?

The screen assumes the 2025 scan is independent of the tree mask. If that scan had
been produced with this same mask applied, it would hold no cane inside any ring by
construction, the screen would find nothing to protect, and the mask would quietly
swallow whatever cane was really there.

`--stage circularity` settles it. Each ring is moved 500 m in a random direction and
screened again. A displaced ring sits on ordinary farmland, so its overlap is what the
district's cane density alone produces.

| district | cane in rings | cane 500 m away | ratio |
|---|---|---|---|
| Bhakkar | 0.01% | 0.31% | 0.03 |
| Chiniot | 1.15% | 24.85% | 0.05 |
| Khanewal | 0.01% | 1.04% | 0.01 |
| Layyah | 0.00% | 0.09% | 0.00 |
| Mirpur Khas | 0.40% | 9.21% | 0.04 |
| Rahim Yar Khan | 0.81% | 25.27% | 0.03 |
| Multan | 0.02% | 0.61% | 0.03 |

Rings hold twenty to a hundred times less cane than the land beside them, and the
ratio is never near zero-with-nothing-around-it: in Chiniot, 45% of blocks overlap
some cane, which a hard mask could not produce. The scan is not simply avoiding the
rings, and the screen means what it says.

One caveat worth keeping. Multan's low protection rate (0.6% of blocks) is not
evidence that nobody intercrops there: the scan sees very little cane anywhere near
Multan's orchards, 0.61% of the surrounding land. In a district that thin the screen
has little to work with either way, so Multan's mask rests on weaker evidence than
Rahim Yar Khan's or Chiniot's and is worth an eyeball before it ships.

## Where the intercropping actually is

Share of delineated orchard blocks that overlap 2025 cane, and are therefore
protected rather than masked:

| district | blocks | protected | share |
|---|---|---|---|
| Chiniot | 353 | 160 | 45.3% |
| Rahim Yar Khan | 2,847 | 845 | 29.7% |
| Tando Allahyar | 2,378 | 608 | 25.6% |
| Sargodha | 10,116 | 1,482 | 14.7% |
| Mirpur Khas | 4,450 | 633 | 14.2% |
| Khairpur | 4,586 | 457 | 10.0% |
| Shaheed Benazirabad | 2,388 | 234 | 9.8% |
| Naushahro Feroze | 3,230 | 297 | 9.2% |
| Matiari | 1,694 | 147 | 8.7% |
| Toba Tek Singh | 1,328 | 95 | 7.2% |
| Mandi Bahauddin | 2,127 | 148 | 7.0% |
| Hyderabad | 1,311 | 68 | 5.2% |
| Multan | 2,770 | 16 | 0.6% |
| Larkana | 1,879 | 8 | 0.4% |
| Bhakkar | 15,362 | 44 | 0.3% |
| Khanewal | 1,729 | 6 | 0.3% |
| Layyah | 4,942 | 6 | 0.1% |
| Sukkur | 2,849 | 0 | 0.0% |

Rahim Yar Khan is the mango belt and comes back with almost a third of its orchard
blocks carrying cane, which is the pattern the mill managers describe.

## Where it sits in the pipeline

1. time-series RandomForest produces the sieved cane map
2. **orchard mask** removes cane inside never-cane orchard blocks, by rule
3. **residual phenology filter**, forbidden to act inside the protected layer
4. static XGBoost v4 on a single-date mosaic, through the guarded inference path

Only step 2 is new and only step 3 changed. `static_training/orchard_mask.py` carries
both, `apply_mask` for the first and `rasterize_layer` for the second, which
`orchard_filter.filter_crop_map` now takes as `protect_polygons`.

## Rebuilding

    python3 build_orchard_exclusion_mask.py --stage rings
    python3 build_orchard_exclusion_mask.py --stage screen
    python3 build_orchard_exclusion_mask.py --stage circularity
    python3 build_orchard_exclusion_mask.py --stage verify

The mask is a static layer and only needs rebuilding when the tree mask or the
national scan is redrawn. It is district-scoped: 20 of the 59 districts have rings,
and a district with none was never digitised. Saying nothing about it is correct;
treating it as orchard-free would be a lie the pipeline would act on.
