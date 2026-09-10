# Sentinel-1 support in `scripts/sentinel.py` — integration design

Status: design only. Nothing implemented. Build order in §7 is safe to execute as soon as
the radar-usefulness experiment returns "yes".

Target file: `/mnt/c/Work_Work_Work/Python/Scripts/cropscan/scripts/sentinel.py` (2196 lines)
Also touched: `/mnt/c/Work_Work_Work/Python/Scripts/cropscan/scripts/geo_inference_workers.py`

Scope: **time-series path only** (`fetch_sentinel_imagery` → `_process_tile` →
`_composite_window`). The static/single-date path is explicitly out of scope — see §3.9.

---

## 0. Measured facts (queried 2026-09-10, not taken from docs)

AOI bbox `70.29, 28.58, 70.40, 28.66`, window `2025-11-24 → 2026-09-09`,
collection `sentinel-1-rtc` on Planetary Computer.

| Fact | Measured value |
|---|---|
| Items returned | **45** |
| AOI coverage per item | **100 % on all 45** (min = median = 100 %); single slice, no mosaicking needed |
| Tracks present | **2**: `ascending` / `sat:relative_orbit` **71**, and `descending` / rel. orbit **5** |
| Ascending (71) | **23 items, 23 unique dates**, 2025-11-29 → 2026-09-08 |
| Descending (5) | **22 items, 22 unique dates**, 2025-11-25 → 2026-09-04 |
| Revisit gaps within a track | **7, 12 and 24 days** (not a clean 6/12 — 24 d = a dropped acquisition) |
| Platforms | `sentinel-1a` ×33, `sentinel-1d` ×12 (S1D ramp-up causes the 7 d gaps) |
| Local solar overpass | ascending **≈18:02**, descending **≈05:57** |
| Polarisations | `['VV','VH']` on **all 45** items (no HH/HV here, though the collection declares them) |
| Data assets | exactly **`vv`**, **`vh`** (COG). Plus `tilejson`, `rendered_preview` only. **No SCL/mask/valid-data asset of any kind.** |
| Asset dtype / nodata | **float32**, **nodata = -32768.0** (declared in both `raster:bands` and the file) |
| Native grid | **EPSG:32642, 10.0 m**, 512×512 blocks, overviews 2–64 |
| `eo:cloud_cover` | **absent** from every item |
| `sar:looks_equivalent_number` (ENL) | **4.4** |
| gamma0 over the AOI (VV) | linear p1/p50/p99 = **0.032 / 0.165 / 0.471**; dB = **-15.0 / -7.8 / -3.3**; max 27.4 (=14.4 dB) |
| gamma0 over the AOI (VH) | linear p1/p50/p99 = **0.0054 / 0.0328 / 0.0954**; dB = **-22.7 / -14.8 / -10.2** |
| Collection temporal extent | 2014-10-10 → open |

**Fallback provider: there is none.** Element84 Earth Search has no `sentinel-1-rtc`; it
only carries `sentinel-1-grd` (50 items over the same AOI/window — uncalibrated, not
terrain-corrected, and *not* interchangeable with RTC). `providers` must default to
`("mpc",)` for radar; silently falling back to Earth Search would swap gamma0 RTC for
raw GRD DN mid-run.

### 0.1 Two measurements that change the design

**(a) The 8-day median never averages more than one radar scene.**
Per-track occupancy of the 37 8-day windows over this range:

| step (d) | asc: windows / empty / 1 scene / ≥2 | desc: windows / empty / 1 / ≥2 |
|---|---|---|
| 6  | 49 / **26** / 23 / 0 | 49 / **27** / 22 / 0 |
| **8**  | 37 / **14** / 23 / **0** | 37 / **15** / 22 / **0** |
| **12** | 25 / **2** / 23 / **0** | 25 / **4** / 20 / 1 |
| 16 | 19 / 1 / 13 / **5** | 19 / 3 / 10 / **6** |
| 24 | 13 / 0 / 3 / **10** | 13 / 1 / 3 / 9 |

So at `step=8` the "8-day multi-scene median" that the optical path relies on **does not
exist for radar**: every non-empty window holds exactly one scene, and 14 of 37 windows
(38 %) are empty. This is the single most consequential finding — it invalidates the usual
"the median already handles speckle" argument (§6) and dictates `step=12` (§3.8).

**(b) The STAC `query` extension fails silently on unknown/absent fields.** Verified:

```
no query                    -> 45 items
eo:cloud_cover  lt 80       ->  0 items      <- the known bug, confirmed
sat:orbit_state eq ascending-> 23
sat:relative_orbit eq 71    -> 23
asc + 71 together           -> 23
sat:nonsense    eq 1        ->  0 items      <- typo == empty result, no error
```

The orbit query works. But a typo in a property name returns zero items with no
exception — the same failure shape as the cloud filter. Every STAC query built for radar
must be covered by a test that asserts a **non-zero** item count.

---

## 1. Verification of the seven known breakages

| # | Claim | Verdict | Detail |
|---|---|---|---|
| 1 | `_PROVIDERS` (L233) has one `collection`, plus `scl_asset`/`cloud_field` read unguarded | **Confirmed, and wider than stated** | `scl_asset` at L237, L245; `cloud_field` at L238, L246. Unguarded reads at **L595, 600, 605, 612, 616, 618, 619** (`_composite_window`), **L1213** (`_search_items`), **L1291, 1296, 1298** (`_scan_date_masks`), **L1443** (`select_static_dates`), **L1562, 1569, 1574, 1578** (`_layer_match_coeffs`), **L1643, 1656, 1658, 1664, 1673** (`_static_layers`). The brief's list (595/598/1291/1562/1643) misses 600, 605, 612, 616, 1213 and 1443. |
| 2 | Cloud filter drops every S1 scene silently | **Confirmed empirically** | `query={"eo:cloud_cover":{"lt":80}}` → **0 items** vs 45 without. No exception, no warning. `_composite_window` then takes the `if not items` branch at L607 and returns an all-NaN window — which downstream becomes an all-zero band. |
| 3 | `_composite_window` (L586) always loads SCL and applies `_scl_keep_mask` | **Confirmed** | SCL appended to the band list at L616, mask applied L618. `sentinel-1-rtc` has no SCL asset at all, so odc-stac raises on the missing band — this one is *loud*, and it is the only one of the seven that is. |
| 4 | `_harmonize_offset` subtracts the S2 baseline offset | **Confirmed — and worse than "meaningless"** | S1 items carry no `s2:processing_baseline`, so L470-473 falls through to the date branch; every scene is ≥ 2022-01-25, so `newer == len(items)` and the function **returns 1000** for MPC. L620 then computes `clip(0.165 - 1000, 0, None)` → **0.0 for every pixel**. Silent, total data loss. |
| 5 | `_process_tile` (L753) `np.clip(np.nan_to_num(stack, nan=0.0), 0, 65535).astype("uint16")` at **L785** destroys backscatter | **Confirmed, measured** | Ran the cast on a real AOI load: linear gamma0 (p1 0.032 → p99 0.471) maps to `uint16` values `[0,1,2,3,4]` — **>99 % of pixels become 0**. dB (all negative except bright scatterers) is clipped to 0 outright. Additionally **L787-790 sets no `nodata` key at all** — the optical tiles rely on the 0-means-nodata convention, which is unusable for dB where 0 dB is a real, common value. |
| 6 | `SENTINEL2_BANDS` / `resolve_bands` (L283) raise on unknown band names | **Confirmed** | `resolve_bands(["vv"])` → `ValueError: Unknown band 'vv'`. This is a **loud** failure and is the reason the static path needs no defensive work (§3.9). |
| 7 | `parse_stac_bands` matches only `red_`/`nir_` | **Confirmed, silent** | `geo_inference_workers.py` L11-23. On radar descriptions both index lists come back empty, and the guard `assert len(red_idx) == len(nir_idx)` is satisfied by `0 == 0`. `dates` is `[]`, so `n_timesteps = 0`, `get_penalty_matrix(0, …)` yields a 0×0 matrix and the whole tile is smoothed/predicted over an empty band axis. No exception at the parse site. |

### 1.1 Four additional breakages not in the brief

| # | Site | Problem |
|---|---|---|
| 8 | `_build_vrt_manual` **L873** | `gdal_dt = {"uint16":…, "float32":…, "uint8":…}.get(dtype, "UInt16")` — an `int16` tile falls to the **`"UInt16"` default**. A -8.00 dB pixel stored as `-800` is then read through the VRT as **64736**. Silent, and only in the fallback path (i.e. only on machines without `gdalbuildvrt`), so it will not reproduce on a dev box that has GDAL CLI installed. |
| 9 | `_clip_mosaic_to_geom` **L896** default `nodata=0`, and `fetch_sentinel_imagery` **L1048** calls it without passing one | Out-of-polygon pixels in `sentinel_clipped.tif` are filled with **0**, which in dB is a perfectly plausible bright-target value. The AOI boundary becomes a ring of fake +0 dB rather than nodata. |
| 10 | `_build_vrt` **L1039** globs the fixed pattern `sentinel_*m_tile_*.tif`, and `_process_tile` **L760** writes that name | An S1 run into an out_dir that already holds S2 tiles (or two S1 tracks into one dir) mixes incompatible tiles into one VRT — different band counts, different dtypes, different date labels. |
| 11 | `_composite_window` L616 uses `resampling="nearest"` | Correct for the categorical SCL band and harmless for optical. For radar it means the UTM→WGS84 reprojection performs **no multi-looking at all** — see §6. |

---

## 2. Design decisions, stated up front

| Decision | Choice | One-line reason |
|---|---|---|
| Collection | `sentinel-1-rtc` on MPC only | Terrain-flattened gamma0; GRD is not comparable and Earth Search has no RTC. |
| On-disk units | **dB**, `int16`, scale **0.01** | See §4. |
| On-disk nodata | **-32768** | Matches the source; 0 dB is a real value so the optical 0-nodata convention is unusable. |
| Compositing domain | **linear power**, converted to dB only at write time | See §4.2. |
| Orbit | **one track per run**, auto-selected, recorded everywhere | See §5. |
| Speckle filter | **none** — use `resampling="average"` and, if needed, `res_m=20` | See §6. |
| Default `step` | **12 days**, not 8 | §0.1(a): at 8 d, 38 % of windows are empty. |
| Static path | **not supported**; already fails loudly | §3.9. |

---

## 3. Where each branch goes

Guiding rule (from the brief): smallest change, no radar special-cases inside the optical
path. Concretely that means the optical code paths must remain **byte-identical** except
for two-line early-exit guards, and every radar behaviour hangs off a single
`sensor = p.get("sensor", "s2")` discriminator carried on the provider dict.

### 3.1 Sensor-aware registries — **new siblings**, near L231-269 (**≈ +45 lines**)

Add alongside the existing `_PROVIDERS`, not inside it:

```python
_S1_PROVIDERS = {
    "mpc": {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-1-rtc",
        "sensor": "s1",
        "scl_asset": None,        # radar has no scene classification
        "cloud_field": None,      # radar has no cloud metadata
        "sign": True,
        "harmonize_offset": 0,
        "src_nodata": -32768.0,   # declared by the RTC assets
        "orbit_state_field": "sat:orbit_state",
        "relative_orbit_field": "sat:relative_orbit",
    },
}

SENTINEL1_BANDS = {
    "vv": {"mpc": "vv", "res_m": 10, "aliases": ["VV"]},
    "vh": {"mpc": "vh", "res_m": 10, "aliases": ["VH"]},
}
DEFAULT_S1_BANDS = ("vv", "vh")
```

and add `"sensor": "s2"` to each of the two existing `_PROVIDERS` entries (2 lines).

Deliberately **not** adding `hh`/`hv`: the collection declares them but all 45 items over
this AOI are VV+VH, and an unused code path is an untested one. Add when an HH AOI appears.

Deliberately **not** merging the band registries: nothing collides today (`vv`/`vh` vs
`red`/`nir`), but a merged lookup would let `resolve_bands(["red","vv"])` succeed and
produce a band list that no single collection can serve.

### 3.2 `resolve_bands` (L283) — **guarded, via a new parameter** (**≈ +8 lines**)

`def resolve_bands(bands, sensor="s2")`, selecting `_BAND_LOOKUP` or a new
`_S1_BAND_LOOKUP` and naming the right registry in the error message. Every existing
call site keeps the default and is unchanged.

A parameter beats a sibling `resolve_s1_bands` here because the normalisation logic
(strip/lower/de-separate/de-dup/order-preserve) is 15 lines that must not drift, and the
only difference between the two is which dict is consulted.

### 3.3 `_open_catalog` (L433) / `_pick_provider` (L632) — **guarded** (**≈ +6 lines**)

`_open_catalog(provider, sensor="s2")` picks `_S1_PROVIDERS if sensor == "s1" else
_PROVIDERS` and raises a clear `KeyError` when a provider has no entry for that sensor
(so `providers=("mpc","earthsearch")` with `sensor="s1"` skips earthsearch rather than
silently serving GRD). `_pick_provider(providers, sensor="s2")` threads it through.

### 3.4 `_harmonize_offset` (L447) — **guarded, 2 lines, highest value per line**

```python
if p.get("sensor", "s2") != "s2":
    return 0
```

Placed at the top of the function, this single guard protects all three call sites
(L619, L1578, L1673) at once. Without it: 1000 subtracted from gamma0 (§1 row 4).

### 3.5 `_composite_window` (L586) — **guarded branch** (**≈ +32 lines**)

This is the one place a genuine branch is warranted rather than a sibling. The function is
46 lines, of which ~28 (geobox shape handling, the `prov` provenance dict, the empty-items
NaN path, the S3 read-through cache call, `valid_pct`) are sensor-independent and must
stay in lockstep. A `_composite_window_sar` sibling would duplicate all of it. Three
narrow branches instead:

**(a) the search (replaces L599-601).** Build `query` conditionally:

```python
sensor = p.get("sensor", "s2")
if sensor == "s1":
    query = {"sat:orbit_state": {"eq": orbit_state},
             "sat:relative_orbit": {"eq": int(relative_orbit)}}
else:
    query = {p["cloud_field"]: {"lt": cloud_lt}}
```

`orbit_state` / `relative_orbit` arrive as new keyword parameters (default `None`), and
`_composite_window` **raises** if `sensor == "s1"` and either is `None` — an unfiltered
radar search is never a valid state (§5).

**(b) provenance (L604-606).** Record `orbit_state` / `relative_orbit` per scene instead
of `cloud`. Then assert single-track:

```python
tracks = {(it.properties["sat:orbit_state"], it.properties["sat:relative_orbit"]) for it in items}
if len(tracks) > 1:
    raise RuntimeError(f"window {start}..{end} mixes tracks {tracks}")
```

Cheap, and it converts the worst silent failure in the whole feature into a crash.

**(c) the load and reduce (replaces L612-624).** No SCL asset, no keep-mask, no offset,
`resampling="average"`, explicit `nodata`:

```python
_s2_seed_and_rewrite_items(items, band_assets)          # cache still applies; no scl_asset
ds = odc_load(items, bands=band_assets, geobox=geobox,
              resampling="average", groupby="id", dtype="float32",
              nodata=np.nan, chunks={})
for b, asset in zip(band_order, band_assets):
    out[b] = ds[asset].median("time", skipna=True).values.astype("float32")
```

Output stays **linear power**, NaN = no data — same contract the optical branch returns,
so `_process_tile` needs no knowledge of which branch ran until the packing step.

Note the existing `np.clip(arr - off, 0, None)` at L620 is *not* reused. It is harmless
for linear power (power is non-negative) but would be catastrophic if the composite ever
returned dB, and it is one more thing to reason about. Keep the branches separate.

The S3 cache (`_s2_seed_and_rewrite_items`, L558) works unchanged — it is keyed on
`item.id` + asset name and knows nothing about optics. Only the `_S2_CACHE_PREFIX = "s2"`
constant (L495) needs a per-sensor value (`"s1"`) so RTC objects do not land under the
S2 prefix and confuse the bucket's lifecycle rules. **+1 line** (make it a dict lookup on
`p["sensor"]`).

### 3.6 `_process_tile` (L753) — **guarded**, via one new packing helper (**≈ +45 lines**)

New sibling helper, ~20 lines, placed just above `_process_tile`:

```python
def _pack_stack(stack, sensor):
    """(array, dtype, nodata, scale, units) ready for GTiff. Optical is unchanged."""
    if sensor == "s1":
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(np.where(stack > 0, stack, np.nan))
        out = np.where(np.isfinite(db), np.clip(db, -50.0, 30.0) * 100.0, -32768.0)
        return out.astype("int16"), "int16", -32768, 0.01, "dB"
    out = np.clip(np.nan_to_num(stack, nan=0.0), 0, 65535).astype("uint16")
    return out, "uint16", None, 1.0, "DN"
```

Then in `_process_tile`:

- **L760** filename: prefix by sensor and track —
  `f"s1_{orbit_state[:3]}{relative_orbit:03d}_{res_m}m_tile_{tid:04d}.tif"` for radar,
  unchanged `sentinel_…` for optical. Fixes breakage #10.
- **L785** replaced by `out, dtype, nodata, scale, units = _pack_stack(stack, sensor)`.
- **L787-790** profile: `"dtype": dtype` and add `**({"nodata": nodata} if nodata is not None else {})`.
  Optical keeps its current no-nodata profile byte-for-byte.
- After the write loop (L795): for radar set `dst.scales = [scale] * count` and
  `dst.update_tags(sensor="sentinel-1", product="rtc-gamma0", units="dB", scale=str(scale),
  nodata="-32768", orbit_state=…, relative_orbit=…, polarizations=",".join(band_order))`.
- Manifest (L799-806) gains: `sensor`, `collection`, `product`, `units`, `scale`, `dtype`,
  `nodata`, `orbit_state`, `relative_orbit`, `orbit_selection` (`"auto"`/`"manual"`),
  `speckle_filter: null`, `resampling`, and `cloud_lt: null` (the key stays present so
  manifest consumers do not need a schema branch).
- `build_overviews(..., Resampling.average)` at L796 stays. Averaging dB is a geometric
  mean in power — slightly dark, but overviews are for display only.

`_process_tile` gains three new keyword parameters (`sensor`, `orbit_state`,
`relative_orbit`) threaded from `fetch_sentinel_imagery` through `_with_tile_retries`
(which already forwards `**kw` verbatim — **no change needed at L820**).

### 3.7 `_build_vrt_manual` (L873) — **1 line**

```python
gdal_dt = {"uint16": "UInt16", "int16": "Int16", "float32": "Float32", "uint8": "Byte"}.get(dtype, "UInt16")
```

And `_build_vrt` / `_clip_mosaic_to_geom` call sites in `fetch_sentinel_imagery`
(L1039, L1048) take the sensor-appropriate `vrt_name`, `pattern`, and `nodata=-32768`
(**≈ +4 lines**). Fixes breakages #8 and #9.

### 3.8 `fetch_sentinel_imagery` (L931) — **guarded** (**≈ +22 lines**)

New parameters: `sensor="s2"`, `orbit_state=None`, `relative_orbit=None`.
Behaviour when `sensor == "s1"`:

- `bands` defaults to `DEFAULT_S1_BANDS`; `resolve_bands(bands, sensor="s1")`.
- `providers` defaults to `("mpc",)`; passing `"earthsearch"` raises rather than
  falling back to GRD.
- `step` defaults to **12** (not 8). Justified by §0.1(a): at 12 d only 2 of 25 ascending
  windows are empty vs 14 of 37 at 8 d, and no window yet holds ≥2 scenes so the median
  is still an exact single-scene passthrough (see §4.2 on why ≥2 would matter). Do **not**
  go to 16 d: 5 asc windows then hold 2 scenes, which triggers the even-count-median
  average. If 16 d is ever wanted, the reducer must move to an explicit
  `.mean()`-in-linear or an odd-count-safe median.
- `cloud_lt` is ignored and recorded as `null`. Do not silently accept it — log once at
  INFO that it is inapplicable.
- If `orbit_state`/`relative_orbit` are `None`, call `select_s1_orbit` (§3.10) once for
  the whole AOI and use the result for every tile — exactly the pattern
  `fetch_sentinel_static_imagery` already uses for dates (L2070), so the tiles can never
  disagree about which track they show.

### 3.9 The static path — **no change, deliberately** (**0 lines**)

`fetch_sentinel_static_*`, `select_static_dates` (L1318), `_scan_date_masks` (L1263),
`_layer_match_coeffs` (L1530) and `_static_layers` (L1628) are **already fail-loud** for
radar: they all route band resolution through `resolve_static_bands` → `resolve_bands`,
which raises `ValueError: Unknown band 'vv'` (verified, §1 row 6). Because they take no
`sensor` parameter there is no way to reach the unguarded `p["scl_asset"]` reads at
L1291/1562/1643 with a radar provider dict.

Adding guards there would be dead code. **Do not add a `sensor=` parameter to any static
function.** The static concept itself is optical-specific: it picks the least-cloudy real
acquisition, and radar has no cloud — every S1 acquisition is equally "clear", so date
selection collapses to "the nearest date on the chosen track", a different and much
simpler problem. If a radar static image is ever wanted, it is a separate ~40-line
function, not a branch through this machinery.

The only thing to add is a sentence in each static docstring: *"Optical only — see
`fetch_sentinel_imagery(sensor='s1')` for radar."*

### 3.10 `select_s1_orbit` — **new sibling function** (**≈ +40 lines**)

Placed near `_date_windows`. Standalone and side-effect-free, so it is testable before
anything else exists:

```python
def select_s1_orbit(aoi, start, end, providers=("mpc",), prefer=None):
    """Group S1 items over the AOI by (sat:orbit_state, sat:relative_orbit) and pick one.

    Returns {"orbit_state", "relative_orbit", "n_items", "dates", "median_gap_days",
             "candidates": [ ...all tracks, same fields... ]}
    """
```

Selection rule, in order: (1) drop tracks whose footprints do not cover ≥99 % of the AOI;
(2) most unique acquisition dates; (3) tie-break on the smallest maximum inter-acquisition
gap (a track with one 24-day hole is worse than an evenly spaced one). `prefer=`
(`"ascending"`/`"descending"`) short-circuits (1)-(3) within that pass.

For this AOI it returns `ascending` / `71` (23 dates vs 22).

---

## 4. The radiometric contract

### 4.1 What is stored

| Property | Value |
|---|---|
| Quantity | terrain-flattened **gamma0**, from `sentinel-1-rtc` |
| Units on disk | **decibels** |
| dtype | **`int16`** |
| Scale | **0.01** — stored value × 0.01 = dB. `-823` → `-8.23 dB` |
| Valid range | `-5000 … 3000` (-50.00 … +30.00 dB), clipped at write |
| nodata | **`-32768`** |
| Declared where | (a) GeoTIFF `nodata` tag; (b) GDAL band `scales = 0.01`; (c) dataset tags `sensor/product/units/scale/orbit_state/relative_orbit/polarizations`; (d) the per-tile `.manifest.json` (authoritative) |
| Band descriptions | `vv_2026_08_30` (unchanged convention, §5.3) |

**Why dB and not linear power.** The downstream Whittaker smoother
(`geo_inference_workers.process_smoothing_chunk`) is a linear, second-difference-penalised
operator, and the RF consumes the smoothed values directly. Speckle is *multiplicative* in
power and *additive* in dB, so a linear smoother is only well-posed in dB — in power its
residuals are heteroscedastic (bright targets dominate the penalty). Crop growth curves
are also closer to linear in dB. Storing linear and converting at read time would push a
`log10` into every consumer and guarantee that one of them forgets.

**Why `int16` × 0.01 and not `float32`.** 0.01 dB is two orders of magnitude below S1's
~0.5 dB radiometric accuracy, so the quantisation is lossless in any sense that matters.
`int16` is 2 bytes against float32's 4, matching the optical tiles' size class, and dB is
a smooth field so `deflate` + `predictor=2` (already in the profile at L789) compresses it
well — float32 linear power compresses poorly because the mantissa is noise. The optical
path's `uint16` is not reusable because **half the values are negative**.

**Why nodata `-32768` and not `0`.** 0.00 dB is a real, common value (bright agricultural
targets and urban double-bounce; the measured AOI max is +14.4 dB). Reusing the optical
0-means-nodata convention would silently delete every 0 dB pixel and, worse, fill the
outside-AOI ring with a plausible bright value (breakage #9). `-32768` is also exactly
what the RTC source assets declare, so the convention is inherited rather than invented.

**How a reader knows.** In priority order: the manifest `"units": "dB", "scale": 0.01,
"nodata": -32768` (the same file the optical path already writes, L799); then the GeoTIFF
dataset tags for anyone reading the raster alone; then GDAL's band `scales`, so
`src.read(1, masked=True) * src.scales[0]` yields dB in any GDAL-aware tool without
special-casing. All three must be written — tags get stripped by some conversions, scales
get ignored by some readers, manifests get separated from rasters. Redundancy is cheap.

### 4.2 Compositing happens in linear, conversion happens at write

`_composite_window` returns **linear power**; `_pack_stack` converts to dB immediately
before the raster write. Nothing else in the pipeline sees linear values.

A correction to the premise in the brief: **the median *does* commute with dB.** `10·log10`
is strictly monotonic, so `median(dB(x)) == dB(median(x))` exactly. The ordering
nonetheless matters, for three reasons that are about *means*, not medians:

1. **Even-count medians are means.** `xarray.median` on an even number of samples averages
   the two central values. At `step=12` no window holds ≥2 scenes so this never fires today,
   but at `step=16` five ascending windows do (§0.1a). `(a+b)/2` in power ≠ `(dB(a)+dB(b))/2`.
2. **Spatial resampling is a mean.** `resampling="average"` (§6) averages source pixels.
   Averaging in dB is a geometric mean of power, which is biased low — for a 4-look average
   of Rayleigh-distributed power the bias is roughly -1.0 dB, and it varies with local
   texture, so it shows up as a spurious contrast change at field edges. Averaging in
   linear is the unbiased estimator.
3. **NaN arithmetic.** `log10(0)` is `-inf` and `log10(negative)` is NaN. Doing the
   conversion once, in one guarded place with an explicit `np.where(stack > 0, …)`, means
   there is exactly one line to get right instead of one per consumer.

So: **all averaging in linear power, exactly one `10·log10` in `_pack_stack`.**

---

## 5. Orbit handling

### 5.1 Why they cannot be mixed

Ascending (local ~18:02) and descending (local ~05:57) view the same field from opposite
sides at ~39° incidence. Row-direction anisotropy in cereals, layover/shadow geometry, and
a ~12-hour difference in dew and canopy moisture give systematic VV offsets of order 1-3 dB
between the two passes. If both are loaded into one window the median picks one or averages
two, and which one it picks flips with the acquisition calendar — injecting a
pseudo-periodic sawtooth into the time series. The Whittaker smoother then does the worst
possible thing: it is a low-pass, so it smooths that sawtooth into a smooth spurious trend
that looks exactly like phenology.

At `step=8`, mixing tracks would make **12 of 37** windows contain 2 scenes — one ascending
and one descending — so this would not be a rare edge case, it would be a third of the
series.

### 5.2 How the filter is applied

Server-side, in the STAC query, both fields together (verified: 23 items, §0.1b):

```python
query = {"sat:orbit_state":    {"eq": orbit_state},        # "ascending" | "descending"
         "sat:relative_orbit": {"eq": int(relative_orbit)}}
```

Both are required. `sat:orbit_state` alone is not sufficient in general — a larger AOI can
be covered by two ascending tracks with different incidence angles, which are as
incompatible with each other as asc is with desc. (This AOI happens to have one track per
pass, but the code must not depend on that.)

Because an unknown property name returns 0 items with no error (§0.1b), the branch must be
covered by a test asserting a non-zero count, and `_composite_window` must raise when
`orbit_state`/`relative_orbit` are `None` for radar. There is no valid "no orbit filter"
state.

Belt and braces, since a server-side filter that silently no-ops is exactly the failure
mode we are guarding against: after the search, assert client-side that all returned items
share one `(orbit_state, relative_orbit)` and raise otherwise (§3.5b). Two lines.

### 5.3 How it is recorded

- **Filename**: `s1_asc071_10m_tile_0001.tif`. This is load-bearing, not cosmetic — it is
  what stops `_build_vrt`'s glob from stitching two tracks into one mosaic (breakage #10),
  and it lets both tracks live in one `out_dir`.
- **VRT / clipped names**: `s1_asc071.vrt`, `s1_asc071_clipped.tif`, with the matching glob
  pattern passed to `_build_vrt`.
- **Manifest**: top-level `"orbit_state"`, `"relative_orbit"`, `"orbit_selection"`
  (`"auto"` when `select_s1_orbit` chose it, `"manual"` when the caller passed it), and
  `"orbit_candidates"` — the full `select_s1_orbit` candidate list, so the run records
  what it *didn't* pick and why the alternative had fewer dates.
- **Per-window provenance**: each `scenes[]` entry carries its own `orbit_state` and
  `relative_orbit`, replacing the optical `cloud` field. Makes a contaminated window
  detectable from the manifest alone, after the fact.
- **GeoTIFF dataset tags**: `orbit_state`, `relative_orbit` — so a stray `.tif` separated
  from its manifest is still self-describing.

---

## 6. Speckle — recommendation

**Do not implement a Lee filter, or any other spatial speckle filter, in this pipeline.**

The reasoning, and one correction to the premise.

**The premise does not hold.** The brief assumes an "8-day multi-scene median" already
provides multi-looking. It does not: measured over this AOI and range, **every non-empty
8-day window contains exactly one scene per track** (§0.1a). The median is an identity
operation. Radar arrives with ENL = 4.4, i.e. a per-pixel coefficient of variation of
**47.7 %** in linear power (≈ ±2.1 dB, 1σ), and the compositing step reduces that by
exactly nothing.

**What does reduce it, in order of value per line of code:**

1. **`resampling="average"` instead of `"nearest"` in the radar `odc_load` (1 line).**
   This is required for correctness independent of speckle — nearest-neighbour resampling
   of a continuous physical quantity through a UTM→WGS84 reprojection is simply wrong, and
   it is only in the current code because SCL is categorical. At 10 m → 10 m the kernel
   overlap alone gives roughly 1.5-2 effective looks.
2. **`res_m=20` with `average` (0 lines — it is already a parameter).** Exact 4× spatial
   multi-looking: ENL 4.4 → ~17.6, CV **47.7 % → 23.8 %**. For field-scale crop work at
   ~28.6°N in Punjab, 20 m is well inside typical parcel size. This is the recommendation
   if the experiment reports that speckle is limiting.
3. **The existing Whittaker smoother.** With `step=12` it runs over ~25 time steps and is
   a strong temporal low-pass. Applied to dB — where speckle is additive and roughly
   Gaussian — it is close to the right estimator for the temporal axis, and it is already
   built and tested.

**Why not Lee.** (a) Refined-Lee is adaptive and its output is no longer an unbiased
gamma0 estimate; it would break the radiometric contract defined in §4 the moment it is
enabled, and the manifest would have to carry a filter-parameter provenance block to stay
honest. (b) Spatial averaging by resampling achieves the same variance reduction with an
estimator that stays unbiased and needs no parameters. (c) Filtering belongs in the feature
stage, where it can be toggled and A/B'd against the unfiltered stack, not baked into the
fetch layer where every consumer inherits it silently. (d) It is a non-trivial algorithm to
implement and validate, on the critical path of a feature whose value is still being tested.

**Acceptance test for whichever route is chosen:** pick a homogeneous bare field inside the
AOI, compute the per-pixel CV of linear VV over a ~50×50 px patch on a single date. Expect
≈0.48 at `res_m=10, nearest`, ≈0.35-0.40 at `res_m=10, average`, ≈0.24 at
`res_m=20, average`. Record the number in the manifest as `"measured_cv"` on the first
run so the choice is auditable.

Record `"speckle_filter": null` in the manifest regardless, so the field exists from day one
and a future filtered product is distinguishable from an old unfiltered one.

---

## 7. Line estimate and build order

| Step | Area | File / lines | Est. | Independently testable by |
|---|---|---|---|---|
| 1 | `select_s1_orbit` — new sibling | `sentinel.py`, near L385 | **+40** | Call it on the AOI; assert it returns `ascending`/`71` with 23 dates and lists `descending`/`5` with 22. Touches no existing code. |
| 2 | `_S1_PROVIDERS`, `SENTINEL1_BANDS`, `_S1_BAND_LOOKUP`, `sensor` key on `_PROVIDERS` | L231-269 | **+45** | Pure data. `_PROVIDERS["mpc"]["sensor"] == "s2"`. |
| 3 | `resolve_bands(sensor=)`, `_open_catalog(sensor=)`, `_pick_provider(sensor=)` | L283, 433, 632 | **+14** | `resolve_bands(["VV","vh"], sensor="s1") == ["vv","vh"]`; `resolve_bands(["red"])` byte-identical to before; `_open_catalog("earthsearch","s1")` raises. |
| 4 | `_harmonize_offset` sensor guard | L447 | **+2** | `_harmonize_offset(_S1_PROVIDERS["mpc"], items) == 0`; the two optical cases still return 1000 / 0. |
| 5 | `_build_vrt_manual` dtype map | L873 | **+1** | Write a 1-band `int16` tile with a negative value, build the manual VRT, read it back, assert the value survives as negative. |
| 6 | `_composite_window` radar branch | L586 | **+32** | Direct call with a one-tile geobox and one 12-day window: assert ≥1 item, linear VV median ≈0.16 (dB ≈ -8), no NaN inside the swath, `prov["scenes"][0]["relative_orbit"] == 71`. Assert the mixed-track guard raises when handed two tracks' items. |
| 7 | `_pack_stack` + `_process_tile` write path | new helper before L753; L760, 785, 787-790, 795-806 | **+45** | Write one tile; reopen: `dtype == "int16"`, `nodata == -32768`, `descriptions[0] == "vv_2025_12_07"`, `read(1)*0.01` median ≈ -8 dB, tags carry `units=dB` and the orbit. |
| 8 | `_S2_CACHE_PREFIX` per sensor | L495 | **+1** | Cache key for an S1 item starts `s1/`. |
| 9 | `fetch_sentinel_imagery(sensor=…)` + VRT/clip names and `nodata` | L931, 1039, 1048 | **+26** | End-to-end on the AOI, 1-2 tiles, `step=12`: VRT builds, `s1_asc071_clipped.tif` has `nodata=-32768`, manifest carries the full radar block. |
| 10 | `parse_stac_bands` made band-agnostic | `geo_inference_workers.py` L11-23 | **+22** | Existing S2 descriptions produce byte-identical output to today; radar descriptions yield `{"vv":[…],"vh":[…]}` + 25 dates; an all-unmatched description list now **raises** instead of returning `([],[],[])`. |
| 11 | CLI `--sensor` / `--orbit-state` / `--relative-orbit` | L2110-2196 | **+12** | `--mode tiled --sensor s1` runs. |
| 12 | Docstring notes on the static functions | L1714, 1936, 1318 | **+3** | n/a |
| | **Total** | | **≈ 243** (≈ 221 in `sentinel.py`) | |

Steps 1-5, 8 and 10 change no optical behaviour at all and can land, be reviewed and be
merged before the radar experiment reports. Steps 6-7 are where the real work is. Step 9
is the first point at which an end-to-end radar run exists.

A useful mid-point: after step 7 you can build a single tile by calling `_process_tile`
directly and inspect it in QGIS, without `fetch_sentinel_imagery` knowing about radar yet.

---

## 8. Band naming and `parse_stac_bands`

### 8.1 Naming needs no change

`_process_tile` builds descriptions at **L783** as `f"{b}_{edate}"` where
`edate = e.replace("-", "_")`. With `band_order = ["vv","vh"]` that already yields
`vv_2026_08_30`, `vh_2026_08_30`. **No change to the naming code.**

The one constraint this imposes: **a radar band name must contain no underscore**, because
the date parser splits on `_` and takes the last three fields. `vv` and `vh` are fine. A
future ratio band must be named `vvvh` or `rvi`, never `vv_vh_ratio`.

### 8.2 What `parse_stac_bands` needs

Current implementation (`geo_inference_workers.py` L11-23) hardcodes `red_`/`nir_` and,
worse, its guard `assert len(red_idx) == len(nir_idx)` passes on `0 == 0`, so radar input
produces an empty band index and an empty date list with no error (§1 row 7).

Replace with a band-agnostic parser plus a thin compatibility wrapper, so no existing
caller changes:

```python
_BAND_DATE_RE = re.compile(r"^(?P<band>[a-z0-9]+)_(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})$")

def parse_band_index(descriptions):
    """{band_name: [band indices]}, plus the ordered unique dates. Sensor-agnostic."""
    by_band, dates = {}, []
    for i, d in enumerate(descriptions):
        m = _BAND_DATE_RE.match((d or "").strip().lower())
        if not m:
            continue
        by_band.setdefault(m["band"], []).append(i)
        date = f"{m['y']}-{m['m']}-{m['d']}"
        if date not in dates:
            dates.append(date)
    if not by_band:
        raise ValueError("no '<band>_YYYY_MM_DD' band descriptions found")
    n = {len(v) for v in by_band.values()}
    if len(n) != 1:
        raise ValueError(f"bands have unequal time-step counts: "
                         f"{ {k: len(v) for k, v in by_band.items()} }")
    return by_band, dates

def parse_stac_bands(descriptions):
    """Back-compat shim: red/nir indices + dates. Raises if the stack is not optical."""
    by_band, dates = parse_band_index(descriptions)
    try:
        return by_band["red"], by_band["nir"], dates
    except KeyError as e:
        raise ValueError(f"expected red and nir bands, found {sorted(by_band)}") from e
```

Two behavioural changes to be explicit about, both turning silence into a crash:
`parse_stac_bands` now **raises** on a radar stack instead of returning `([], [], [])`, and
it raises when band counts disagree instead of asserting only red-vs-nir. The optical
happy path returns exactly what it returns today.

Callers: `geo_inference_workers.py` L100 and `build_orchard_training_set.py` L377 both use
the shim and need no edit. A radar consumer calls `parse_band_index` and picks
`by_band["vv"]` / `by_band["vh"]`.

The smoother itself (`process_smoothing_chunk`) is already band-agnostic — but note its
`nodata_val` parameter and its `clip_bounds`. For radar, callers must pass
`nodata_val=-32768` (not `0`, not `None`) and `clip_bounds=(-5000, 3000)` in stored units,
or `(-50, 30)` if the values are scaled to dB first. Passing the optical `clip_bounds` of
`(0, 10000)` would clip every negative dB value to 0 — the same silent destruction as
breakage #5, one layer further downstream.

---

## 9. Failure-mode summary — what breaks, and how quietly

Ordered by how hard it is to notice. Everything above the line is silent.

| # | If this is missed | Symptom | Why it is silent |
|---|---|---|---|
| 1 | `_harmonize_offset` guard (§3.4) | **Every pixel 0.** | Returns 1000 via the date fallback; `clip(0.165-1000, 0, None)` = 0. Looks like "no data over the AOI", which is a plausible thing to blame on the search. |
| 2 | `uint16` packing at L785 (§3.6) | 99 % of pixels 0, a handful of 1-4. | The tile writes, opens, has the right shape and band names. Only a histogram reveals it. |
| 3 | `eo:cloud_cover` filter left in (§3.5a) | Every window empty → all-NaN → all-zero bands. | Verified: 45 items → 0, no exception. Indistinguishable from a genuinely empty AOI. |
| 4 | Orbit filter omitted or mistyped (§5.2) | A 1-3 dB sawtooth that the Whittaker smooths into fake phenology. | Both failure directions are silent: no filter → mixed tracks (12 of 37 windows at step 8); typo'd property → 0 items. This is the one that produces *plausible but wrong* science rather than obvious garbage. |
| 5 | `int16` missing from the VRT dtype map (L873, §3.7) | -8.00 dB reads back as 64736. | Only in the `gdalbuildvrt`-absent fallback, so it will not reproduce on a dev box with GDAL CLI. Worst reproducibility profile of anything here. |
| 6 | nodata left at the optical `0` (§4.1, breakage #9) | Real 0 dB pixels deleted; outside-AOI ring filled with a plausible bright value. | 0 dB is a normal value, so nothing looks anomalous at the boundary. |
| 7 | Tile filename prefix not changed (§3.6, breakage #10) | Two tracks, or S2 and S1, stitched into one VRT. | `gdalbuildvrt` warns on stderr and continues; in a threaded run that scrolls past. |
| 8 | Compositing in dB rather than linear (§4.2) | ~-1 dB texture-dependent bias at field edges from `average` resampling; a wrong value in any even-count window. | Small, spatially structured, and looks like real edge behaviour. |
| 9 | `resampling="nearest"` kept (§6, breakage #11) | Full 47.7 % speckle, no multi-look, aliasing from the UTM→WGS84 warp. | Looks like "radar is just noisy", which is exactly the conclusion the experiment is trying to test. |
| 10 | `clip_bounds` left optical in the smoother (§8.2) | Every negative dB clipped to 0. | Happens downstream of the fetcher, so the tiles on disk are correct and the bug appears to be in the model. |
| 11 | `parse_stac_bands` left as-is (§8.2) | `n_timesteps = 0`, 0×0 penalty matrix, empty band axis. | `assert 0 == 0` passes. |
| 12 | Earth Search left in `providers` (§0) | GRD DN silently substituted for RTC gamma0 on provider failover. | Only happens when MPC is down, i.e. rarely and non-reproducibly. |
| — | *(below the line: loud)* | | |
| 13 | SCL asset not removed from the load list (§3.5c) | odc-stac raises: no `SCL` asset on the item. | Immediate exception. |
| 14 | `resolve_bands` not sensor-aware (§3.2) | `ValueError: Unknown band 'vv'`. | Immediate exception — and the reason the whole static path is already safe (§3.9). |
