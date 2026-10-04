# Masking recipe — cloned from `gee/export_data_2023_2025.js` (verified V1 export recipe)

Source: `gee/export_data_2023_2025.js` (GreenGrid-AI, GEE). Quoted lines are verbatim.

## Landsat (V1 used `LANDSAT/LC09/C02/T1_L2`; PC equivalent `landsat-c2-l2`)

**Pixel mask — ONLY bits 3 (cloud) and 4 (cloud shadow) are masked.**
Deviation from the "typical" recipe: dilated cloud (bit 1), cirrus (bit 2),
snow (bit 5) and fill (bit 0) are NOT masked — followed the script.

```js
function maskL9Clouds(image) {
  var qa = image.select("QA_PIXEL");
  return image.updateMask(
    qa.bitwiseAnd(1 << 4).eq(0)
      .and(qa.bitwiseAnd(1 << 3).eq(0))
  );
}
```

**Scene-level filters: NONE on CLOUD_COVER; collection is T1 only.**
The collection id itself enforces T1 (`LANDSAT/LC09/C02/T1_L2`); there is no
`filter(ee.Filter.lt("CLOUD_COVER", ...))` anywhere in the script. On PC we
filter `landsat:collection_category == "T1"`.

```js
var l9_raw = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(delhiGeom)
    .filterDate(startDate, endDate)
    .sort("CLOUD_COVER");
```

**Scale factors** (applied in V1 before compositing):

```js
srBands = image.select("SR_B.*").multiply(0.0000275).add(-0.2);
```

**ST_B10 is NOT scaled in the V1 script** — the raw band is carried through
`addBands(thermalBand)` and exported as raw digital numbers (confirmed: the V1
composite `landsat9_2026_07_composite_30m.tif` band 7 ranges 41050–51844 DN
≈ 289–326 K after scaling). Our investigation applies the standard C2-L2
scale `ST_K = DN × 0.00341802 + 149.0` (documented MTL scale) to obtain
physical Kelvin for ST metrics; this is a deliberate deviation required to
report ST in °C.

**Valid_ST definition used here** (adds sanity gates the script lacks, since
V1's median composite silently ignores masked pixels):
`Valid_ST = QA bits 3,4 clear AND raw != 0 AND 250 ≤ ST_K ≤ 350`
(≈ −23…77 °C). `Valid_SR = QA bits 3,4 clear` (bit 0 fill ignored, as script does).

**Deviation — platforms:** V1 exported Landsat 9 only (`LC09`). This
investigation includes both LC08 and LC09 from `landsat-c2-l2` (per Phase 2
scope), still T1-only, with per-platform counts reported (`n_scenes_L8/L9`).

## Sentinel-2 (GEE `COPERNICUS/S2_SR_HARMONIZED`; PC equivalent `sentinel-2-l2a`)

**Scene-level filter: CLOUDY_PIXEL_PERCENTAGE < 60:**

```js
.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
```

On PC this maps to `eo:cloud_cover` (< 60), verified against item properties.

**Pixel mask — SCL classes 3, 8, 9, 10 only.**
Deviation from the "typical" recipe: classes 0 (no data) and 1 (saturated/
defective) are NOT masked in V1 — followed the script.

```js
function maskS2Clouds(image) {
  var scl = image.select("SCL");
  return image.updateMask(
    scl.neq(3)     // cloud shadow
      .and(scl.neq(8))   // medium probability cloud
      .and(scl.neq(9))   // high probability cloud
      .and(scl.neq(10))  // cirrus
  );
}
```

**Valid_S2 = SCL ∉ {3, 8, 9, 10}** (classes 0/1 kept, per script).
The V1 composite divides reflectance by 10000 (we don't need reflectance here;
only SCL-based validity).

## Compositing
V1 uses a **median** composite per window (`l9_raw.map(maskL9Clouds)...median()`).
This investigation does not rebuild composites; it accumulates per-pixel
valid-observation counts, ST (count/sum/sumsq/min/max), and coverage stats
per window-year — the per-pixel evidence a median composite would draw from.
