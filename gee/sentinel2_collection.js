// ============================================================
// GreenGrid AI — Phase 2: Dataset Collection
// Google Earth Engine Script: Sentinel-2 Collection Explorer
// ============================================================
//
// PURPOSE:
//   This script identifies and previews Sentinel-2 Level-2A imagery
//   over Delhi NCT for:
//     - July 2022 (historical baseline)
//     - July 2026 (current condition)
//
// HOW TO USE:
//   1. Open Google Earth Engine Code Editor: https://code.earthengine.google.com
//   2. Paste this entire script into the editor
//   3. Click Run
//   4. Review the Console output (image metadata, tile IDs, dates, cloud %)
//   5. Inspect the Map layers
//   6. Record actual tile IDs and dates in dataset_metadata.csv
//
// DATASET:
//   Copernicus Sentinel-2 Level-2A (Surface Reflectance)
//   Earth Engine ID: COPERNICUS/S2_SR_HARMONIZED
//   (Harmonized = adjusts for the Jan 2022 processing baseline change)
//
// BANDS (relevant for Phase 2):
//   B2   - Blue        (490 nm) — 10 m
//   B3   - Green       (560 nm) — 10 m
//   B4   - Red         (665 nm) — 10 m
//   B5   - Red Edge 1  (705 nm) — 20 m
//   B6   - Red Edge 2  (740 nm) — 20 m
//   B7   - Red Edge 3  (783 nm) — 20 m
//   B8   - NIR         (842 nm) — 10 m
//   B8A  - Narrow NIR  (865 nm) — 20 m
//   B11  - SWIR 1      (1610 nm) — 20 m
//   B12  - SWIR 2      (2190 nm) — 20 m
//   QA60 - Cloud mask band
//   SCL  - Scene Classification Layer (cloud/shadow/snow/etc)
//
// NOTE: Sentinel-2 tiles are 100×100 km (MGRS grid).
//       Delhi NCT requires tile 44RNR (primary) and possibly 43RGQ / 44RNQ.
//       The exact tiles will be confirmed by the Console output.
// ============================================================

// ─── 1. DELHI NCT STUDY AREA ─────────────────────────────────────────────
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");
var delhiNCT = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
));

var delhiGeom = delhiNCT.geometry();
print("Delhi NCT area (km²):", delhiGeom.area(100).divide(1e6));

Map.centerObject(delhiGeom, 10);
Map.addLayer(
  delhiNCT,
  {color: "FF0000", fillColor: "FF000022", width: 2},
  "Delhi NCT Boundary"
);

// ─── 2. SENTINEL-2 COLLECTION DEFINITION ─────────────────────────────────
// Use HARMONIZED collection to ensure consistent processing across 2022–2026
var S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED";

// Maximum cloud cover filter to apply before mosaic/composite
var MAX_CLOUD_COVER = 30; // %

// Cloud mask using SCL (Scene Classification Layer — most robust for S2)
function maskS2Clouds_SCL(image) {
  var scl = image.select("SCL");
  // SCL classes to KEEP (clear land/water/vegetation):
  //   4 = Vegetation, 5 = Non-vegetated land, 6 = Water,
  //   7 = Unclassified, 8 = Cloud medium prob (optional), 11 = Snow/Ice
  // Classes to REMOVE: 3=cloud shadow, 8=cloud med prob, 9=cloud high prob, 10=thin cirrus
  var clearMask = scl.neq(3)   // cloud shadow
    .and(scl.neq(8))           // cloud medium probability
    .and(scl.neq(9))           // cloud high probability
    .and(scl.neq(10));         // thin cirrus
  return image.updateMask(clearMask);
}

// Alternative cloud mask using QA60 (simpler, backup)
function maskS2Clouds_QA60(image) {
  var qa60 = image.select("QA60");
  var cloudBit   = 1 << 10; // bit 10 = opaque cloud
  var cirrusBit  = 1 << 11; // bit 11 = cirrus
  var cloudMask  = qa60.bitwiseAnd(cloudBit).eq(0)
    .and(qa60.bitwiseAnd(cirrusBit).eq(0));
  return image.updateMask(cloudMask);
}

// Sentinel-2 reflectance scale: divide by 10000 (values are in 0–10000)
function scaleS2Reflectance(image) {
  var bands = image.select(["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]);
  return bands.divide(10000)
    .addBands(image.select(["QA60","SCL"]))
    .copyProperties(image, image.propertyNames());
}

// ─── 3. JULY 2022 — HISTORICAL BASELINE ──────────────────────────────────
print("=== SENTINEL-2 — JULY 2022 (Historical Baseline) ===");

var s2_2022 = ee.ImageCollection(S2_COLLECTION)
  .filterBounds(delhiGeom)
  .filterDate("2022-07-01", "2022-07-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
  .sort("CLOUDY_PIXEL_PERCENTAGE");

print("Number of S2 scenes over Delhi NCT, July 2022 (cloud < 30%):", s2_2022.size());

// Print metadata
var s2_2022_info = s2_2022.map(function(img) {
  return img.set({
    "date":         ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
    "cloud_pct":    img.get("CLOUDY_PIXEL_PERCENTAGE"),
    "mgrs_tile":    img.get("MGRS_TILE"),
    "product_id":   img.get("PRODUCT_ID"),
    "spacecraft_id":img.get("SPACECRAFT_NAME")
  });
});

print("July 2022 — Acquisition dates:");
print(s2_2022_info.aggregate_array("date"));
print("Cloud % :");
print(s2_2022_info.aggregate_array("cloud_pct"));
print("MGRS Tiles:");
print(s2_2022_info.aggregate_array("mgrs_tile"));
print("Product IDs:");
print(s2_2022_info.aggregate_array("product_id"));

// Best single scene
var s2_2022_best = ee.Image(s2_2022.first());
print("Best S2 scene — July 2022:");
print("  Date:", s2_2022_best.date().format("YYYY-MM-dd"));
print("  Cloud %:", s2_2022_best.get("CLOUDY_PIXEL_PERCENTAGE"));
print("  MGRS Tile:", s2_2022_best.get("MGRS_TILE"));
print("  Product ID:", s2_2022_best.get("PRODUCT_ID"));

// Cloud-masked median composite for full Delhi coverage
var s2_2022_composite = s2_2022
  .map(maskS2Clouds_SCL)
  .map(scaleS2Reflectance)
  .median()
  .clip(delhiGeom);

// True Colour RGB (B4=Red, B3=Green, B2=Blue)
Map.addLayer(
  s2_2022_composite,
  {bands: ["B4", "B3", "B2"], min: 0.0, max: 0.3, gamma: 1.4},
  "S2 July 2022 — True Colour Composite"
);

// False Colour composite (NIR, Red, Green) — highlights vegetation in red
Map.addLayer(
  s2_2022_composite,
  {bands: ["B8", "B4", "B3"], min: 0.0, max: 0.5, gamma: 1.4},
  "S2 July 2022 — False Colour (NIR/Red/Green)"
);

// ─── 4. JULY 2026 — CURRENT CONDITION ────────────────────────────────────
print("=== SENTINEL-2 — JULY 2026 (Current Condition) ===");

var s2_2026 = ee.ImageCollection(S2_COLLECTION)
  .filterBounds(delhiGeom)
  .filterDate("2026-07-01", "2026-07-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
  .sort("CLOUDY_PIXEL_PERCENTAGE");

print("Number of S2 scenes over Delhi NCT, July 2026 (cloud < 30%):", s2_2026.size());

var s2_2026_info = s2_2026.map(function(img) {
  return img.set({
    "date":         ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
    "cloud_pct":    img.get("CLOUDY_PIXEL_PERCENTAGE"),
    "mgrs_tile":    img.get("MGRS_TILE"),
    "product_id":   img.get("PRODUCT_ID"),
    "spacecraft_id":img.get("SPACECRAFT_NAME")
  });
});

print("July 2026 — Acquisition dates:");
print(s2_2026_info.aggregate_array("date"));
print("Cloud %:");
print(s2_2026_info.aggregate_array("cloud_pct"));
print("MGRS Tiles:");
print(s2_2026_info.aggregate_array("mgrs_tile"));
print("Product IDs:");
print(s2_2026_info.aggregate_array("product_id"));

var s2_2026_best = ee.Image(s2_2026.first());
print("Best S2 scene — July 2026:");
print("  Date:", s2_2026_best.date().format("YYYY-MM-dd"));
print("  Cloud %:", s2_2026_best.get("CLOUDY_PIXEL_PERCENTAGE"));
print("  MGRS Tile:", s2_2026_best.get("MGRS_TILE"));
print("  Product ID:", s2_2026_best.get("PRODUCT_ID"));

var s2_2026_composite = s2_2026
  .map(maskS2Clouds_SCL)
  .map(scaleS2Reflectance)
  .median()
  .clip(delhiGeom);

Map.addLayer(
  s2_2026_composite,
  {bands: ["B4", "B3", "B2"], min: 0.0, max: 0.3, gamma: 1.4},
  "S2 July 2026 — True Colour Composite"
);

Map.addLayer(
  s2_2026_composite,
  {bands: ["B8", "B4", "B3"], min: 0.0, max: 0.5, gamma: 1.4},
  "S2 July 2026 — False Colour (NIR/Red/Green)"
);

// ─── 5. BEFORE/AFTER COMPARISON ──────────────────────────────────────────
// Show NIR band difference to highlight vegetation change (raw data preview only)
var nir_diff = s2_2026_composite.select("B8")
  .subtract(s2_2022_composite.select("B8"))
  .clip(delhiGeom);

Map.addLayer(
  nir_diff,
  {min: -0.1, max: 0.1, palette: ["red", "white", "green"]},
  "NIR Difference (2026 - 2022) — preview only"
);

// ─── 6. TILE COVERAGE INFORMATION ─────────────────────────────────────────
print("=== S2 MGRS TILE INFO ===");
print("Sentinel-2 MGRS tiles expected to cover Delhi NCT:");
print("  Primary tile  : 44RNR");
print("  Secondary tile: 43RGQ (partial western edge)");
print("  Tertiary tile : 44RNQ (partial southern edge)");
print("Check the Console output above to confirm actual tile IDs in your imagery.");

// ─── 7. SUMMARY ───────────────────────────────────────────────────────────
print("=== NEXT STEPS ===");
print("1. Record the actual acquisition dates and product IDs from Console output");
print("2. Update dataset_metadata.csv with actual values");
print("3. Run export_data.js to export Sentinel-2 bands to Google Drive");
print("4. Phase 3 will calculate NDVI and other vegetation indices from these bands");
