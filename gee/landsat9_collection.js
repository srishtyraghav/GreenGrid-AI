// ============================================================
// GreenGrid AI — Phase 2: Dataset Collection
// Google Earth Engine Script: Landsat 9 Collection Explorer
// ============================================================
//
// PURPOSE:
//   This script identifies and previews Landsat 9 Collection 2
//   Level-2 Science Product (LC09) imagery over Delhi NCT for:
//     - July 2022 (historical baseline)
//     - July 2026 (current condition)
//
// HOW TO USE:
//   1. Open Google Earth Engine Code Editor: https://code.earthengine.google.com
//   2. Paste this entire script into the editor
//   3. Click Run
//   4. Review the Console output (image metadata, dates, cloud cover)
//   5. Inspect the Map layers (Delhi boundary + image composites)
//   6. Record the actual image IDs and dates in dataset_metadata.csv
//
// DATASET:
//   USGS Landsat 9 Collection 2 Tier 1 Level-2 Science Products
//   Earth Engine ID: LANDSAT/LC09/C02/T1_L2
//   Bands included:
//     SR_B1  - Coastal Aerosol (0.435–0.451 µm) — 30 m
//     SR_B2  - Blue          (0.452–0.512 µm) — 30 m
//     SR_B3  - Green         (0.533–0.590 µm) — 30 m
//     SR_B4  - Red           (0.636–0.673 µm) — 30 m
//     SR_B5  - NIR           (0.851–0.879 µm) — 30 m
//     SR_B6  - SWIR 1        (1.566–1.651 µm) — 30 m
//     SR_B7  - SWIR 2        (2.107–2.294 µm) — 30 m
//     ST_B10 - Thermal Infrared (10.60–11.19 µm) — 30 m [for LST in Phase 3]
//     QA_PIXEL - Cloud/Quality mask
//
// NOTE: Landsat 9 launched September 27, 2021. No imagery before that date.
//       July 2022 is the first full July available for Landsat 9.
// ============================================================

// ─── 1. DELHI NCT STUDY AREA ─────────────────────────────────────────────
// Load the Delhi state boundary from the LSIB dataset
// LSIB = Large Scale International Boundary Polygons (Simplified)
// We use FAO GAUL (Global Administrative Unit Layers) for subnational boundaries

// Option A: FAO GAUL (Administrative Level 1 = State)
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");
var delhiNCT = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
));

// Verify the boundary was found
var count = delhiNCT.size();
print("Delhi NCT feature count:", count);

// Get the geometry for spatial filtering
var delhiGeom = delhiNCT.geometry();
var delhiArea_km2 = delhiGeom.area(100).divide(1e6);
print("Delhi NCT area (km²):", delhiArea_km2);

// ─── 2. DISPLAY DELHI NCT ON MAP ─────────────────────────────────────────
Map.centerObject(delhiGeom, 10);
Map.addLayer(
  delhiNCT,
  {color: "FF0000", fillColor: "FF000022", width: 2},
  "Delhi NCT Boundary"
);

// ─── 3. LANDSAT 9 COLLECTION DEFINITION ──────────────────────────────────
// Landsat 9 Collection 2 Tier 1 Level-2 (atmospherically corrected SR + ST)
var L9_COLLECTION = "LANDSAT/LC09/C02/T1_L2";

// Bands needed for Phase 2 (raw collection) — LST/NDVI calculated later in Phase 3
var L9_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"];
var L9_BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "TIR", "QA"];

// Scale factors for Landsat Collection 2 Level-2 (USGS standard)
// SR bands: multiply by 0.0000275 and add -0.2 (gives reflectance 0–1)
// ST band:  multiply by 0.00341802 and add 149.0 (gives Kelvin) — Phase 3 only
var SCALE_SR = 0.0000275;
var OFFSET_SR = -0.2;

// Cloud mask function using the QA_PIXEL band
// Bit 3 = Cloud shadow, Bit 4 = Cloud
function maskL9Clouds(image) {
  var qa = image.select("QA_PIXEL");
  // Cloud bit (4) and Cloud shadow bit (3) must both be 0
  var cloudMask = qa.bitwiseAnd(1 << 4).eq(0)  // clear of clouds
    .and(qa.bitwiseAnd(1 << 3).eq(0));          // clear of cloud shadow
  return image.updateMask(cloudMask);
}

// Apply scale factors to surface reflectance bands (NOT TIR — that's for Phase 3)
function applyScaleFactors(image) {
  var srBands = image.select("SR_B.*")
    .multiply(SCALE_SR).add(OFFSET_SR);
  var thermalBand = image.select("ST_B10"); // keep raw thermal — scale in Phase 3
  var qaBand = image.select("QA_PIXEL");
  return ee.Image(srBands.addBands(thermalBand).addBands(qaBand)
    .copyProperties(image, image.propertyNames()));
}

// ─── 4. JULY 2022 — HISTORICAL BASELINE ──────────────────────────────────
print("=== LANDSAT 9 — JULY 2022 (Historical Baseline) ===");

var l9_2022 = ee.ImageCollection(L9_COLLECTION)
  .filterBounds(delhiGeom)
  .filterDate("2022-07-01", "2022-07-31")
  .sort("CLOUD_COVER");  // sort by cloud cover (least cloudy first)

print("Number of Landsat 9 scenes over Delhi NCT, July 2022:", l9_2022.size());

// Print metadata for each scene
var l9_2022_info = l9_2022.map(function(img) {
  return img.set({
    "date":        ee.Date(img.get("DATE_ACQUIRED")).format("YYYY-MM-dd"),
    "cloud_cover": img.get("CLOUD_COVER"),
    "path":        img.get("WRS_PATH"),
    "row":         img.get("WRS_ROW"),
    "image_id":    img.id()
  });
});

// Print the full list with key metadata
print("July 2022 scenes (sorted by cloud cover):");
print(l9_2022_info.aggregate_array("date"));
print("Cloud cover (%):", l9_2022_info.aggregate_array("cloud_cover"));
print("WRS Path:", l9_2022_info.aggregate_array("path"));
print("WRS Row:", l9_2022_info.aggregate_array("row"));
print("Image IDs:", l9_2022_info.aggregate_array("image_id"));

// Best image = least cloud cover
var l9_2022_best = ee.Image(l9_2022.first());
print("Best image (least cloudy) — July 2022:", l9_2022_best);
print("  Date:", l9_2022_best.date().format("YYYY-MM-dd"));
print("  Cloud cover:", l9_2022_best.get("CLOUD_COVER"));
print("  WRS Path/Row:", [l9_2022_best.get("WRS_PATH"), l9_2022_best.get("WRS_ROW")]);

// Display best July 2022 image (True Colour: Red=B4, Green=B3, Blue=B2)
var l9_2022_best_scaled = applyScaleFactors(maskL9Clouds(l9_2022_best));
Map.addLayer(
  l9_2022_best_scaled.clip(delhiGeom),
  {bands: ["SR_B4", "SR_B3", "SR_B2"], min: 0.0, max: 0.3, gamma: 1.4},
  "L9 July 2022 — True Colour (best scene)"
);

// Create a cloud-masked median composite for full Delhi coverage
var l9_2022_composite = l9_2022
  .map(maskL9Clouds)
  .map(applyScaleFactors)
  .median()
  .clip(delhiGeom);

Map.addLayer(
  l9_2022_composite,
  {bands: ["SR_B4", "SR_B3", "SR_B2"], min: 0.0, max: 0.3, gamma: 1.4},
  "L9 July 2022 — Composite (all scenes merged)"
);

// ─── 5. JULY 2026 — CURRENT CONDITION ────────────────────────────────────
print("=== LANDSAT 9 — JULY 2026 (Current Condition) ===");

var l9_2026 = ee.ImageCollection(L9_COLLECTION)
  .filterBounds(delhiGeom)
  .filterDate("2026-07-01", "2026-07-31")
  .sort("CLOUD_COVER");

print("Number of Landsat 9 scenes over Delhi NCT, July 2026:", l9_2026.size());

var l9_2026_info = l9_2026.map(function(img) {
  return img.set({
    "date":        ee.Date(img.get("DATE_ACQUIRED")).format("YYYY-MM-dd"),
    "cloud_cover": img.get("CLOUD_COVER"),
    "path":        img.get("WRS_PATH"),
    "row":         img.get("WRS_ROW"),
    "image_id":    img.id()
  });
});

print("July 2026 scenes (sorted by cloud cover):");
print(l9_2026_info.aggregate_array("date"));
print("Cloud cover (%):", l9_2026_info.aggregate_array("cloud_cover"));
print("WRS Path:", l9_2026_info.aggregate_array("path"));
print("WRS Row:", l9_2026_info.aggregate_array("row"));
print("Image IDs:", l9_2026_info.aggregate_array("image_id"));

var l9_2026_best = ee.Image(l9_2026.first());
print("Best image (least cloudy) — July 2026:", l9_2026_best);
print("  Date:", l9_2026_best.date().format("YYYY-MM-dd"));
print("  Cloud cover:", l9_2026_best.get("CLOUD_COVER"));
print("  WRS Path/Row:", [l9_2026_best.get("WRS_PATH"), l9_2026_best.get("WRS_ROW")]);

var l9_2026_best_scaled = applyScaleFactors(maskL9Clouds(l9_2026_best));
Map.addLayer(
  l9_2026_best_scaled.clip(delhiGeom),
  {bands: ["SR_B4", "SR_B3", "SR_B2"], min: 0.0, max: 0.3, gamma: 1.4},
  "L9 July 2026 — True Colour (best scene)"
);

var l9_2026_composite = l9_2026
  .map(maskL9Clouds)
  .map(applyScaleFactors)
  .median()
  .clip(delhiGeom);

Map.addLayer(
  l9_2026_composite,
  {bands: ["SR_B4", "SR_B3", "SR_B2"], min: 0.0, max: 0.3, gamma: 1.4},
  "L9 July 2026 — Composite (all scenes merged)"
);

// ─── 6. LANDSAT 9 PATH/ROW COVERAGE ──────────────────────────────────────
// Delhi NCT is covered by WRS-2 Path 146, Row 040 (primary)
// and potentially Path 147, Row 040 (secondary)
// The Console output above will confirm the exact paths/rows.
print("=== COVERAGE INFO ===");
print("Expected WRS-2 Path/Row for Delhi NCT:");
print("  Primary: Path 146, Row 040");
print("  Note: Check Console above to confirm actual paths/rows in imagery");

// ─── 7. THERMAL BAND VISUALISATION (False Colour) ─────────────────────────
// Display the raw thermal band (ST_B10) for visual inspection
// NOTE: Do NOT convert to LST here — that is Phase 3 work
var thermal_2022 = l9_2022_best.select("ST_B10").clip(delhiGeom);
Map.addLayer(
  thermal_2022,
  {min: 20000, max: 35000, palette: ["blue", "cyan", "yellow", "orange", "red"]},
  "L9 July 2022 — Raw Thermal Band (ST_B10, raw DN)"
);

// ─── 8. SUMMARY ───────────────────────────────────────────────────────────
print("=== NEXT STEPS ===");
print("1. Review the Console output above for actual image IDs and dates");
print("2. Record image IDs in dataset_metadata.csv");
print("3. Run export_data.js to export data to Google Drive");
print("4. Phase 3 will apply scale factors and calculate LST from ST_B10");
