// ============================================================
// GreenGrid AI — Phase 2: Dataset Collection
// Google Earth Engine Script: Export Data to Google Drive
// ============================================================
//
// PURPOSE:
//   Export Phase 2 satellite data for Delhi NCT to Google Drive,
//   to be downloaded locally into the project data/ folder.
//
// HOW TO USE:
//   1. Run landsat9_collection.js and sentinel2_collection.js first
//   2. Note the actual image IDs/dates from those scripts
//   3. Paste this script into GEE Code Editor and run
//   4. Go to Tasks tab (top-right in GEE) and click RUN for each export
//   5. Choose your Google Drive folder: GreenGridAI_Phase2
//   6. Download files from Drive to your local project:
//        data/raw/landsat9/2022_07/
//        data/raw/landsat9/2026_07/
//        data/raw/sentinel2/2022_07/
//        data/raw/sentinel2/2026_07/
//
// NOTE ON LANDSAT 9 BANDS EXPORTED:
//   SR_B2  Blue (30m)    | SR_B3  Green (30m) | SR_B4  Red (30m)
//   SR_B5  NIR (30m)     | SR_B6  SWIR1 (30m) | SR_B7  SWIR2 (30m)
//   ST_B10 Thermal (30m) | QA_PIXEL            (for cloud masking in Phase 3)
//
// NOTE ON SENTINEL-2 BANDS EXPORTED:
//   B2=Blue(10m), B3=Green(10m), B4=Red(10m), B8=NIR(10m)
//   B5=RedEdge1(20m), B6=RedEdge2(20m), B7=RedEdge3(20m), B8A=NarrowNIR(20m)
//   B11=SWIR1(20m), B12=SWIR2(20m), SCL=Classification(20m)
//
// DO NOT calculate LST, NDVI, or NDBI here — that is Phase 3.
// ============================================================

// ─── DELHI NCT BOUNDARY ───────────────────────────────────────────────────
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");
var delhiNCT = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
));
var delhiGeom = delhiNCT.geometry();

// ─── EXPORT SETTINGS ──────────────────────────────────────────────────────
var DRIVE_FOLDER = "GreenGridAI_Phase2";
var CRS = "EPSG:4326";
var MAX_PIXELS = 1e12; // large value to avoid "too many pixels" error

// ─── CLOUD MASK FUNCTIONS ─────────────────────────────────────────────────
function maskL9Clouds(image) {
  var qa = image.select("QA_PIXEL");
  return image.updateMask(
    qa.bitwiseAnd(1 << 4).eq(0).and(qa.bitwiseAnd(1 << 3).eq(0))
  );
}

function maskS2Clouds(image) {
  var scl = image.select("SCL");
  return image.updateMask(scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)));
}

// ─── SCALE FACTORS (Landsat 9 Collection 2 Level-2) ──────────────────────
function applyL9ScaleFactors(image) {
  var srBands = image.select("SR_B.*").multiply(0.0000275).add(-0.2);
  var thermalBand = image.select("ST_B10"); // raw; scale applied in Phase 3
  var qaBand = image.select("QA_PIXEL");
  return ee.Image(srBands.addBands(thermalBand).addBands(qaBand)
    .copyProperties(image, image.propertyNames()));
}

// ─── 1. LANDSAT 9 — JULY 2022 ────────────────────────────────────────────
var l9_2022_raw = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
  .filterBounds(delhiGeom)
  .filterDate("2022-07-01", "2022-07-31")
  .sort("CLOUD_COVER");

var l9_2022_best = ee.Image(l9_2022_raw.first());
print("L9 2022 best scene:", l9_2022_best.date().format("YYYY-MM-dd"));
print("Cloud cover:", l9_2022_best.get("CLOUD_COVER"));

// Individual best scene export — cast all bands to Float so types are consistent
// (SR bands are Float64 after scaling; ST_B10 and QA_PIXEL are UInt16 — must match)
var l9_2022_export_scene = applyL9ScaleFactors(maskL9Clouds(l9_2022_best))
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"])
  .toFloat()   // cast all bands to Float32 — fixes UInt16 / Float64 mismatch
  .clip(delhiGeom);

Export.image.toDrive({
  image: l9_2022_export_scene,
  description: "L9_2022_07_Delhi_BestScene_30m",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "landsat9_2022_07_best_scene_30m",
  region: delhiGeom,
  scale: 30,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

// Median composite (all July 2022 scenes merged — better spatial coverage)
var l9_2022_composite = l9_2022_raw
  .map(maskL9Clouds)
  .map(applyL9ScaleFactors)
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"])
  .median()
  .clip(delhiGeom);

Export.image.toDrive({
  image: l9_2022_composite,
  description: "L9_2022_07_Delhi_Composite_30m",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "landsat9_2022_07_composite_30m",
  region: delhiGeom,
  scale: 30,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

print("✓ Export tasks queued for Landsat 9 — July 2022");

// ─── 2. LANDSAT 9 — JULY 2026 ────────────────────────────────────────────
var l9_2026_raw = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
  .filterBounds(delhiGeom)
  .filterDate("2026-07-01", "2026-07-31")
  .sort("CLOUD_COVER");

var l9_2026_best = ee.Image(l9_2026_raw.first());
print("L9 2026 best scene:", l9_2026_best.date().format("YYYY-MM-dd"));
print("Cloud cover:", l9_2026_best.get("CLOUD_COVER"));

var l9_2026_export_scene = applyL9ScaleFactors(maskL9Clouds(l9_2026_best))
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"])
  .toFloat()   // cast all bands to Float32 — fixes UInt16 / Float64 mismatch
  .clip(delhiGeom);

Export.image.toDrive({
  image: l9_2026_export_scene,
  description: "L9_2026_07_Delhi_BestScene_30m",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "landsat9_2026_07_best_scene_30m",
  region: delhiGeom,
  scale: 30,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

var l9_2026_composite = l9_2026_raw
  .map(maskL9Clouds)
  .map(applyL9ScaleFactors)
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"])
  .median()
  .clip(delhiGeom);

Export.image.toDrive({
  image: l9_2026_composite,
  description: "L9_2026_07_Delhi_Composite_30m",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "landsat9_2026_07_composite_30m",
  region: delhiGeom,
  scale: 30,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

print("✓ Export tasks queued for Landsat 9 — July 2026");

// ─── 3. SENTINEL-2 — JULY 2022 ───────────────────────────────────────────
var s2_2022_raw = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(delhiGeom)
  .filterDate("2022-07-01", "2022-07-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
  .sort("CLOUDY_PIXEL_PERCENTAGE");

var s2_2022_best = ee.Image(s2_2022_raw.first());
print("S2 2022 best tile:", s2_2022_best.get("MGRS_TILE"));
print("Date:", s2_2022_best.date().format("YYYY-MM-dd"));
print("Cloud %:", s2_2022_best.get("CLOUDY_PIXEL_PERCENTAGE"));

// Export 10-m bands (composite)
var s2_2022_10m = s2_2022_raw
  .map(maskS2Clouds)
  .select(["B2", "B3", "B4", "B8"])  // 10 m native resolution
  .median()
  .divide(10000)                  // scale to 0–1 reflectance
  .clip(delhiGeom);

Export.image.toDrive({
  image: s2_2022_10m,
  description: "S2_2022_07_Delhi_10m_Composite",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "sentinel2_2022_07_10m_composite",
  region: delhiGeom,
  scale: 10,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

// Export 20-m bands (composite)
var s2_2022_20m = s2_2022_raw
  .map(maskS2Clouds)
  .select(["B5", "B6", "B7", "B8A", "B11", "B12", "SCL"])  // 20 m native
  .median()
  .clip(delhiGeom);

// Scale reflectance bands only, cast SCL to float so all bands match type
var s2_2022_20m_scaled = s2_2022_20m.select(["B5", "B6", "B7", "B8A", "B11", "B12"])
  .divide(10000)
  .addBands(s2_2022_20m.select("SCL").toFloat())  // cast UInt8 → Float to match
  .clip(delhiGeom);

Export.image.toDrive({
  image: s2_2022_20m_scaled,
  description: "S2_2022_07_Delhi_20m_Composite",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "sentinel2_2022_07_20m_composite",
  region: delhiGeom,
  scale: 20,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

print("✓ Export tasks queued for Sentinel-2 — July 2022");

// ─── 4. SENTINEL-2 — JULY 2026 ───────────────────────────────────────────
var s2_2026_raw = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(delhiGeom)
  .filterDate("2026-07-01", "2026-07-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
  .sort("CLOUDY_PIXEL_PERCENTAGE");

var s2_2026_best = ee.Image(s2_2026_raw.first());
print("S2 2026 best tile:", s2_2026_best.get("MGRS_TILE"));
print("Date:", s2_2026_best.date().format("YYYY-MM-dd"));
print("Cloud %:", s2_2026_best.get("CLOUDY_PIXEL_PERCENTAGE"));

var s2_2026_10m = s2_2026_raw
  .map(maskS2Clouds)
  .select(["B2", "B3", "B4", "B8"])
  .median()
  .divide(10000)
  .clip(delhiGeom);

Export.image.toDrive({
  image: s2_2026_10m,
  description: "S2_2026_07_Delhi_10m_Composite",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "sentinel2_2026_07_10m_composite",
  region: delhiGeom,
  scale: 10,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

var s2_2026_20m = s2_2026_raw
  .map(maskS2Clouds)
  .select(["B5", "B6", "B7", "B8A", "B11", "B12", "SCL"])
  .median()
  .clip(delhiGeom);

var s2_2026_20m_scaled = s2_2026_20m.select(["B5", "B6", "B7", "B8A", "B11", "B12"])
  .divide(10000)
  .addBands(s2_2026_20m.select("SCL").toFloat())  // cast UInt8 → Float to match
  .clip(delhiGeom);

Export.image.toDrive({
  image: s2_2026_20m_scaled,
  description: "S2_2026_07_Delhi_20m_Composite",
  folder: DRIVE_FOLDER,
  fileNamePrefix: "sentinel2_2026_07_20m_composite",
  region: delhiGeom,
  scale: 20,
  crs: CRS,
  maxPixels: MAX_PIXELS,
  fileFormat: "GeoTIFF"
});

print("✓ Export tasks queued for Sentinel-2 — July 2026");

// ─── 5. SUMMARY ──────────────────────────────────────────────────────────
print("=== ALL EXPORT TASKS QUEUED ===");
print("Go to the 'Tasks' tab in GEE and click RUN for each task.");
print("Files will be saved to Google Drive folder: " + DRIVE_FOLDER);
print("Download them locally to:");
print("  data/raw/landsat9/2022_07/");
print("  data/raw/landsat9/2026_07/");
print("  data/raw/sentinel2/2022_07/");
print("  data/raw/sentinel2/2026_07/");
print("Then run: python3 src/data_collection/validate_satellite_data.py");
